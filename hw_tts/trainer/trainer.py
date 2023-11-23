import random
from pathlib import Path
from random import shuffle
from typing import Optional
import PIL
import torch
from hw_tts.base import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import ROOT_PATH, MetricTracker, inf_loop, get_WaveGlow
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
from waveglownet.inference import get_wav



class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        print(device)
        super().__init__(model, criterion, optimizer, lr_scheduler, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = self.config["trainer"].get("log_step", 50)
        self.batch_accum_steps = self.config["trainer"].get("batch_accum_steps", 1)
        self.batch_expand_size = self.config["trainer"]["batch_expand_size"]
        self.WaveGlow = get_WaveGlow()
        self.WaveGlow.to(device)

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "pitch_loss",
            "energy_loss", "grad norm", writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        names = ["src_seq", "gt_mel", "gt_duration", "gt_energy",
                 "mel_pos", "src_pos", "gt_pitch"]
        for tensor_for_gpu in names:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        progress_bar = tqdm(range(self.len_epoch), desc='train')

        for list_batch_idx, list_batch in enumerate(self.train_dataloader):
            stop = False
            for batch_idx, batch in enumerate(list_batch):
                progress_bar.update(1)
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics,
                        index=batch_idx,
                        total=self.len_epoch,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                self.train_metrics.update("grad norm", self.get_grad_norm())
                full_batch_idx = batch_idx + list_batch_idx * self.batch_expand_size
                if full_batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + full_batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(full_batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    
                    self._log_scalars(self.train_metrics)
                    # self._log_predictions(**batch)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                if full_batch_idx + 1 >= self.len_epoch:
                    stop = True
                    break
            if stop:
                break
        log = last_train_metrics
        self._log_predictions(**batch)
        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker,
                      index: Optional[int] = None, total: Optional[int] = None):
        if (index + 1) % self.batch_accum_steps == 0 or index + 1 == total:
            self.optimizer.zero_grad()
        
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["mel_output"] = outputs

        if is_train:
            mel_loss, duration_loss, energy_loss, pitch_loss = self.criterion(**batch)
            batch["mel_loss"] = mel_loss
            batch["duration_loss"] = duration_loss
            batch["pitch_loss"] = pitch_loss
            batch["energy_loss"] = energy_loss
            batch["loss"] = mel_loss + duration_loss + pitch_loss + energy_loss
            batch["loss"].backward()
            if (index + 1) % self.batch_accum_steps == 0 or index + 1 == total:
                self._clip_grad_norm()
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        if is_train:
            metrics.update("loss", batch["loss"].item())
            metrics.update("mel_loss", batch["mel_loss"].item())
            metrics.update("duration_loss", batch["duration_loss"].item())
            metrics.update("pitch_loss", batch["pitch_loss"].item())
            metrics.update("energy_loss", batch["energy_loss"].item())
        else:
            metrics.update("loss", 0) # we do not count loss in eval mode

        return batch
    
    @torch.inference_mode()
    def _synthesis(self, mel):
        mel = mel.contiguous().transpose(-1, -2).unsqueeze(0)
        audio = get_wav(mel, self.WaveGlow)
        return audio
        
    def _log_predictions(
            self,
            src_seq,
            gt_mel,
            mel_pred,
            gt_pitch,
            gt_energy,
            mel_pos,
            src_pos,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

        tuples = list(zip(src_seq, gt_mel, mel_pred))
        shuffle(tuples)
        for __, _, mel_output in tuples[:examples_to_log]:
            audio = self._synthesis(mel_output)
            self._log_audio(audio, 22050, "Train Synt")

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, audio, sr, name):
        self.writer.add_audio(f"Audio_{name}", audio, sample_rate=sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))