import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_tts.model as module_model
from hw_tts.logger import get_visualizer
from hw_tts.trainer import Trainer
from hw_tts.utils import ROOT_PATH, MetricTracker, get_WaveGlow
from hw_tts.utils.object_loading import get_dataloaders
from hw_tts.utils.parse_config import ConfigParser
from hw_tts.text import text_to_sequence
from waveglownet.inference import get_wav
import numpy as np
import torchaudio

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def make_src_pos_for_inference(texts):
        length_text = np.array([])
        length_text = np.append(length_text, texts.size(0))
        src_pos = list()
        max_len = int(max(length_text))
        for length_src_row in length_text:
            src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
        src_pos = torch.from_numpy(np.array(src_pos))
        return {"src_seq_inference": texts, "src_pos_inference": src_pos}
    
@torch.inference_mode()
def inference(model, texts, wave_glow):
    model.eval()
    print(texts)
    t = text_to_sequence(texts[1], ["english_cleaners"])
    inference_batch = make_src_pos_for_inference(t)
    mel_out = model(src_seq=inference_batch["src_seq_inference"],
                        src_pos=inference_batch["src_pos_inference"])["mel_output"]
    mel = mel_out[0, ...]
    mel = mel.contiguous().transpose(-1, -2).unsqueeze(0)
    audio = get_wav(mel, wave_glow)
    return audio


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    print(device)
    model = model.to(device)
    model.eval()
    WaveGlow = get_WaveGlow()
    WaveGlow.to(device)


    with torch.no_grad():
        audios = []
        for text in enumerate(tqdm(config["test_texts"])):
            audio = inference(model, text, WaveGlow)
            audios += [audio]

    i = 0
    for audio in audios:
        path_to_save = Path(f"/kaggle/working/{i}.wav")
        torchaudio.save(path_to_save, audio, sample_rate=22050)
            
        

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "mix_dir": str(test_data_folder / "mix"),
                            "ref_dir": str(test_data_folder / "refs"),
                            "target_dir": str(test_data_folder / "targets")
                        },
                    }
                ],
            }
        }

    # assert config.config.get("data", {}).get("test", None) is not None
    # config["data"]["test"]["batch_size"] = args.batch_size
    # config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)