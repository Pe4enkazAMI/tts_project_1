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

    
@torch.inference_mode()
def inference(model, texts, wave_glow, device, **kwargs):
    model.eval()
    t = text_to_sequence(texts[1], ["english_cleaners"])
    t_len = len(t) 
    t_pos = list(np.pad([i+1 for i in range(int(t_len))], (0, 0), 'constant'))
    t_pos = torch.from_numpy(np.array(t_pos))
    mel_out = model(src_seq=torch.tensor(t).to(device).unsqueeze(0),
                    src_pos=t_pos.to(device).unsqueeze(0), 
                    alpha=kwargs["speed"], beta=kwargs["pitch"], gamma=kwargs["energy"])["mel_output"]
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

    testing = [ 
        "I love Sofia Kostina to the moon and back",
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]
    speeds = [0.8, 1, 1.2]
    pithces = [0.8, 1, 1.2]
    energies = [0.8, 1, 1.2]
    with torch.no_grad():
        for text in enumerate(tqdm(testing)):
            for speed in speeds:
                for pitch in pithces:
                    for energy in energies:
                        filename = f'{text[:5]}_{speed}_{pitch}_{energy}'

                        audio = inference(model,
                                            text,
                                            WaveGlow,
                                            device, 
                                            speed=speed, 
                                            pitch=pitch, 
                                            energy=energy)
                        path_to_save = Path(f"/kaggle/working/{filename}.wav")                
                        torchaudio.save(path_to_save, audio.unsqueeze(0), sample_rate=22050)
        
        

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
    args.add_argument(
        '-tts',
        '--test_texts',
        default="FUCK ME HARD",
        type=str,
        help="Text to generate"

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