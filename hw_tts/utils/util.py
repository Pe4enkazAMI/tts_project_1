import json
import os
import re
import time
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


import numpy as np
import pandas as pd
import textgrid
import torch
from hw_tts.text import _arpabet_to_sequence, _clean_text, text_to_sequence
from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(data_path, mel_ground_truth, alignment_path,
                       pitch_path, energy_path,
                       text_cleaners, batch_expand_size):    
    data_path = str(ROOT_PATH / data_path)
    mel_ground_truth = str(ROOT_PATH / mel_ground_truth)
    alignment_path = str(ROOT_PATH / alignment_path)
    energy_path = str(ROOT_PATH / energy_path)

    buffer = list()
    text = process_text(data_path)

    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)

        dur_gt_name = os.path.join(
            alignment_path, f"{i}.npy")
        duration = np.load(dur_gt_name)

        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, text_cleaners))

        pi_gt_name = os.path.join(
            pitch_path, "ljspeech-pitch-%05d.npy" % (i+1))
        pitch = np.load(pi_gt_name).astype(np.float32)

        en_gt_name = os.path.join(
            energy_path, "ljspeech-energy-%05d.npy" % (i+1))
        energy = np.load(en_gt_name)


        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        pitch = torch.from_numpy(pitch)
        energy = torch.from_numpy(energy)
        mel_gt_target = torch.from_numpy(mel_gt_target)
            
        buffer.append({"text": character, "duration": duration,
                       "pitch": pitch,
                       "energy": energy,
                       "mel_target": mel_gt_target,
                       "batch_expand_size": batch_expand_size})


    return buffer


def get_character_duration(intervals):
    sr = 22050 # for ljspeech
    hop_length = 256
    win_length = 1024
    min_times = []
    max_times = []
    letters = []
    bad_tokens = ['sil', 'sp', '_', '~', '', 'spn']
    for i in range(len(intervals)):
        min_times.append(int(intervals[i].minTime * sr))
        max_times.append(int(intervals[i].maxTime * sr))
        letters.append(intervals[i].mark if intervals[i].mark not in bad_tokens else ' ')
    alignments = np.zeros(len(letters), dtype=int)
    for i in range(len(letters)):
        start = (min_times[i] - win_length) // hop_length + 1
        end = (max_times[i] - win_length) // hop_length + 1
        alignments[i] = end - start
    alignments[-1] += 1
    return '_'.join(letters), alignments


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


def get_WaveGlow():
    waveglow_path = os.path.join("waveglownet", "pretrained_model")
    waveglow_path = os.path.join("/kaggle/input/fastspeecch-dataset",
                                  "waveglow_256channels.pt")
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        out_list = list()
        max_len = mel_max_length
        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    else:
        out_list = list()
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded