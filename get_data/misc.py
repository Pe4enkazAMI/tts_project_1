import os
import shutil
from pathlib import Path
import numpy as np
import pyworld as pw
import torch
import torchaudio
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from ctypes import alignment
import numpy as np
import textgrid
import re
import nltk


import numpy as np

def get_energy():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    save_dir = data_dir / "data" / "energy"
    mel_dir = data_dir / "data" / "mels"
    save_dir.mkdir(exist_ok=True, parents=True)

    assert mel_dir.exists(), "Mel dir not found, download data first"

    min_energy = 1e10
    max_energy = -1e10

    for fpath in mel_dir.iterdir():
        mel = np.load(fpath)
        energy = np.linalg.norm(mel, axis=-1)
        new_name = fpath.name.replace('mel', 'energy')
        np.save(save_dir / new_name, energy)     

        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())

    print('min:', min_energy, 'max:', max_energy) # pass this values to config


def get_pitch():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    wav_dir = data_dir / "data" / "LJSpeech-1.1" / "wavs"
    mel_dir = data_dir / "data" / "mels"
    save_dir = data_dir / "data" / "pitch"
    save_dir.mkdir(exist_ok=True, parents=True)

    assert wav_dir.exists(), "Wav dir not found, download data first"
    
    names = []
    for fpath in wav_dir.iterdir():
        names.append(fpath.name)

    names_dict = {name: i for i, name in enumerate(sorted(names))}

    min_pitch = 1e10
    max_pitch = 1e-10

    for fpath in tqdm(wav_dir.iterdir(), total=len(names)):
        real_i = names_dict[fpath.name]
        new_name = "ljspeech-pitch-%05d.npy" % (real_i+1)
        mel_name = "ljspeech-mel-%05d.npy" % (real_i+1)
        mel = np.load(mel_dir / mel_name)
        audio, sr = torchaudio.load(fpath)
        audio = audio.to(torch.float64).numpy().sum(axis=0)
        frame_period = (audio.shape[0] / sr * 1000) / mel.shape[0]
        _f0, t = pw.dio(audio, sr, frame_period=frame_period)
        f0 = pw.stonemask(audio, _f0, t, sr)[:mel.shape[0]]
        nonzeros = np.nonzero(f0)
        x = np.arange(f0.shape[0])[nonzeros]
        values = (f0[nonzeros][0], f0[nonzeros][-1])
        f = interp1d(x, f0[nonzeros], bounds_error=False, fill_value=values)
        new_f0 = f(np.arange(f0.shape[0]))
        np.save(save_dir / new_name, new_f0)
        min_pitch = min(min_pitch, new_f0.min())
        max_pitch = max(max_pitch, new_f0.max())

    print('min:', min_pitch, 'max:', max_pitch)


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

def process():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    alignment_path = data_dir / 'alignments'
    alignment_path_text = data_dir / 'aling' / 'text'
    alignment_path_npy = data_dir / 'aling' / 'npy'
    alignment_path_text.mkdir(exist_ok=True, parents=True)
    alignment_path_npy.mkdir(exist_ok=True, parents=True)

    for i, fpath in tqdm(enumerate(alignment_path.iterdir())):
        print(fpath)
        duration = textgrid.TextGrid.fromFile(str(fpath))[1]
        character, duration = get_character_duration(duration)
        np.save(alignment_path_npy / f'{fpath.name[:-9]}.npy', duration)
        with open(str(alignment_path_text / f'{fpath.name[:-9]}.txt'), 'w') as f:
            f.write(character)


def phonemizer():
    nltk.download('cmudict')
    corpus = nltk.corpus.cmudict.dict()

    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]

    corpus['kantorovich'] = ['K AA0 N T AO R AH V IH1 CH'.split()] # add word to lexicon
    
    phoneme_tests = []

    for line in tests:
        line = line.lower()
        phoneme_line = []
        for word in line.split():
            word = re.sub(r'[^\w\s]', '', word)
            phoneme = corpus[word][0]
            phoneme_line.extend(phoneme)
        phoneme_line = ' '.join(phoneme_line)
        phoneme_tests.append(phoneme_line)

    for line in phoneme_tests:
        print(line)


if __name__ == '__main__':
    get_energy()
    get_pitch()
    process()
    phonemizer()