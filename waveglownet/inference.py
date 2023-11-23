import torch

def get_wav(mel, waveglow, sigma=1.0, sampling_rate=22050):
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)
        audio = audio 
    audio = audio.squeeze()
    audio = audio.cpu()

    return audio