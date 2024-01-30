# FastSpeech2

## Installation guide

Installation guide is pretty simple.

Minimal requirements sufficient to run the code are just pytorch and the following libraries, make sure to run the following code
```shell
pip install wandb
pip install textgrid
pip install librosa
pip install pyworld
pip install inflect
```

Weights can be downloaded here https://www.kaggle.com/datasets/yanmaximov/myanchik

To run the inference please run the following code 

```shell
python test.py -r WEIGHTS -c CONFIG
```

## Details

This is just a fine implementation of FastSpeech2 model, without unnecessary details and things. 

During project author tried to study optimization, however it did not workout. Though, the generations are pretty good. 
