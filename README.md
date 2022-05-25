# DDA
Official repository of "Back to Source: Diffusion-Driven Test-Time Adaptation".

## Introduction
This repo is based on [guided-diffusion](https://github.com/openai/guided-diffusion) and [mim](https://github.com/open-mmlab/mim). We mainly provide the following functionality:
+ Adapt an image using a diffusion model.
+ Test using self-ensemble given image pairs.

## File_structure

The basic file structure is shown as follows:
```
DDA
    |---- dataset
        |---- imagenetc
        |---- generated
    |---- ckpt
        |---- *.pth
    |---- image_adapt
        |---- guided_diffusion
        |---- scripts
    |---- model_adapt
        |---- configs
    |---- README.md
```

Structure of dataset can be found [here](./dataset/README.md).

### Installation
```bash
conda create -n DDA python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install openmim cupy_cuda113
mim install mmcv-full 
mim install mmcls
```

You can find how to download checkpoint [here](.ckpt/README.md). 

### Usage


```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python model_adapt/test_ensemble.py model_adapt/configs/ensemble/rednet26_ensemble_b64_imagenet.py ckpt/rednet26-4948f75f.pth --metrics accuracy --select sum
```
