# DDA
Official repository of "Back to Source: Diffusion-Driven Test-Time Adaptation".

## Introduction
This repo is based on [guided-diffusion](https://github.com/openai/guided-diffusion) and [mim](https://github.com/open-mmlab/mim). We mainly provide the following functionality:
+ Adapt an image using a diffusion model.
+ Test using self-ensemble given image pairs.

## File_structure

```
DDA
    |---- image_adapt
    |---- model_adapt
        |---- configs
    |---- README.md
```

### Installation
```bash
conda create -n DDA python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install openmim cupy_cuda113
mim install mmcv-full 
mim install mmcls
```

### Usage

The official usage is 
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=0 python tools/test_ensemble.py model_adapt/configs/rednet26_select_b64_imagenet.py --metrics accuracy --select l2norm
```
