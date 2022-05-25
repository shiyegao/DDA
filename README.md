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

## Installation
```bash
conda create -n DDA python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install openmim cupy_cuda113
mim install mmcv-full 
mim install mmcls
```

## Pre-trained Models
You can find how to download checkpoint [here](./ckpt/README.md). 

## Usage

### Diffusion Generation


### Ensemble Test

You can choose corruption type/severity in [configs](./model_adapt/configs/_base_/datasets). Ensemble methods can be changed in [args](./model_adapt/test_ensemble.py#L99).


The basic command form is 
```bash
python model_adapt/test_ensemble.py [config] [checkpoint] --metrics accuracy --ensemble [ensemble method]
```

Or you can just run
```bash
bash test.sh
```


## Results

| Architecture    | Data/Size   | Params/FLOPs| ImageNet Acc. |Source-Only* | MEMO* | DDA*|
|:---------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----:|:---:|
| RedNet-26      | 1K/224^2  | 1.7/9.2   | 76.0          | 15.0 | 20.6 | **25.0** |
| ResNet-50      | 1K/224^2  | 4.1/25.6  | 76.6          | 18.7 | 24.7 | **27.3** |
| Swin-T         | 1K/224^2  | 4.5/28.3  | 81.2          | 33.1 | 29.5 | **37.0** |
| ConvNeXt-T     | 1K/224^2  | 4.5/28.6  | 81.7          | 39.3 | 37.8 | **41.4** |
| Swin-B         | 1K/224^2  | 15.1/87.8 | 83.4          | 41.0 | 37.0 | **42.0** |
| ConvNeXt-B     | 1K/224^2  | 15.4/88.6 | 83.9          | 45.6 | 45.8 | **46.1** |

Colomns with * are ImageNet-C Acc.