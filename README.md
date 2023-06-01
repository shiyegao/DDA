# Back to the Source: Diffusion-Driven Adaptation to Test-Time Corruption
> [**Back to the Source: Diffusion-Driven Adaptation to Test-Time Corruption**](http://arxiv.org/pdf/2207.03442)            
> Jin Gao*, Jialing Zhang*, Xihui Liu, Trevor Darrell, Evan Shelhamer*, Dequan Wang*       
> *arXiv technical report ([arXiv 2207.03442](http://arxiv.org/abs/2207.03442))*  

## Abstract
Test-time adaptation harnesses test inputs to improve the accuracy of a model trained on source data when tested on shifted target data. Most methods update the source model by (re-)training on each target domain. While re-training can help, it is sensitive to the amount and order of the data and the hyperparameters for optimization. We update the target data instead, and project all test inputs toward the source domain with a generative diffusion model. Our diffusion-driven adaptation (DDA) method shares its models for classification and generation across all domains, training both on source then freezing them for all targets, to avoid expensive domain-wise re-training. We augment diffusion with image guidance and classifier self-ensembling to automatically decide how much to adapt. Input adaptation by DDA is more robust than model adaptation across a variety of corruptions, models, and data regimes on the ImageNet-C benchmark. With its input-wise updates, DDA succeeds where model adaptation degrades on too little data (small batches), on dependent data (correlated orders), or on mixed data (multiple corruptions).

## Introduction
This repo is based on [guided-diffusion](https://github.com/openai/guided-diffusion) and [mim](https://github.com/open-mmlab/mim). We mainly provide the following functionality:
+ Adapt an image using a diffusion model.
+ Test using self-ensemble given image pairs.

## File Structure

The basic file structure is shown as follows:
```
DDA
├── ckpt
│   └── *.pth
├── dataset
│   ├── generated
│   ├── imagenetc
│   └── README.md
├── image_adapt
│   ├── guided_diffusion
│   ├── scripts
│   └── *.py
├── model_adapt
│   ├── configs
│   └── *.py
├── README.md
├── download_ckpt.sh
├── image_adapt.sh
└── test.sh
```

Structure of dataset can be found [here](./dataset/README.md).

## Installation
```bash
conda create -n DDA python=3.8 
conda activate DDA
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install openmim blobfile tqdm pandas cupy_cuda113
conda install mpi4py
mim install mmcv-full 
mim install mmcls
```

## Pre-trained Models

We provide a bash script for easy downloading by just run ```bash download_ckpt.sh```.
If you want to download a certain model, you can find the corresponding ```wget``` command and only run the line.
We also provide the source of such checkpoints, more details of which are hidden in the links as follows.

The pre-trained diffusion model: [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) from [guided-diffusion](https://github.com/openai/guided-diffusion).

The pre-trained recognition model: [mm_models](./ckpt/README.md). 


## Usage

### Diffusion Generation

```bash
bash image_adapt.sh
```

### Ensemble Test

You can choose corruption type/severity in [configs](./model_adapt/configs/_base_/datasets). Ensemble methods can be set according to [args](./model_adapt/test_ensemble.py#L99).


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
| ResNet-50      | 1K/224  | 4.1/25.6  | 76.6          | 18.7 | 24.7 | **29.7** |
| Swin-T         | 1K/224  | 4.5/28.3  | 81.2          | 33.1 | 29.5 | **40.0** |
| ConvNeXt-T     | 1K/224  | 4.5/28.6  | 81.7          | 39.3 | 37.8 | **44.2** |
| Swin-B         | 1K/224  | 15.1/87.8 | 83.4          | 41.0 | 37.0 | **44.5** |
| ConvNeXt-B     | 1K/224  | 15.4/88.6 | 83.9          | 45.6 | 45.8 | **49.4** |

Columns with * are ImageNet-C Acc.

## Citation

If our code or models help your work, please cite our [paper](https://arxiv.org/abs/2207.03442):
```BibTeX
@article{gao2022back,
  title={Back to the Source: Diffusion-Driven Test-Time Adaptation},
  author={Gao, Jin and Zhang, Jialing and Liu, Xihui and Darrell, Trevor and Shelhamer, Evan and Wang, Dequan},
  journal={arXiv preprint arXiv:2207.03442},
  year={2022}
}
```
