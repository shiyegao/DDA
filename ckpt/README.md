## Checkpoint

### Diffusion Model
The pre-trained diffusion model: [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) from [guided-diffusion](https://github.com/openai/guided-diffusion)

### Recognition Model
You can find configs and checkpoints of recognition models in [mmclassification](https://github.com/open-mmlab/mmclassification/tree/master/configs).

Specifically, the six models in our paper are shown as follows:

|      Model      |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------:|:------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| RedNet-26       | From scratch |  9.23     | 1.73    | 75.96 | 93.19 | [config](https://github.com/d-li14/involution/blob/main/cls/configs/rednet/rednet26_b32x64_warmup_coslr_imagenet.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EWmTnvB1cqtIi-OI4HfxGBgBKzO0w_qc3CnErHhNfBitlg?e=XPws5X) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EVJ_eDMSsr1JqhInx67OCxcB-P54pj3o5mGO_rYVsRSk3A?e=70tJAc) |
| ResNet-50      | From scratch  | 25.56     | 4.12     | 76.55 | 93.06 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.log.json) |
|  Swin-T        | From scratch |   28.29   |    4.36   |   81.18   |   95.61   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-tiny_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925.log.json)|
| ConvNeXt-T    | From scratch | 28.59 | 4.46 | 82.05 | 95.86  | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-tiny_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth) |
|  Swin-B       | From scratch |   87.77   |   15.14   |   83.36   |   96.44   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742.log.json)|
|  Swin-B       | ImageNet-21k |   87.77   |   15.14   |   85.16   |   97.50   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-base_16xb64_in1k.py)| [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth)|
| ConvNeXt-B    | From scratch | 88.59 | 15.36 | 83.85 | 96.74 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-base_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128_in1k_20220124-d0915162.pth) |
| ConvNeXt-B    | ImageNet-21k | 88.59 | 15.36 | 85.81 | 97.86 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-base_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_32xb128_in1k_20220124-eb2d6ada.pth) |
