# download checkpoints in the 'ckpt' folder
# cd ckpt

# diffusion model
# wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

# recognition model
# download in https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EWmTnvB1cqtIi-OI4HfxGBgBKzO0w_qc3CnErHhNfBitlg?e=XPws5X  # rednet26
wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth  # resnet50
wget https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth  # swinT
wget https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth  # convnextT
wget https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth  # swinB1k
wget https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth  # swinB21k
wget https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128_in1k_20220124-d0915162.pth  # convnextB1k
wget https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_32xb128_in1k_20220124-eb2d6ada.pth  # convnextB21k