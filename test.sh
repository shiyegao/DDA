export PYTHONPATH=$PYTHONPATH:$(pwd)

# RedNet26
python model_adapt/test_ensemble.py model_adapt/configs/ensemble/rednet26_ensemble_b64_imagenet.py ckpt/rednet26-4948f75f.pth --metrics accuracy --ensemble sum

# Resnet50
python model_adapt/test_ensemble.py model_adapt/configs/ensemble/resnet50_ensemble_b64_imagenet.py ckpt/resnet50_8xb32_in1k_20210831-ea4938fc.pth --metrics accuracy --ensemble sum

# SwinT
python model_adapt/test_ensemble.py model_adapt/configs/ensemble/swinT_ensemble_b64_imagenet.py ckpt/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth --metrics accuracy --ensemble sum

# ConvnextT
python model_adapt/test_ensemble.py model_adapt/configs/ensemble/convnextT_ensemble_b64_imagenet.py ckpt/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth --metrics accuracy --ensemble sum

# SwinB
python model_adapt/test_ensemble.py model_adapt/configs/ensemble/swinB_ensemble_b64_imagenet.py ckpt/swin_base_patch4_window7_224-4670dd19.pth --metrics accuracy --ensemble sum

# ConvnextB
python model_adapt/test_ensemble.py model_adapt/configs/ensemble/convnextB_ensemble_b64_imagenet.py ckpt/convnext-base_3rdparty_32xb128_in1k_20220124-d0915162.pth --metrics accuracy --ensemble sum