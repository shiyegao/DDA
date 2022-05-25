

CUDA_VISIBLE_DEVICES=0 python model_adapt/test_ensemble.py model_adapt/configs/ensemble/rednet26_ensemble_b64_imagenet.py ckpt/rednet26-4948f75f.pth --metrics accuracy --select l2norm