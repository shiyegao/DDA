_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/models/swin_transformer_base_softmax.py',
    '../_base_/datasets/imagenetc_swin_224_val2.py',
    '../_base_/default_runtime.py'
]
data = dict(
    samples_per_gpu=64, # samples_per_gpu=64 for 4 gpus
    workers_per_gpu=4,
) 