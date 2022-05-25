_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/imagenetc_swin_224_val2.py',
    '../_base_/default_runtime.py'
]

# data settings
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
)
