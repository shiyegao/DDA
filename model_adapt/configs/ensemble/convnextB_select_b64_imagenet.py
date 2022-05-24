_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/imagenetc25_bs64_swin_224_val2.py',
    '../_base_/default_runtime.py'
]

# data settings
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
)
