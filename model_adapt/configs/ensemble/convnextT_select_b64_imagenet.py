_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/imagenetc25_bs32_ReCr2.py',
    '../_base_/default_runtime.py'
]

# data settings
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
)

# model settings
model = dict(
    type='imageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=(3, ),
        drop_path_rate=0.1,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]),
    head=dict(
        type='linearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))