_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/imagenetc_ReCr2.py',
    '../_base_/default_runtime.py'
]

# data settings
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        cal_acc=True,
        topk=(1, 5),
    )
)