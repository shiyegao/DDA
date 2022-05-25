_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/imagenetc_swin_224_val2.py',
    '../_base_/default_runtime.py'
]

# data settings
data = dict(
    samples_per_gpu=64, # samples_per_gpu=64 for 4 gpus
    workers_per_gpu=4,
)
 
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False
        ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))