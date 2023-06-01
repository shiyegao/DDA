# dataset settings
dataset_type = 'ImageNetC_2'
corruption = ['gaussian_noise', 'shot_noise', 'impulse_noise', 
    'defocus_blur', 'glass_blur', 'motion_blur', 
    'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
corruption = corruption[9]
severity = 5

data_prefix1 = 'dataset/imagenetc'
data_prefix2 = 'dataset/generated'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix1,
        data_prefix2=data_prefix2,
        pipeline=train_pipeline,
        corruption=corruption,
        severity=severity),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix1,
        data_prefix2=data_prefix2,
        pipeline=test_pipeline,
        corruption=corruption,
        severity=severity),
    test=dict(
        type=dataset_type,
        data_prefix=data_prefix1,
        data_prefix2=data_prefix2,
        pipeline=test_pipeline,
        corruption=corruption,
        severity=severity))
evaluation = dict(interval=1, metric='accuracy')