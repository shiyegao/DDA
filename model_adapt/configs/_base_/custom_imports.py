custom_imports = dict(imports=[
    'inherit.image_classifier',
    'inherit.linear_cls_heads',
    'inherit.conv_module',
    'new.softmax_entropy_loss',
    'new.cifar',
    'new.imagenet',
    'new.tent',
    'new.rednet',
    'new.evaluation',
    'new.epoch_based_runner',
    'new.pipeline',
], allow_failed_imports=False)