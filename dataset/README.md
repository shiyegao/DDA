## ImageNet-C

After you download [ImageNet-C](https://github.com/hendrycks/robustness) dataset, you can use ```ln -s [source] dataset/imagenetc``` for convenience. The generated images will be stored in the 'dataset/generated' folder, in the same structure of 'dataset/imagenetc' for further ensemble test.

The detailed file structure is shown as follows:
```
DDA
├── dataset
│   ├── imagenetc
│   │   ├── gaussian_noise
│   │   │   ├── 1
│   │   │   ├── 2
│   │   │   ├── 3
│   │   │   ├── 4
│   │   │   └── 5
│   │   │       ├── n01440764
│   │   │       │   ├── ILSVRC2012_val_00000293.JPEG
│   │   │       │   └── ...
│   │   │       └── ...
│   │   └── ...
│   └── generated
│       ├── gaussian_noise
│       │   ├── 1
│       │   ├── 2
│       │   ├── 3
│       │   ├── 4
│       │   └── 5
│       │       ├── n01440764
│       │       │   ├── ILSVRC2012_val_00000293.JPEG
│       │       │   └── ...
│       │       └── ...
│       └── ...
└── ...
```