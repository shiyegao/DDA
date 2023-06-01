import os
import re
import random
import numpy as np
import copy
import torch

from mmcls.datasets import ImageNet, DATASETS


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_prefix_samples(root, folder_to_idx, extensions, shuffle=False):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (root, path, folder_to_idx[folder_name])
                    samples.append(item)
    if shuffle:
        random.shuffle(samples)
    return samples


@DATASETS.register_module()
class ImageNetC(ImageNet):

    ATTRIBUTE = {
        'corruption': [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur',
            'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ],
        'dataset': 'IMAGENETC'
    }
    
    def __init__(self, 
        corruption, 
        severity, 
        shuffle_shallow=False, 
        shuffle_deep=False,
        shuffle_domain=False,
        shuffle_category=False,
        **kwargs
    ):
        '''
            Args:
                shuffle_shallow: shuffle(15, domain), shuffle(5w, img) in each domain
                shuffle_deep: shuffle(15x5w, img), = train_dataloader's 'shuffle=True' by default
                shuffle_domain: shuffle(750, img) in each category
                shuffle_category: shuffle(5w, img) in each domain
        '''
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        self.corruption, self.severity = corruption, severity
        assert 2 > sum([
            shuffle_shallow > 0,
            shuffle_deep > 0, 
            shuffle_domain > 0,
            shuffle_category > 0,
        ])
        self.shuffle_shallow = shuffle_shallow
        self.shuffle_deep = shuffle_deep
        self.shuffle_domain = shuffle_domain
        self.shuffle_category = shuffle_category
        super().__init__(**kwargs)

    def load_annotations(self):
        load_list = []
        for c in self.corruption:
            for s in self.severity:
                load_list.append((c, s))
        load_list = np.array(load_list)

        if self.shuffle_shallow:
            order = np.random.permutation(len(self.corruption) * len(self.severity))
            load_list = load_list[order]
            print('Shuffling:', load_list)

        samples = []
        for l in load_list:
            c, s = l[0], int(l[1])
            assert s in [1, 2, 3, 4, 5]
            assert c in self.ATTRIBUTE['corruption']
            data_prefix = os.path.join(self.data_prefix, c, str(s))
            if self.ann_file is None:
                folder_to_idx = find_folders(data_prefix)
                sample = get_prefix_samples(
                    data_prefix,
                    folder_to_idx,
                    extensions=self.IMG_EXTENSIONS,
                    shuffle=self.shuffle_shallow or self.shuffle_category
                )
                sample = [ i + (l[0] + l[1],) for i in sample]
                samples += sample
                if len(samples) == 0:
                    raise (RuntimeError('Found 0 files in subfolders of: '
                                        f'{data_prefix}. '
                                        'Supported extensions are: '
                                        f'{",".join(self.IMG_EXTENSIONS)}'))
                self.folder_to_idx = folder_to_idx
            elif isinstance(self.ann_file, str):
                with open(self.ann_file) as f:
                    sample =  [x.strip().split(' ') for x in f.readlines()]
                    sample = [ [data_prefix] + i + [l[0] + l[1]] for i in sample]
                    samples += sample
            else:
                raise TypeError('ann_file must be a str or None')

        if self.shuffle_deep > 0 or self.shuffle_domain:
            step = int(self.shuffle_deep) if self.shuffle_deep > 0 else 1
            print('Shuffling: {} to {}, step {}'.format(0, len(samples), step))
            order = np.random.permutation([i for i in range(len(samples) // step)])
            order = np.array([i * step + j for i in list(order) for j in range(step) ])
            samples = np.array(samples)[order]
        
        if self.shuffle_domain:
            samples = sorted(samples, key=lambda x: x[2])

        self.samples = samples
        print(self.ATTRIBUTE['dataset'], self.corruption, self.severity, len(self.samples))
        
        data_infos = []
        for img_prefix, filename, gt_label, domain in self.samples:
            info = {'img_prefix': img_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['domain'] = domain
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class ImageNetC_2(ImageNetC):

    def __init__(self, data_prefix2, **kwargs):
        super().__init__(**kwargs)

        # for second dataset
        self.data_infos = self.data_infos
        self.data_prefix = data_prefix2
        self.data_infos2 = self.load_annotations()

    def prepare_data2(self, idx):
        results = copy.deepcopy(self.data_infos2[idx])
        return self.pipeline(results)

    def __getitem__(self, idx):
        x1 = self.prepare_data(idx)
        x2 = self.prepare_data2(idx)
        return x1, x2
