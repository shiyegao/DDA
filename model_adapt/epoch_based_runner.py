import os.path as osp
import warnings
import platform
import shutil
import time
import warnings
import numpy as np
import torch

import mmcv
from mmcv.runner import BaseRunner, RUNNERS, save_checkpoint, get_host_info


@torch.no_grad()
def select_logits_to_idx(x1: list, x2: list, mode: str):
    logits1 = torch.from_numpy(np.vstack(x1))
    logits2 = torch.from_numpy(np.vstack(x2))
    num = logits1.shape[0]
    if mode == 'first':
        idx = torch.zeros(num, 1)
    elif mode == 'second':
        idx = torch.ones(num, 1)
    elif mode == 'entropy':
        ent1 = - (logits1.softmax(1) * logits1.log_softmax(1)).sum(1, keepdim=True)
        ent2 = - (logits2.softmax(1) * logits2.log_softmax(1)).sum(1, keepdim=True)
        idx = ent1>=ent2
    elif mode == 'confidence':
        con1 = logits1.softmax(1).max(1, keepdim=True)[0]
        con2 = logits2.softmax(1).max(1, keepdim=True)[0]
        idx = con1<=con2
    elif mode == 'l2norm':
        n1 = logits1.norm(p=2, dim=1, keepdim=True)
        n2 = logits2.norm(p=2, dim=1, keepdim=True)
        idx = n1<=n2
    elif mode == 'var':
        v1 = logits1.var(dim=1, keepdim=True)
        v2 = logits2.var(dim=1, keepdim=True)
        idx = v1<=v2
    else:
        raise NotImplementedError(f"No such select mode {mode}")
    
    # generate the selected logits
    logits = logits1 * idx.logical_not() + logits2 * idx
    logits = logits.numpy()
    return [logits[i] for i in range(logits.shape[0])], idx


@torch.no_grad()
def tackle_img_from_idx(img1: dict, img2: dict, idx: torch.tensor) -> dict:
    '''img: dict(
        'img': torch.tensor(128, 3, 224, 224), 
        'gt_labels': torch.tensor(128),
        'img_metas'.data[0]: list(128 dicts):
            dict_keys(['filename', 'ori_filename', 'ori_shape', 'img_shape', 'img_norm_cfg', 'ori_img'])
        )
    '''
    ith = idx.clone()
    while img1['img'].dim() > ith.dim():
        ith = ith.unsqueeze(-1)
    img1['img'] = img1['img'] * ith.logical_not() + img2['img'] * ith
    for i in range(idx.shape[0]):
        if idx[i]==1:
            img1['img_metas'].data[0][i] = img2['img_metas'].data[0][i]
    return img1


@torch.no_grad()
def confuse_img_from_logits(x1: list, x2: list, img1: dict, img2: dict, mode: str) -> dict:
    logits1 = torch.from_numpy(np.vstack(x1))
    logits2 = torch.from_numpy(np.vstack(x2))
    if mode == 'entropy_fuse':
        ent1 = - (logits1.softmax(1) * logits1.log_softmax(1)).sum(1, keepdim=True)
        ent2 = - (logits2.softmax(1) * logits2.log_softmax(1)).sum(1, keepdim=True)
        w = torch.cat((ent2, ent1), 1).softmax(1)
    elif mode == 'confidence_fuse':
        con1 = logits1.softmax(1).max(1, keepdim=True)[0]
        con2 = logits2.softmax(1).max(1, keepdim=True)[0]
        w = torch.cat((con1, con2), 1).softmax(1)
    elif mode == 'l2norm_fuse':
        n1 = logits1.norm(p=2, dim=1, keepdim=True)
        n2 = logits2.norm(p=2, dim=1, keepdim=True)
        w = torch.cat((n1, n2), 1).softmax(1)
    elif mode == 'var_fuse':
        v1 = logits1.var(dim=1, keepdim=True)
        v2 = logits2.var(dim=1, keepdim=True)
        w = torch.cat((v1, v2), 1).softmax(1)
    else:
        raise NotImplementedError(f"No such select mode {mode}")
    
    # generate the selected logits
    while img1['img'].dim() > w.dim():
        w = w.unsqueeze(-1)
    img1['img'] = img1['img'] * w[:, :1] + img2['img'] * w[:, 1:]
    return img1


def  ensemble_from_logits(x1: list, x2: list, mode: str):
    logits1 = torch.from_numpy(np.vstack(x1))
    logits2 = torch.from_numpy(np.vstack(x2))
    if mode == 'sum':
        logits = logits1 + logits2
    elif mode == 'entropy_sum':
        ent1 = - (logits1.softmax(1) * logits1.log_softmax(1)).sum(1, keepdim=True)
        ent2 = - (logits2.softmax(1) * logits2.log_softmax(1)).sum(1, keepdim=True)
        logits = logits1 * ent2  + logits2 * ent1
    elif mode == 'confidence_sum':
        con1 = logits1.softmax(1).max(1, keepdim=True)[0]
        con2 = logits2.softmax(1).max(1, keepdim=True)[0]
        logits = logits1 * con1  + logits2 * con2
    else:
        raise NotImplementedError(f"No such select mode {mode}")
    logits = logits.numpy()
    return [logits[i] for i in range(logits.shape[0])]


@RUNNERS.register_module()
class epochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.data_batch = data_batch
            self.runner_kwargs = kwargs
            self.train_mode = True
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=self.train_mode, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.data_batch = data_batch
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class epochBasedRunnerEnsemble(epochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    
    def __init__(self, select, *args, **kwargs):
        self.select = select
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def select_img(self, img1, img2):
        logits1 = self.model(return_loss=False, **img1)
        logits2 = self.model(return_loss=False, **img2)
        if 'fuse' in self.select:
            warnings.warn("In image fusion, img['metas'] has NOT been changed! Thus online may be INCORRECT!")
            img = confuse_img_from_logits(logits1, logits2, img1, img2, self.select)
        else:
            _, idx = select_logits_to_idx(logits1, logits2, self.select)
            img = tackle_img_from_idx(img1, img2, idx)
        return img

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, (data_batch1, data_batch2) in enumerate(self.data_loader):
            self._inner_iter = i
            self.data_batch = self.select_img(data_batch1, data_batch2)

            self.runner_kwargs = kwargs
            self.train_mode = True
            self.call_hook('before_train_iter')
            self.run_iter(self.data_batch, train_mode=self.train_mode, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, (data_batch1, data_batch2) in enumerate(self.data_loader):
            self._inner_iter = i
            self.data_batch = self.select_img(data_batch1, data_batch2)
            self.call_hook('before_val_iter')
            self.run_iter(self.data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')
