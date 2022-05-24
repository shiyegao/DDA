import time
import shutil
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
import pycls.core.distributed as distp
from torch.nn.modules.batchnorm import _BatchNorm

import mmcv
from mmcls.models.losses import accuracy
from mmcv.runner import HOOKS, EvalHook, DistEvalHook, get_dist_info

from .epoch_based_runner import select_logits_to_idx, tackle_img_from_idx, confuse_img_from_logits, ensemble_from_logits


def domain_accuracy_record(results, gt_labels, domains):
    ''' Record domain-specific accuracy. '''

    res = {}
    for d in set(domains):
        d_idx = (np.array(domains)==d)
        accs = accuracy(results[d_idx], gt_labels[d_idx], topk=(1, 5))
        res[d + "-top1"] = accs[0].item()
        res[d + "-top5"] = accs[1].item()
    return res


def copy_dataloader_and_delete(runner):
     for i in range(len(runner.hooks)):
        if runner.hooks[i].__class__.__name__ in ['EvalHook', 'DistEvalHook']:
            dataloader = runner.hooks[i].dataloader
            print('delete', runner.hooks[i].__class__.__name__)
            del runner.hooks[i]
            return dataloader


def single_gpu_test_ensemble(
        model,
        data_loader,
        mode,
        show=False,
        out_dir=None,
        **show_kwargs
    ):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, (data1, data2)  in enumerate(data_loader):
        with torch.no_grad():
            result1 = model(return_loss=False, **data1)
            result2 = model(return_loss=False, **data2)
            if 'sum' in mode:
                result = ensemble_from_logits(result1, result2, mode)
                data = data1
            elif 'fuse' in mode:
                data = confuse_img_from_logits(result1, result2, data1, data2, mode)
                result = model(return_loss=False, **data)
            else:
                result, idx = select_logits_to_idx(result1, result2, mode)
                data = tackle_img_from_idx(data1, data2, idx)

        batch_size = len(result)
        results.extend(result)

        if show or out_dir:
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [model.CLASSES[lb] for lb in pred_label]

            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                result_show = {
                    'pred_score': pred_score[i],
                    'pred_label': pred_label[i],
                    'pred_class': pred_class[i]
                }
                model.module.show_result(
                    img_show,
                    result_show,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test_ensemble(model, data_loader, mode, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

        This method tests model with multiple gpus and collects the results
        under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
        it encodes results to gpu tensors and use gpu communication for results
        collection. On cpu mode it saves the results on different gpus to 'tmpdir'
        and collects them by the rank 0 worker.

        Args:
            model (nn.Module): Model to be tested.
            data_loader (nn.Dataloader): Pytorch data loader for 2 dataset.
            mode (str): criterion for unsemble
            tmpdir (str): Path of directory to save the temporary results from
                different gpus under cpu mode.
            gpu_collect (bool): Option to use either gpu or cpu to collect results.

        Returns:
            list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        # if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
        #     raise OSError((f'The tmpdir {tmpdir} already exists.',
        #                    ' Since tmpdir will be deleted after testing,',
        #                    ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    dist.barrier() 
    for i, (data1, data2) in enumerate(data_loader):
        with torch.no_grad():
            result1 = model(return_loss=False, **data1)
            result2 = model(return_loss=False, **data2)
            if 'sum' in mode:
                result = ensemble_from_logits(result1, result2, mode)
                data = data1
            elif 'fuse' in mode:
                data = confuse_img_from_logits(result1, result2, data1, data2, mode)
                result = model(return_loss=False, **data)
            else:
                result, idx = select_logits_to_idx(result1, result2, mode)
                data = tackle_img_from_idx(data1, data2, idx)

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


@torch.no_grad()
def compute_precise_bn_stats(model, loader):
    """Computes precise BN stats on training data."""
    # Compute the number of minibatches to use
    num_iter = len(loader)
    # Retrieve the BN layers
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    # Initialize BN stats storage for computing mean(mean(batch)) and mean(var(batch))
    running_means = [torch.zeros_like(bn.running_mean) for bn in bns]
    running_vars = [torch.zeros_like(bn.running_var) for bn in bns]
    # Remember momentum values
    momentums = [bn.momentum for bn in bns]
    # Set momentum to 1.0 to compute BN stats that only reflect the current batch
    for bn in bns:
        bn.momentum = 1.0
    # Average the BN stats for each BN layer over the batches
    prog_bar = mmcv.ProgressBar(len(loader.dataset))
    for _, inputs in enumerate(loader):
        result = model(return_loss=False, **inputs)
        batch_size = len(result)
        for i, bn in enumerate(bns):
            running_means[i] += bn.running_mean / num_iter
            running_vars[i] += bn.running_var / num_iter
        for _ in range(batch_size):
            prog_bar.update()
    # Sync BN stats across GPUs (no reduction if 1 GPU used)
    running_means = distp.scaled_all_reduce(running_means)
    running_vars = distp.scaled_all_reduce(running_vars)
    # Set BN stats and restore original momentum values
    for i, bn in enumerate(bns):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


def single_gpu_test_BN(
        model,
        data_loader,
        show=False,
        out_dir=None,
        **show_kwargs
    ):
    results = []
    dataset = data_loader.dataset
    compute_precise_bn_stats(model, data_loader)
    prog_bar = mmcv.ProgressBar(len(dataset))
    model.eval()
    for i, data in enumerate(data_loader):
        with torch.no_grad(): 
            result = model(return_loss=False, **data)

        batch_size = len(result)
        results.extend(result)

        if show or out_dir:
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [model.CLASSES[lb] for lb in pred_label]

            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                result_show = {
                    'pred_score': pred_score[i],
                    'pred_label': pred_label[i],
                    'pred_class': pred_class[i]
                }
                model.module.show_result(
                    img_show,
                    result_show,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test_BN(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
        Args:
            model (nn.Module): Model to be tested.
            data_loader (nn.Dataloader): Pytorch data loader for 2 dataset.
            mode (str): criterion for unsemble
            tmpdir (str): Path of directory to save the temporary results from
                different gpus under cpu mode.
            gpu_collect (bool): Option to use either gpu or cpu to collect results.

        Returns:
            list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        # if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
        #     raise OSError((f'The tmpdir {tmpdir} already exists.',
        #                    ' Since tmpdir will be deleted after testing,',
        #                    ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    dist.barrier() 
    compute_precise_bn_stats(model, data_loader)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


@HOOKS.register_module()
class DomainEvalHook(EvalHook):
    """Non-Distributed evaluation hook.
        Support domain-specific evaluation statistics
    """

    def __init__(self,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=None,
                 less_keys=None,
                 out_dir=None,
                 file_client_args=None,
                 **eval_kwargs):

        if interval <= 0:
            raise ValueError(f'interval must be a positive number, '
                             f'but got {interval}')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')

        self.interval = interval
        self.start = start
        self.by_epoch = by_epoch

        assert isinstance(save_best, str) or save_best is None, \
            '""save_best"" should be a str or None ' \
            f'rather than {type(save_best)}'
        self.save_best = save_best
        self.eval_kwargs = eval_kwargs
        self.initial_flag = True

        if test_fn is None:
            from mmcv.engine import single_gpu_test
            self.test_fn = single_gpu_test
        else:
            self.test_fn = test_fn

        if greater_keys is None:
            self.greater_keys = self._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys, )
            assert is_seq_of(greater_keys, str)
            self.greater_keys = greater_keys

        if less_keys is None:
            self.less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys, )
            assert is_seq_of(less_keys, str)
            self.less_keys = less_keys

        if self.save_best is not None:
            self.best_ckpt_path = None
            self._init_rule(rule, self.save_best)

        self.out_dir = out_dir
        self.file_client_args = file_client_args

    def before_run(self, runner, *args, **kwargs):
        if not hasattr(self, "dataloader"):
            self.dataloader = copy_dataloader_and_delete(runner)
        super().before_run(runner, *args, **kwargs)
   
    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        
        results = self.test_fn(runner.model, self.dataloader)
        domains = [img['domain'] for img in self.dataloader.dataset.data_infos]
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results, domains)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)

    def evaluate(self, runner, results, domains=None):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)

        eval_res.update(
            domain_accuracy_record(
                np.vstack(results),
                self.dataloader.dataset.get_gt_labels(),
                domains
            )
        )

        for name, val in eval_res.items():
            runner.log_buffer.output[name+'_'] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None


@HOOKS.register_module()
class DomainDistEvalHook(DomainEvalHook):
    """Distributed evaluation hook.
        Support domain-specific evaluation statistics
    """

    def __init__(self,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=None,
                 less_keys=None,
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,
                 out_dir=None,
                 file_client_args=None,
                 **eval_kwargs):

        if test_fn is None:
            from mmcv.engine import multi_gpu_test
            test_fn = multi_gpu_test

        super().__init__(
            start=start,
            interval=interval,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            test_fn=test_fn,
            greater_keys=greater_keys,
            less_keys=less_keys,
            out_dir=out_dir,
            file_client_args=file_client_args,
            **eval_kwargs)

        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            domains = [img['domain'] for img in self.dataloader.dataset.data_infos]
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results, domains)
            # the key_score may be `None` so it needs to skip the action to
            # save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)


@HOOKS.register_module()
class EnsembleEvalHook(EvalHook):
    """Non-Distributed evaluation hook.
        Support ensemble evaluation method
    """

    def __init__(self,
                 select='9',
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=None,
                 less_keys=None,
                 out_dir=None,
                 file_client_args=None,
                 **eval_kwargs):

        self.select = select
        if interval <= 0:
            raise ValueError(f'interval must be a positive number, '
                             f'but got {interval}')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')

        self.interval = interval
        self.start = start
        self.by_epoch = by_epoch

        assert isinstance(save_best, str) or save_best is None, \
            '""save_best"" should be a str or None ' \
            f'rather than {type(save_best)}'
        self.save_best = save_best
        self.eval_kwargs = eval_kwargs
        self.initial_flag = True

        self.test_fn = single_gpu_test_ensemble

        if greater_keys is None:
            self.greater_keys = self._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys, )
            assert is_seq_of(greater_keys, str)
            self.greater_keys = greater_keys

        if less_keys is None:
            self.less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys, )
            assert is_seq_of(less_keys, str)
            self.less_keys = less_keys

        if self.save_best is not None:
            self.best_ckpt_path = None
            self._init_rule(rule, self.save_best)

        self.out_dir = out_dir
        self.file_client_args = file_client_args

    def before_run(self, runner, *args, **kwargs):
        if not hasattr(self, "dataloader"):
            self.dataloader = copy_dataloader_and_delete(runner)
        super().before_run(runner, *args, **kwargs)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""

        results = self.test_fn(runner.model, self.dataloader, self.select)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results, domains)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


@HOOKS.register_module()
class EnsembleDistEvalHook(EnsembleEvalHook):
    """Distributed evaluation hook.
        Support ensemble evaluation method
    """

    def __init__(self,
                 select='9',
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=None,
                 less_keys=None,
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,
                 out_dir=None,
                 file_client_args=None,
                 **eval_kwargs):

        super().__init__(
            start=start,
            interval=interval,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            test_fn=test_fn,
            greater_keys=greater_keys,
            less_keys=less_keys,
            out_dir=out_dir,
            file_client_args=file_client_args,
            **eval_kwargs)

        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

        self.select = select
        self.test_fn = multi_gpu_test_ensemble

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            self.select,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            # the key_score may be `None` so it needs to skip the action to
            # save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)


@HOOKS.register_module()
class EnsembleDomainEvalHook(DomainEvalHook):
    """Non-Distributed evaluation hook.
        Support ensemble evaluation method
    """

    def __init__(self, select='9', **eval_kwargs):
        self.select = select
        self.test_fn = single_gpu_test_ensemble

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""

        results = self.test_fn(runner.model, self.dataloader, self.select)
        domains = [img['domain'] for img in self.dataloader.dataset.data_infos]
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results, domains)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)



@HOOKS.register_module()
class EnsembleDomainDistEvalHook(DomainDistEvalHook):
    """Distributed evaluation hook.
        Support ensemble evaluation method
    """

    def __init__(self, select='9', **eval_kwargs):

        super().__init__(**eval_kwargs)
        self.select = select
        self.test_fn = multi_gpu_test_ensemble

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            self.select,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            domains = [img['domain'] for img in self.dataloader.dataset.data_infos]
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results, domains)
            # the key_score may be `None` so it needs to skip the action to
            # save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)



def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
