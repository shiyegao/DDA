import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from image_adapt.guided_diffusion import dist_util, logger
from image_adapt.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from image_adapt.guided_diffusion.image_datasets import load_data
from torchvision import utils
from image_adapt.resizer import Resizer
import math


# added
def load_reference(data_dir, batch_size, image_size, class_cond=False, corruption="shot_noise", severity=5,):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
        corruption=corruption,
        severity=severity,
    )
    for large_batch, model_kwargs, filename in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs, filename


def main():
    args = create_argparser().parse_args()

    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.D, 2).is_integer()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.D), int(args.image_size / args.D))
    down = Resizer(shape, 1 / args.D).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.D).to(next(model.parameters()).device)
    resizers = (down, up)

    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        corruption=args.corruption,
        severity=args.severity,
    )

    logger.log("creating samples...")
    count = 0
    while count * args.batch_size < args.num_samples:
        model_kwargs, filename = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=model_kwargs["ref_img"],
            resizers=resizers,
            M=args.M,
            N=args.N,
        )

        for i in range(args.batch_size):
            path = os.path.join(logger.get_dir(), args.corruption, str(args.severity), filename[i].split('/')[0])
            os.makedirs(path, exist_ok=True)
            out_path = os.path.join(path, filename[i].split('/')[1])

            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

        count += 1
        logger.log(f"created {count * args.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
        D=32, # scaling factor
        M=0, # refinement range
        N=50,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        save_latents=False,
        corruption="shot_noise",
        severity=5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()