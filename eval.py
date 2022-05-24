"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os
import os.path as osp

import numpy as np
import torch.distributed as dist
import torch as th
from model.set_diffusion import dist_util, logger
from model.set_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from dataset import create_loader
from model import select_model
from utils.util import count_params, set_seed
from utils.path import set_folder

DIR=set_folder()

def main():
    args = create_argparser().parse_args()
    print(args)
    # dct = vars(args)
    # for k in sorted(dct):
    #     print(k, dct[k])
    # print()  
    
    # dist_util.setup_dist()
    logger.configure(dir=DIR, mode="-".join(["evaluation", args.mode_evaluation]), args=args, tag="")

    logger.log("creating model and diffusion...")
    model = select_model(args)(args)
    print(count_params(model))

    #print(args.model_path)
    _path = list(args.model_path.split("/"))
    for j in range(len(_path)):
        _p = list(_path[j].split("_"))
        _p = [ i for i in _p if i not in ["", None, "None", "none"] ]
        _path[j] = "_".join(_p)
    #_path = list(args.model_path.split("_"))
    #_path = [ i for i in _path if i not in ["", None, "None", "none"] ]
    model_path = "/".join(_path)
    
    model.load_state_dict(
        dist_util.load_state_dict(osp.join(DIR, model_path), map_location="cpu")
    )
    model.to(args.device)
    model.eval()

    logger.log("creating data loader...")

    # args.dataset = ""
    args.transfer=False
    if args.transfer:
        if args.image_size == 32:
            args.dataset = "minimagenet"
        else:
            args.dataset = "cub"

    if args.mode_evaluation == "out-distro":
        loader = create_loader(args, split="test", shuffle=True)
    else:
        loader = create_loader(args, split="train", shuffle=True)

    logger.log("evaluating...")
    run_bpd_evaluation(model, loader, args.num_samples, args.clip_denoised, args)


def run_bpd_evaluation(model, data, num_samples, clip_denoised, args):
    all_bpd = []
    all_mse = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    while num_complete < num_samples:
        # iterate loader
        batch = next(data)
        try:
            batch = next(data)
        except StopIteration:
            data = iter(data)
            batch = next(data)

        x_set = batch[:, :5]
        x_set = x_set.to(args.device)
        x_p = batch[:, 5:]
        x_p = x_p.to(args.device)
        # build c
        if args.model == "ddpm":
            c = None
        else:
            perm = th.LongTensor([np.random.permutation([0, 1, 2, 3, 4])]).squeeze()
            c_x_set = x_set[:, perm]
            out = model.sample_conditional(c_x_set, args.sample_size, 1)
            del c_x_set
            c = out["c"]
        x = x_p.view(-1, args.in_channels, args.image_size, args.image_size)

        #model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        minibatch_metrics = model.diffusion.calc_bpd_loop(
            model.generative_model, x, c, clip_denoised=clip_denoised, model_kwargs={}
        )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) #/ dist.get_world_size()
            #dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.view(-1, args.sample_size)
        # sum over the set
        total_bpd = total_bpd.sum(-1)

        ndims = np.prod(x.shape[1:])
        # if args.mode_context in ["variational", "variational_discrete"]:
        #     total_bpd += out["klc"]

        # average over the per-set likelihood
        total_bpd = total_bpd.mean() / args.sample_size
        #/ dist.get_world_size()
        #dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())

        total_mse = minibatch_metrics["total_mse"]
        total_mse = total_mse.mean()
        
        all_mse.append(total_mse.item())
        num_complete += x.shape[0] #dist.get_world_size() * x.shape[0]

        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")
        logger.log(f"done {num_complete} samples: mse={np.mean(all_mse)}")

    # add KLc for stochastic formulation

    #if dist.get_rank() == 0:
    for name, terms in all_metrics.items():
        if args.transfer:
            out_path = os.path.join(logger.get_dir(), f"full_{name}_{args.mode_evaluation}_{args.sample_size}_transfer_{args.dataset}_terms.npz")
        else: 
            out_path = os.path.join(logger.get_dir(), f"full_{name}_{args.mode_evaluation}_{args.sample_size}_terms.npz")
        logger.log(f"saving {name} terms to {out_path}")
        np.savez(out_path, np.mean(np.stack(terms), axis=0))

    #dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
    model_path="",
    clip_denoised=True,
    num_samples=10000,
    batch_size=32,
    batch_size_eval=32,
    use_ddim=False,
    model='vfsddpm',
    dataset='cifar100',
    pool='cls', # mean, mean_patch
    mode_evaluation='out-distro',
    image_size=32,
    sample_size=5,
    patch_size=8,
    hdim=256,
    in_channels=3,
    encoder_mode='vit',
    context_channels=256,
    num_classes=1,
    mode_context="deterministic",
    augment=False,
    device="cuda",
    data_dir="/home/gigi/ns_data"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    s = set_seed(0)
    main()
