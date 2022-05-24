import json
import os
import pickle
import torch
import random
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
import numpy as np
from torch.nn import functional as F
import argparse


def set_seed(s=0):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    random.seed(s)
    return s


def dataset_kwargs(args):
    kwargs = {
        "omniglot_ns": {"size": 28, "nc": 1},
        "omniglot_back_eval": {"size": 28, "nc": 1},
        "omniglot_random": {"size": 28, "nc": 1},
        "mnist": {"size": 28, "nc": 1},
        "doublemnist": {"size": 28, "nc": 1},
        "triplemnist": {"size": 28, "nc": 1},
        "minimagenet": {"size": 32, "nc": 3},
        "cifar100": {"size": 32, "nc": 3},
        "cifar100mix": {"size": 32, "nc": 3},
        "cub": {"size": 64, "nc": 3},
        "celeba": {"size": 64, "nc": 3},
    }
    return kwargs


def model_kwargs(args):
    dts = dataset_kwargs(args)[args.dataset]
    img_dim = dts["size"]
    nc = dts["nc"]

    kwargs = {
        "img_dim": img_dim,
        "patch_dim": args.patch_dim,
        "in_ch": nc,
        "ch_enc": args.ch_enc,
        "batch_size": args.batch_size,
        "sample_size": args.sample_size,
        "num_layers": args.num_layers,
        "hidden_dim_c": args.hidden_dim_c,
        "n_stochastic": args.n_stochastic,
        "c_dim": args.c_dim,
        "z_dim": args.z_dim,
        "h_dim": args.h_dim,
        "hidden_dim": args.hidden_dim,
        "activation": F.elu,
        "dropout_sample": args.dropout_sample,
        "print_parameters": args.print_parameters,
        "aggregation_mode": args.aggregation_mode,
        "ladder": args.ladder,
        "ll": args.likelihood,
        "encoder_mode": args.encoder_mode
    }
    return kwargs


def count_params(model):
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return nparams


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_args(args):
    cfg = {"args": {}}
    dct = vars(args)
    for key in dct:
        cfg["args"][str(key)] = dct[key]
    path = os.path.join(args.ckpt_dir, "cfg.pkl")
    with open(path, "wb") as f:
        pickle.dump(cfg, f)
    return cfg


def save_checkpoint(args, state_dict, epoch):
    filename = args.name + "_" + args.timestamp + "_{}.pt".format(epoch + 1)
    path = os.path.join(args.ckpt_dir, filename)
    torch.save(state_dict, path)


def save_metrics(args, epoch, metrics):
    filename = args.name + "_" + args.timestamp + "_{}.json".format(epoch + 1)
    path = os.path.join(args.log_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f)


def set_paths(args):
    # year, month, day
    args.ymd = args.timestamp.split("-")[0]
    # hour, minute, second
    args.hms = args.timestamp.split("-")[1]
    # create paths
    args.ckpt_dir = os.path.join(
        args.output_dir, args.dataset, "ckpt", args.name, args.ymd, args.hms
    )
    args.fig_dir = os.path.join(
        args.output_dir, args.dataset, "fig", args.name, args.ymd, args.hms
    )
    args.log_dir = os.path.join(
        args.output_dir, args.dataset, "log", args.name, args.ymd, args.hms
    )
    args.run_dir = os.path.join(
        args.output_dir, args.dataset, "run", args.name, args.ymd, args.hms
    )

    if args.tag != "":
        args.ckpt_dir += "_" + args.tag
        args.fig_dir += "_" + args.tag
        args.log_dir += "_" + args.tag
        args.run_dir += "_" + args.tag
    return args


def load_args(args):
    path = os.path.join(args.ckpt_dir, "cfg.pkl")
    with open(path, "rb") as f:
        cfg = pickle.load(f)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.__dict__ = cfg["args"]
    return args


def load_checkpoint(args, epoch, model):
    filename = args.name + "_" + args.timestamp + "_{}.pt".format(epoch)
    path = os.path.join(args.ckpt_dir, filename)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model


def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters

    return f


def select_optimizer(args, model):
    # optimizer
    if args.optimizer == "adam":
        optimizer = Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

    elif args.optimizer == "adamw":
        optimizer = Adam(
            model.parameters(),
            weight_decay=args.weight_decay,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )

    # scheduler
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=args.patience,
            factor=args.lr_step,
            min_lr=args.lr_min,
        )
    elif args.scheduler == "exp":
        scheduler = ExponentialLR(optimizer, gamma=args.lr_step)

    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.patience, gamma=args.lr_step)

    elif args.scheduler == "warmup":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=linear_warmup(args.warmup_iters)
        )

    return optimizer, scheduler


def process_batch(args, batch, flag):
    x = batch
    x = x.unsqueeze(1)

    return {"x": x.to(args.device)}


# def process_batch(args, batch, flag):
#     x = batch
#     x = x.unsqueeze(1)

#     bs, ns, c, h, w = x.size()

#     if flag == "train" or flag == "marginal":

#         x = x.view(-1, args.sample_size * args.num_classes, c, h, w)

#         col_idx = list(range(args.sample_size * args.num_classes))
#         random.shuffle(col_idx)

#         x = x[:, torch.tensor(col_idx)]
#         x = x.view(-1, args.sample_size, c, h, w)

#         if args.randomize_sample_size:
#             ns = np.random.choice([5, 10, 20])
#             x = x.view(-1, ns, c, h, w)

#         row_idx = list(range(x.shape[0]))
#         random.shuffle(row_idx)
#         x = x[torch.tensor(row_idx)]

#     return {"x": x.to(args.device)}

