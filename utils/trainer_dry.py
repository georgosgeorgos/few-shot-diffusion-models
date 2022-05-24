import os
import shutil
import time

import numpy as np
import torch
import torchvision
from dataset.omniglot_ns import load_mnist_test_batch
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.logger import Logger
from utils.util import (mkdirs, process_batch, save_args, save_checkpoint,
                        save_metrics, set_paths)


def eval_model(args, model, loader, lst):
    log = {l: [] for l in lst}
    for batch in loader:
        with torch.no_grad():
            x = batch.to(args.device) #process_batch(args, batch, "test")["x"]
            out = model.forward(x)
            out = model.loss(out)
        
        for l in lst:
            log[l].append(out[l].data.item())
    return log

def run(args, model, optimizer, scheduler, loaders, lst):
    scheduler = StepLR(optimizer, step_size=50, gamma=args.lr_step)

    train_loader, val_loader, test_loader, _ = loaders
    args.len_tr = len(train_loader.dataset)
    args.len_vl = len(val_loader.dataset)
    args.len_ts = len(test_loader.dataset)
    
    # main training loop
    bar = tqdm(range(args.epochs))
    for epoch in bar:

        model.train()
        train_log = {l: [] for l in lst}

        # create new sets for training epoch
        train_loader.dataset.init_sets()
        for itr, batch in enumerate(train_loader):
            x = batch.to(args.device) #process_batch(args, batch, "train")["x"]
            out = model.step(x, 
                             args.alpha, 
                             optimizer, 
                             args.clip_gradients, 
                             args.free_bits)
            for l in lst:
                train_log[l].append(out[l].data.item())
                
            # print logs
            if itr % 10 == 0:
                print_str = "VLB tr:{:.10f}, KL_z tr:{:.10f}, KL_c tr:{:.10f}"
                bar.set_description(print_str.format(np.mean(train_log["vlb"][-100:]),
                                                    np.mean(train_log["kl_z"][-100:]),
                                                    np.mean(train_log["kl_c"][-100:])
                                                    ))
        # reduce weight on loss
        args.alpha *= args.alpha_step
        
        model.eval()
        # eval model at each epoch
        val_log = eval_model(args, model, val_loader, lst)
        # test model at each epoch
        test_log = eval_model(args, model, test_loader, lst)

        # update learning rate if learning plateu
        if args.adjust_lr:
            scheduler.step()