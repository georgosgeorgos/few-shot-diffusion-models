from asyncore import write
import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from utils.path import set_folder, mkdirs

import random
import  model.set_diffusion.dist_util

from .nn import mean_flat
from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def r(x):
    _max = np.max(x)
    _min = np.min(x)
    x = (x - _min) / (_max - _min)
    return x

def vis_samples(samples, cond, steps, mode="outdistro"):
    sx = 10
    sy = 5 + 5

    fig, ax = plt.subplots(sx, sy+1, figsize=(10, 10))
    plt.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        labelbottom="off",
        right="off",
        left="off",
        labelleft="off",
    )

    print(samples.shape, cond.shape)
    for i in range(sx):
        for j in range(sy//2):

            _c = r(cond[sx*i+j])
            ax[i, j].imshow(_c)

            _x = r(samples[sx*i+j])
            ax[i, j+sy//2+1].imshow(_x)

    for i in range(sx):
        for j in range(sy+1):
            ax[i, j].get_xaxis().set_ticks([])
            ax[i, j].get_yaxis().set_ticks([])
            ax[i, j].axis("off")

    ax[0, 2].title.set_text("Set")
    ax[0, 8].title.set_text("Samples")
    plt.subplots_adjust(wspace=0, hspace=0)

    path = "./_vis/" + logger.get_vis_name() + "/"
    mkdirs(path)
    print(path)
    plt.savefig(os.path.join(path, f"vis_model{steps}_{mode}.png"), bbox_inches='tight')
    #plt.savefig(os.path.join(path, f"vis_model{steps}_{mode}.pdf"), format="pdf", bbox_inches='tight', dpi=1000)


def conditional_sampling(args, model, loader, steps, num_samples=100, mode="outdistro"):
    logger.log("sampling starting...")
    all_images = []
    all_conditioning_images = []
    
    bs = args.batch_size
    if mode == "outdistro":
        bs = args.batch_size_eval
    # we want to use always the same batch for visualization
    while len(all_images) * bs < num_samples:

        with  th.no_grad():
            # iterate loader
            #x_set = next(loader)
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(loader)
                batch = next(loader)

            x_set = batch[:, :-1].to(args.device)
            ns = x_set.shape[1]

            if args.model == "ddpm":
                c = None
            else:
                out = model.sample_conditional(x_set, ns)
                c = out["c"]
            
            c = c.unsqueeze(1) # attention here
            # repeat k element n times
            c = th.repeat_interleave(c, ns, dim=1)
            c = c.view(-1, c.size(-2), c.size(-1))

            sample_fn = (
                model.diffusion.p_sample_loop
                if not args.use_ddim
                else model.diffusion.ddim_sample_loop
            )
            
            sample = sample_fn(
                model.generative_model,
                (
                    bs * ns,
                    args.in_channels,
                    args.image_size,
                    args.image_size,
                ),
                c=c,
                clip_denoised=args.clip_denoised,
                model_kwargs={},
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        all_images.extend([sample.cpu().numpy()])

        # normalize x_set
        x_set = x_set.view(-1, args.in_channels, args.image_size, args.image_size)
        x_set = ((x_set + 1) * 127.5).clamp(0, 255).to(th.uint8)
        x_set = x_set.permute(0, 2, 3, 1).contiguous()

        all_conditioning_images.extend([x_set.cpu().numpy()])

        logger.log(f"created {len(all_images) * bs} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]

    arr_cond = np.concatenate(all_conditioning_images, axis=0)
    arr_cond = arr_cond[: num_samples]

    # if dist.get_rank() == 0:
    shape_str = "x".join([str(x) for x in arr.shape])
    steps=str(steps)
    out_path = os.path.join(logger.get_dir(), f"samples_conditional_{shape_str}_model{steps}_{mode}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, arr_cond)

    logger.log("visualization")
    vis_samples(arr, arr_cond, steps, mode=mode)

    # dist.barrier()
    logger.log("sampling complete")

def evaluation_nll(args, model, loader, steps, num_samples=100, mode="outdistro"):
    # log this to wandb
    logger.log(f"evaluation {mode} starting...")
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    while num_complete < num_samples:
        with th.no_grad():
            # iterate loader
            #x_set = next(loader)
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(loader)
                batch = next(loader)

            batch = batch.to(args.device)
            ns = batch.shape[1]

            c_list = []
            for i in range(ns):

                ix = th.LongTensor([k for k in range(ns) if k != i])
                x_set_tmp = batch[:, ix]
                # build c
                if args.model == "ddpm":
                    c = None
                else:
                    out = model.sample_conditional(x_set_tmp, ns, 1)
                    c_tmp = out["c"]
                    c_list.append(c_tmp.unsqueeze(1))

                del x_set_tmp
                th.cuda.empty_cache()

            c = th.cat(c_list, dim=1)
            if args.mode_conditioning == "lag":
                # (b*ns, np, dim)
                c = c.view(-1, c.size(-2), c.size(-1))
            else:
                # (b*ns, dim)
                c = c.view(-1, c.size(-1))
                
            x = batch.view(-1, args.in_channels, args.image_size, args.image_size)
            #model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            minibatch_metrics = model.diffusion.calc_bpd_loop(
                model.generative_model, x, c, clip_denoised=args.clip_denoised, model_kwargs={}
            )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) #/ dist.get_world_size()
            #dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.view(-1, ns)
        total_bpd = total_bpd.sum(-1)

        ndims = np.prod(x.shape[1:])
        if args.mode_context in ["variational", "variational_discrete"]:
            total_bpd += out["klc"]

        total_bpd = total_bpd.mean() / ns
        #dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())

        num_complete += x.shape[0] #dist.get_world_size() * x.shape[0]

        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")
    
    logger.logkv_wb("bpd_" + mode , np.mean(all_bpd), steps)
    logger.logkv_wb("mse_" + mode , np.mean(all_metrics["mse"]), steps)

    #if dist.get_rank() == 0:
    for name, terms in all_metrics.items():
        out_path = os.path.join(logger.get_dir(), f"{name}_model{steps}_{mode}_terms.npz")
        logger.log(f"saving {name} terms to {out_path}")
        np.savez(out_path, np.mean(np.stack(terms), axis=0))

    #dist.barrier()
    logger.log("evaluation nll complete")


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        val_loader=None,
        args=None,
    ):
        self.args=args
        self.val_loader=val_loader

        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.microbatch = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(self.model.diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size #* dist.get_world_size()

        self.use_ddp = False

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            
            # self.ddp_model = DDP(
            #     self.model,
            #     device_ids=[dist_util.dev()],
            #     output_device=dist_util.dev(),
            #     broadcast_buffers=False,
            #     bucket_cap_mb=128,
            #     find_unused_parameters=False,
            # )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                        resume_checkpoint, map_location="cuda"
                )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load_state_dict(
                    ema_checkpoint, map_location="cuda"
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load_state_dict(
                opt_checkpoint, map_location="cuda"
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            (not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps) and self.step < 200000+1
        ):
            cond=None
            try:
                batch = next(self.data)
            except StopIteration:
                self.data = iter(self.data)
                batch = next(self.data)

            self.run_step(batch, cond)
            
            if self.step % self.log_interval == 0:
                # wandb
                logger.dumpkvs(step=self.step)
                # logger.logkv_wb("step", self.step + self.resume_step)
                # logger.logkv_wb("samples", (self.step + self.resume_step + 1) * self.global_batch)
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                # eval on 100 sets with ema? model
                self.model.eval()
                try:
                    evaluation_nll(self.args, self.model, self.data, self.step, num_samples=200, mode="indistro")
                    evaluation_nll(self.args, self.model, self.val_loader, self.step, num_samples=100, mode="outdistro")
                    
                    conditional_sampling(self.args, self.model, self.data, self.step, num_samples=200, mode="indistro")
                    conditional_sampling(self.args, self.model, self.val_loader, self.step, num_samples=200, mode="outdistro")
                except RuntimeError:
                    self.val_loader = iter(self.val_loader)
                self.model.train()

                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

        # evaluate nll for the last checkpoint using more sets
        evaluation_nll(self.args, self.model, self.data, self.step, num_samples=1000, mode="indistro")
        evaluation_nll(self.args, self.model, self.val_loader, self.step, num_samples=1000, mode="outdistro")
        # sample the model for the last checkpoint using more sets
        conditional_sampling(self.args, self.model, self.data, self.step, num_samples=200, mode="indistro")
        conditional_sampling(self.args, self.model, self.val_loader, self.step, num_samples=200, mode="outdistro")

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            if self.step > 2000 and self.step % 10 == 0:
                self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, thresh=0.3):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            #micro = batch[i : i + self.microbatch].to("cuda")

            batch = batch.to("cuda")

            dim = np.prod(batch.shape[2:])
            bs=batch.shape[0]
            ns=batch.shape[1]
            # micro_cond = {
            #     k: v[i : i + self.microbatch].to("cuda")
            #     for k, v in cond.items()
            # }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(bs, batch.device)
            # repeat t for ns element in each set
            t = th.repeat_interleave(t, ns, dim=0)
            weights = th.repeat_interleave(weights, ns, dim=0)
            
            compute_losses = functools.partial(
                self.model,
                batch,
                t,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            # this loss is already in bpd and weights ar 1
            loss = (losses["loss"] * weights)
            # sum over the sample_size to obtain the per-set loss
            loss = loss.view(bs, ns).sum(-1)
            # add klc contribution
            if "klc" in losses:
                # per set loss
                # we risk to overtrain the klc term. Maybe we can train only 10/50% steps?
                # train klc 50% of the times
                loss += losses["klc"]
            # avg over the batch for per-set loss and divide by sample_size for the per-sample loss
            loss = loss.mean() / ns
            
            log_loss_dict(
                self.model.diffusion, t, {k: v * weights for k, v in losses.items() if k not in ["klc"]}
            )
            if "klc" in losses:
                log_loss_dict(
                    self.model.diffusion, t, {"klc": losses["klc"].mean()}, False
                )
            
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        
    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            #if dist.get_rank() == 0:
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        #if dist.get_rank() == 0:
        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        #dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses, q=True):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        #Log the quantiles (four quartiles, in particular).
        if q:
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
