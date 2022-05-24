import matplotlib.pyplot as plt
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn #, einsum, rearrange
import numpy as np

from model.set_diffusion.gaussian_diffusion import GaussianDiffusion
from model.set_diffusion.nn import SiLU, timestep_embedding
from model.set_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from model.set_diffusion.unet import EncoderUNetModel, UNetModel
from model.vit import ViT
from model.vit_set import sViT
from model.set_diffusion.nn import mean_flat


class DDPM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bs = args.batch_size
        self.ch = args.in_channels
        self.image_size = args.image_size

        self.generative_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        print(
            "generative model parameters:", self.count_parameters(self.generative_model)
        )

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, x_set, t):
        x = x_set.view(-1, self.ch, self.image_size, self.image_size)
        loss = self.diffusion.training_losses(self.generative_model, x, t, None)
        return loss

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class VFSDDPM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.ns = args.sample_size
        self.bs = args.batch_size
        self.patch_size = args.patch_size
        self.image_size = args.image_size
        self.encoder_mode = args.encoder_mode
        self.hdim = args.hdim
        self.mode_conditioning = args.mode_conditioning
        
        self.mode_context = args.mode_context
        
        if self.encoder_mode == "unet":
            # load classifier
            self.encoder = EncoderUNetModel(
                    image_size=args.image_size,
                    in_channels=args.in_channels,
                    model_channels=args.num_channels,
                    out_channels=args.hdim,
                    num_res_blocks=2,
                    dropout=args.dropout,
                    num_head_channels=64,
                )
        elif self.encoder_mode == "vit":
            self.encoder = ViT(
                image_size=self.image_size,
                patch_size=self.patch_size,
                num_classes=args.hdim, # not important for the moment
                dim=args.hdim,
                pool=args.pool,  # use avg patch_avg
                channels=args.in_channels,
                dropout = args.dropout, 
                emb_dropout = args.dropout,
                depth=6,
                heads=12,
                mlp_dim=args.hdim,
                ns=self.ns,
            )
        elif self.encoder_mode == "vit_set":
            self.encoder = sViT(
                image_size=self.image_size,
                patch_size=self.patch_size,
                num_classes=args.hdim, # not important for the moment
                dim=args.hdim,
                pool=args.pool,  # use avg patch_avg meanpatch none
                channels=args.in_channels,
                dropout = args.dropout, 
                emb_dropout = args.dropout,
                depth=6,
                heads=12,
                mlp_dim=args.hdim,
                ns=self.ns,
            )

        # add the variational posterior at the set-level
        if self.mode_context == "variational":
            self.posterior = nn.Sequential(
                nn.Linear(args.hdim, args.hdim),
                SiLU(),
                nn.Linear(
                    args.hdim, 2 * args.hdim
                ),
            )
        elif self.mode_context == "variational_discrete":
            # number discrete codes x dim
            self.codebook = nn.Embedding(args.hdim, args.hdim)
            
        self.generative_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        print("encoder parameters:", self.count_parameters(self.encoder))
        print(
            "generative model parameters:", self.count_parameters(self.generative_model)
        )

        # assert self.mode_context == "variational" and self.mode_conditioning != "lag", 'Use mean, cls, sum with variational.'
        # assert self.mode_context == "variational_discrete" and self.mode_conditioning == "lag", 'Use lag with variational_discrete.'
        # assert self.mode_conditioning == "lag" and self.encoder_mode == "vit", "Use vit with lag."

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward_c(self, x_set, t=None):
        """
        Process input set X and return aggregated representation c for the set.
        c can be deterministic or stochastic. c can be obtained using patches (vit) or images (unet)
        """
        bs, ns, ch, h, w = x_set.size()
        # straightforward conditional DDPM
        # we encode the images in the sets 
        # and use the mean as conditioning for DDPM
        # each image here is an element in a set 
        if self.encoder_mode == "unet":
            # image level aggregation
            x = x_set.view(-1, ch, h, w)
            out = self.encoder.forward(x)
            # out = self.adaptor(out)
            out = out.view(bs, ns, -1)
            # final aggregation - simple mean
            hc = out.mean(1)
        # ViT encoder - here each image is split in patches (stacking by channels if sViT)
        # then we use these patches/tokens as input for ViT and output a collection of tokens
        # we can aggregate all the tokens and again condition DDPM with a vector, or
        # use the tokens directly to condition DDPM with an attention mechanism between set-level and sample-level
        else:
            # use time-dependent ViT
            # patch level aggregation
            t_emb = None
            if t is not None:
                t_emb = self.generative_model.time_embed(timestep_embedding(t))
            out = self.encoder.forward_set(x_set, t_emb=t_emb, c_old=None)
            hc = out["hc"]

        # continuous variational only used with film
        if self.mode_context == "variational":
            qm, qv = self.posterior(hc).chunk(2, dim=1)
            cqd = self.normal(qm, qv)

            zeros = torch.zeros(qm.size()).to(qm.device)
            #ones = torch.ones(qv.size()).to(qv.device)
            #pm, pv = self.prior(self.xc).chunk(2, dim=1)
            cpd = self.normal(zeros, zeros)
            
            c = cqd.rsample()
            return {"c": c, "cqd": cqd, "cpd": cpd, "qm": qm}

        # for the discrete variational we have to use a lag aggregation
        elif self.mode_context == "variational_discrete":
            # codebook_tokens x 16 - logits for a categorical
            # gumbel relaxation - this gives us a sample from the relaxation
            soft_one_hot = F.gumbel_softmax(hc, tau = 0.5, dim = -1, hard = False)
            # (bs, code_tokens x visual_patches) x (code_tokens, dim) 
            sampled = torch.einsum('b p c, c d -> b p d', soft_one_hot, self.codebook.weight)
            # (bs, visual_patches, dim)
            c = sampled
            return {"c": c, "logits": hc}

        return {"c": hc}

    def normal(self, loc: torch.Tensor, log_var: torch.Tensor, temperature=None):
        log_std = log_var / 2
        # if temperature:
        #     log_std = log_std * temperature
        scale = torch.exp(log_std)
        distro = td.Normal(loc=loc, scale=scale)
        return distro

    def forward(self, batch, t):
        """
        forward input set X, compute c and condition ddpm on c.
        """
        bs, ns, ch, h, w = batch.size()

        c_list = []
        for i in range(batch.shape[1]):
            ix = torch.LongTensor([k for k in range(batch.shape[1]) if k != i])
            x_set_tmp = batch[:, ix]
            
            out = self.forward_c(x_set_tmp, t)
            c_set_tmp = out["c"]
            c_list.append(c_set_tmp.unsqueeze(1))
            
        c_set = torch.cat(c_list, dim=1)
        
        # out = self.forward_c(batch, t)
        # c_set = out["c"]
        
        #c_set = c_set.unsqueeze(1) # attention here
        # repeat k element n times
        #c_set = torch.repeat_interleave(c_set, ns, dim=1)

        x = batch.view(-1, ch, self.image_size, self.image_size)

        if self.mode_conditioning == "lag":
            # (b*ns, np, dim)
            c = c_set.view(-1, c_set.size(-2), c_set.size(-1))
        else:
            # (b*ns, dim)
            c = c_set.view(-1, c_set.size(-1))
        # forward and denoising process
        losses = self.diffusion.training_losses(self.generative_model, x, t, c)

        if self.mode_context in ["variational"]:
            losses["klc"] = self.loss_c(out)
        elif self.mode_context in ["variational_discrete"]:
            losses["klc"] = self.loss_c_discrete(out)
        return losses

    def loss_c(self, out):
        """
        compute the KL between two normal distribution. 
        Here the context c is a continuous vector.
        """
        klc = td.kl_divergence(out['cqd'], out['cpd'])
        klc = mean_flat(klc) / np.log(2.0)
        return klc

    def loss_c_discrete(self, out):
        """
        Compute the KL between two categorical distributions. 
        Here c is a set of vectors each representing the logits for the codebook.
        """
        log_qy = F.log_softmax(out["logits"], dim = -1)
        log_uniform = torch.log(torch.tensor([1. / log_qy.shape[-1]], device = log_qy.device))
        klc = F.kl_div(log_uniform, log_qy, None, None, 'none', log_target = True)
        klc = mean_flat(klc) / np.log(2.0)
        return klc

    def sample_conditional(self, x_set, sample_size, k=1):
        out = self.forward_c(x_set, None) # improve with per-layer conditioning using t

        if self.mode_context == "deterministic":
            c_set = out["c"]
        elif self.mode_context == "variational":
            #c_set = out["cqd"].sample()
            c_set = out["qm"]
        elif self.mode_context == "variational_discrete":
            # at sampling time we can use argmax/argmin
            # stochastic sampling for codebook index
            #filtered_logits = top_k(out["logits"], thres = 0.5)
            # codebook_indices = gumbel_sample(out["logits"], temperature = 0.5, dim=-1)
            # deterministic sampling for codebook index
            codebook_indices = out["logits"].argmax(dim = -1)
            #bs, nc = codebook_indices.size()
            #codebook_indices = codebook_indices.flatten()
            #codebook_indices = codebook_indices.to(dtype=torch.long)
            #codebook_indices_onehot = F.one_hot(codebook_indices, num_classes = self.hdim)
            # sample from the posterior
            discrete_tokens = self.codebook(codebook_indices)
            c_set = discrete_tokens #.view(bs, nc, -1)
            
        # if we want more than sample_size samples, increase here
        #c_set = c_set.unsqueeze(1) # attention here
        #c_set = torch.repeat_interleave(c_set, k * sample_size, dim=1)
        
        if self.mode_conditioning == "lag":
            # (b*ns, np, dim)
            c = c_set.view(-1, c_set.size(-2), c_set.size(-1))
        else:
            # (b*ns, dim)
            c = c_set.view(-1, c_set.size(-1))

        if self.mode_context == "variational":
            klc = self.loss_c(out)
            return {"c": c, "qm": out["qm"], "klc": klc}

        elif self.mode_context == "variational_discrete":
            klc = self.loss_c_discrete(out)
            return {"c": c, "logits": out["logits"], "klc": klc}

        return {"c": c}

if __name__ == "__main__":
    # attention, adaptive, spatial
    # model = EncoderUNetModel(image_size=64, pool='adaptive')
    # x = torch.randn(12, 5, 3, 64, 64)
    # x = x.view(-1, 3, 64, 64)

    # out = model.forward(x)
    # print(out.size())
    # out = out.view(12, 5, -1).mean(1)
    # print(out.size())

    model = UNetModel(
        image_size=64,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions="16,8",
    )

    x = torch.randn(12, 5, 3, 64, 64)
    x = x.view(-1, 3, 64, 64)

    out = model.forward(x)
    print(out.size())
