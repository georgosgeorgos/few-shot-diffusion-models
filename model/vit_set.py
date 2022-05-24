from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SPT(nn.Module):
    """
    change shift with additional samples and process a small set stacking the images by channel
    """
    def __init__(self, *, dim, patch_size, channels = 3, sample_size=5):
        super().__init__()
        patch_dim = patch_size * patch_size * sample_size * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x_set):
        bs, ns, ch, w, h = x_set.size()
        # add augmentation?
        #shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        #shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        #x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        # (b, ch, ns, w, h)
        x_set = x_set.permute(0, 2, 1, 3, 4).contiguous()
        x_set = x_set.view(bs, ch*ns, w, h)
        return self.to_patch_tokens(x_set)

class sViT(nn.Module):
    """
    a generalization of ViT to process small sets of images
    """
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
    pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., ns=5, t_dim=256):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.ns = ns
        self.np = (image_height // patch_height) * (image_width // patch_width)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean', 'none'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.to_time_embedding = nn.Linear(t_dim, dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def forward_set(self, img, t_emb=None, c_old=None):
        # t here is already embedded and expanded for each element in the set.
        patches = self.to_patch_embedding(img)

        b, n, dim = patches.shape
        ns = self.ns

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        if t_emb is None:
            t_emb = torch.zeros(b, 1, dim).to(patches.device)
        else:
            t_emb = self.to_time_embedding(t_emb)
            t_emb = t_emb.view(b, ns, -1)
            t_emb = t_emb[:, 0].unsqueeze(1)

        x = torch.cat((cls_tokens, t_emb, patches), dim=1)
        
        x += self.pos_embedding[:, :(n + 2)]
        x = self.dropout(x)
        x_set = self.transformer(x)
        
        # if we use positional encoding not elegant
        if self.pool == 'mean':
            x = x_set.mean(dim = 1)
        elif self.pool == 'sum':
            x = x_set.sum(dim = 1)
        # use cls token as conditioning input
        elif self.pool == 'cls':
            x = x_set[:, 0]
        # compute the per-patch mean over the set
        else:
            x = x_set

        x = self.to_latent(x)
        # iterative sampling
        if c_old is not None:
            x += c_old
        x = self.mlp_head(x)
        return {'hc': x, 'patches': x_set, 'cls': x_set[:, 0]}
        