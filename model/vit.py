import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    # add time embedding
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
    pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., ns=1, t_dim=256, 
    hierarchical_patch_embedding=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean', 'mean_patch', 'agg', 'none'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.num_patches = num_patches
        self.ns = ns

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.to_time_embedding = nn.Linear(t_dim, dim)

        # hierarchical patch embedding
        # if hierarchical_patch_embedding:
        #     self.to_patch2_embedding = nn.Sequential(
        #         Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 2* patch_height, p2 = 2*patch_width),
        #         nn.Linear(2*2*patch_dim, dim),
        #     )
        
        self.k=2 
        self.pos_embedding = nn.Parameter(torch.randn(1, (num_patches) + self.k, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # n+1 because cls_tokens
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x_set = self.transformer(x)

        x = x_set.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return {'hc': x}

    def forward_set(self, img, t_emb=None, c_old=None):
        # t here is already embedded and expanded for each element in the set.

        set_to_patch_embeddings = []
        # if more than one image
        if len(img.shape) > 4:
            b, ns, ch, h, w = img.shape
            # for each element in the set
            for i in range(ns):
                inpt = img[:, i]
                # obtain patches
                patch_tmp = self.to_patch_embedding(inpt)
                set_to_patch_embeddings.append(patch_tmp)

            # [(b, p, dim) x ns]  (b, p*ns, dim)
            patches = torch.cat(set_to_patch_embeddings, dim=1)
            if self.pool == 'agg':
                p = patches.shape[1]
                patches = patches.view(b, p//ns, ns, -1)
                patches = patches.mean(dim=2)

        else:
            ns=1
            patches = self.to_patch_embedding(img)

        b, np, dim = patches.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        if t_emb is None:
            t_emb = torch.zeros(b, 1, dim).to(patches.device)
        else:
            t_emb = self.to_time_embedding(t_emb)
            t_emb = t_emb.view(b, ns+1, -1)
            t_emb = t_emb[:, 0].unsqueeze(1)

        x = torch.cat((cls_tokens, t_emb, patches), dim=1)
        #x = torch.cat((cls_tokens, patches), dim=1)

        if self.pool == "agg":
            x += self.pos_embedding[:, :(np + self.k)]

        # adapt positional encoding for a set
        else:
            # repeat the same position every np//ns steps
            tmp_pos = self.pos_embedding[:, self.k:]
            # repeat the full tensor ns times
            patches_pos = tmp_pos.repeat(1, ns, 1)
            # positional embedding for cls
            cls_pos = self.pos_embedding[:, 0].unsqueeze(0)
            # positional embedding for t
            t_pos = self.pos_embedding[:, 1].unsqueeze(0)

            # merge positional encoding for set + t + cls
            pos_embedding_patches = torch.cat((cls_pos, t_pos, patches_pos), dim=1)
            #pos_embedding_patches = torch.cat((cls_pos, patches_pos), dim=1)
            x += pos_embedding_patches[:, :(np + self.k)]
        
        x = self.dropout(x)
        x_set = self.transformer(x)

        # if we use positional encoding not elegant
        if self.pool == "agg":
            # x0=x[:, 0].unsqueeze(1)
            # t0=x[:, 1].unsqueeze(1)
            # x = x[:, self.k:].view(b, np//ns, ns, -1)
            # x = x.mean(dim = 2)
            # x_set = self.transformer(torch.cat((x0, t0, x), dim=1))
            x = x_set #[:, self.k:]
        else:
            if self.pool == 'mean':
                x = x_set.mean(dim = 1)
            elif self.pool == 'sum':
                x = x_set.sum(dim = 1)
            # use cls token as conditioning input
            elif self.pool == 'cls':
                x = x_set[:, 0]
            # compute the per-patch mean over the set
            elif self.pool == "mean_patch":
                x = x_set[:, self.k:]
                # attention here what you average
                x = x.view(b, np//ns, ns, -1)
                x = x.mean(dim = 2)
                x = self.transformer(x)
            # compute the per-patch sum over the set
            elif self.pool == "sum_patch":
                x = x_set[:, self.k:]
                # attention here what you average
                x = x.view(b, np//ns, ns, -1)
                x = x.sum(dim = 2)
                x = self.transformer(x)
            # use all the tokens
            else:
                x = x_set    
            
        x = self.to_latent(x)
        # iterative sampling
        if c_old is not None:
            x += c_old
        x = self.mlp_head(x)
        return {'hc': x, 'patches': x_set, 'cls': x_set[:, 0]}
        

