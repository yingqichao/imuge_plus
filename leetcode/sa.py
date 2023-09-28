import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
class Attention(nn.Module):
    def __init__(self, input_dim, nheads):
        super().__init__()
        self.nheads = nheads
        ndim = input_dim//nheads
        self.ndim = ndim
        assert input_dim%nheads==0, "input_dim is not divisable by nheads"
        self.qkv = nn.Linear(ndim, 3*ndim)
        self.pre_norm = nn.LayerNorm(ndim)
        self.post_norm = nn.LayerNorm(ndim)
        self.act = nn.GELU()
        self.ffn = nn.Linear(ndim, ndim)

        self.scale = self.ndim ** -0.5
    def forward(self, x):
        ## pre norm
        x = self.pre_norm(x)
        B, N, C = x.shape
        ## compute the qkv matrices
        qkv = self.qkv(x).reshape(B, N, 3, self.nheads, self.ndim).permute(2, 0, 3, 1, 4)
        ## [B, H, N, C]
        q, k, v = qkv.rebind(qkv)

        ## multiply

        attn = q@k.transpose(-2,-1) * self.scale
        soft = F.softmax(attn, dim=-1)

        value = (soft@v).transpose(1,2).reshape(B,N,C)

        value = self.post_norm(value)

        return x + value


