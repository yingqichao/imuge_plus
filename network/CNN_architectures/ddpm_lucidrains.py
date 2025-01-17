import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
# from ema_pytorch import EMA
#
# from accelerate import Accelerator
#
# from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        if len(x.shape)==1:
            x = x[:, None]
        emb = x * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, use_bayar=False):
        super().__init__()
        if use_bayar:
            self.proj = BayarConv(dim, dim_out, groups=2)
        else:
            self.proj = WeightStandardizedConv2d(dim, dim_out, 5, padding = 2)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, enable_fft = True, use_bayar=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.enable_fft = enable_fft

        if self.enable_fft:

            self.block1 = Block(dim, dim_out//2, groups = groups, use_bayar=use_bayar)
            self.block2 = Block(dim_out//2, dim_out//2, groups = groups)

            self.block1_fft = Block(2*dim+2, dim_out, groups=groups)
            self.block2_fft = Block(dim_out, dim_out, groups=groups)
        else:
            self.block1 = Block(dim, dim_out , groups=groups, use_bayar=use_bayar)
            self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift, scale_shift_fft = None, None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb_local = self.time_mlp(time_emb)
            time_emb_local = rearrange(time_emb_local, 'b c -> b c 1 1')
            scale_shift = time_emb_local.chunk(2, dim = 1)
            time_emb_fft = self.mlp(time_emb)
            time_emb_fft = rearrange(time_emb_fft, 'b c -> b c 1 1')
            scale_shift_fft = time_emb_fft.chunk(2, dim=1)

        ### local path
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        if self.enable_fft:
            ### global path
            batch = x.shape[0]
            fft_dim = (-2, -1)
            ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

            ffted = self.block1_fft(ffted, scale_shift=scale_shift_fft)
            ffted = self.block2_fft(ffted)

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])

            ifft_shape_slice = x.shape[-2:]
            h_fft = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')

            h = torch.cat([h, h_fft],dim=1)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

from timm.models.vision_transformer import ViTBlock
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model
# from einops.layers.torch import Reduce
class BayarConv(nn.Module):
    def __init__(self, num_in, num_out, groups=2):
        super().__init__()
        self.groups = groups
        assert groups<=2, "group of bayarconv cannot exceed 2!"
        self.num_in_bayar = num_in
        self.num_out_bayar = num_out//groups
        self.BayarConv2D = nn.Conv2d(num_in, num_out//groups, 5, 1, padding=2, bias=False)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        self.bayar_mask[2, 2] = 0
        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        self.bayar_final[2, 2] = -1
        if groups>1:
            self.vanillaConv2D = nn.Conv2d(num_in, num_out // groups, 5, 1, padding=2, bias=False)

        self.norm = nn.GroupNorm(groups, num_out)
        self.act = nn.SiLU()


    def forward(self, x):
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(
            self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(self.num_out_bayar, self.num_in_bayar, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final

        if self.groups>1:
            x = torch.cat([self.BayarConv2D(x), self.vanillaConv2D(x)],dim=1)
        else:
            x = self.BayarConv2D(x)
        x = self.norm(x)
        x = self.act(x)

        return x


import numpy as np
from network.attention.fcanet_channel_attention_official import MultiSpectralAttentionLayer
class Unet(nn.Module):
    def __init__(
        self,
        # out_dim,
        init_dim=None,
        dim=32,
        dim_mults=[1, 2, 4, 8],
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        use_bayar = False,
        use_fft = False,
        use_classification = None,
        # use_middle_features = False,
        use_hierarchical_class = None,
        use_hierarchical_segment = None,
        use_normal_output = None,
        use_SRM = False
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        # self.use_middle_features = use_middle_features
        self.use_hierarchical_class = use_hierarchical_class
        self.use_hierarchical_segment = use_hierarchical_segment
        self.use_normal_output = use_normal_output
        self.use_classification = use_classification
        # determine dimensions
        self.use_bayar = use_bayar
        if self.use_bayar:
            self.BayarConv2D = BayarConv(3, self.use_bayar, groups=1)
            # self.BayarConv2D = nn.Conv2d(3, self.use_bayar, 5, 1, padding=2, bias=False)
            # self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
            # self.bayar_mask[2, 2] = 0
            # self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
            # self.bayar_final[2, 2] = -1
            self.activation = nn.SiLU()
            input_channels += self.use_bayar
        self.use_SRM = use_SRM
        if self.use_SRM:
            self.SRMConv2D = nn.Conv2d(3, 9, 5, 1, padding=2, bias=False)
            self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']

            ##SRM filters (fixed)
            for param in self.SRMConv2D.parameters():
                param.requires_grad = False

            input_channels += 9

        self.use_fft = use_fft
        self.fft_norm = 'ortho'


        init_dim = default(init_dim, dim)
        # self.out_dim = out_dim  # default(out_dim, default_out_dim)

        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups, enable_fft=self.use_fft)

        # time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, use_bayar=use_bayar>0),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)


        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.up_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)

        ######## norm layers
        # self.norm_class = nn.ModuleList([])
        dims_hierachical = [*map(lambda m: dim * m, reversed(dim_mults))]
        # for i in range(len(self.ups)):
        #     self.norm_class.append(nn.LayerNorm(dims_hierachical[i]))

        ######## upsample layers
        self.upsamples = nn.ModuleList([])
        self.bottleneck = nn.ModuleList([])
        for item in reversed(dim_mults):
            self.upsamples.append(nn.Upsample(scale_factor = item, mode = 'nearest'))
            self.bottleneck.append(nn.Conv2d(dim*item, dim, 1))
        # default_out_dim = channels * (1 if not learned_variance else 2)

        ######## classification head at the middle
        if self.use_classification:
            self.middle_class_mlp = nn.ModuleList([])
            for idx in range(len(self.use_classification)):
                self.middle_class_mlp.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                        nn.Flatten(),
                        nn.Linear(mid_dim, mid_dim),
                        nn.SiLU(),
                        nn.Linear(mid_dim, self.use_classification[idx])
                    )
                )


        ######## feature refiners
        if (len(self.use_hierarchical_segment)+len(self.use_hierarchical_class))>0:
            self.refine_layers = nn.ModuleList([])
            self.gates = nn.ModuleList([])
            for idx in range(len(self.use_hierarchical_segment)+len(self.use_hierarchical_class)):
                self.refine_layers.append(
                    nn.Sequential(
                        # nn.Conv2d(sum(dims_hierachical), dim, 1),
                        block_klass(dim, dim, time_emb_dim=time_dim),
                        # nn.Conv2d(dim, self.out_dim[0], 1)
                    )
                )

                self.gates.append(
                    nn.Sequential(
                            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                            nn.Flatten(),
                            nn.Linear(dim, dim),
                            nn.SiLU(),
                            nn.Linear(dim, len(self.use_hierarchical_segment)+len(self.use_hierarchical_class)),
                            nn.Softmax(dim=1),
                    )
                )

        ######## hierarchical feat output (classification)
        if self.use_hierarchical_class:
            self.hierarchical_class_mlp = nn.ModuleList([])
            for idx in range(len(self.use_hierarchical_class)):
                self.hierarchical_class_mlp.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                        nn.Flatten(),
                        nn.Linear(dim, dim),
                        nn.SiLU(),
                        nn.Linear(dim, self.use_hierarchical_class[idx])
                    )
                )

        ######## hierarchical feat output (segmentation)
        if self.use_hierarchical_segment:
            self.final_hierarchical_conv = nn.ModuleList([])
            for idx in range(len(self.use_hierarchical_segment)):
                self.final_hierarchical_conv.append(
                    nn.Sequential(
                        nn.Conv2d(dim, self.use_hierarchical_segment[idx], 1)
                )
            )

        if self.use_normal_output:
            self.final_conv = nn.ModuleList([])
            for idx in range(len(self.use_normal_output)):
                self.final_conv.append(
                    nn.Conv2d(dim, self.use_normal_output[-1], 1)
                )


    def feature_extract_encoder(self,input, time=None, x_self_cond=None):
        # outputs = []
        batch = input.shape[0]
        middle_feats = []

        ori_feats = []
        ori_feats.append(input)

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            input = torch.cat((x_self_cond, input), dim=1)
        ### support bayar conv as init
        if self.use_bayar:
            # self.BayarConv2D.weight.data *= self.bayar_mask
            # self.BayarConv2D.weight.data *= torch.pow(
            #     self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(self.use_bayar, 3, 1, 1), -1)
            # self.BayarConv2D.weight.data += self.bayar_final

            # Symmetric padding
            # x = symm_pad(x, (2, 2, 2, 2))

            # conv_init = self.vanillaConv2D(x)
            ori_feats.append(self.BayarConv2D(input))
            # conv_srm = self.SRMConv2D(x)

            # first_block = conv_bayar  # torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
            # x_bayar = self.activation(first_block)
            # x = torch.cat([input, conv_bayar], dim=1)

        if self.use_SRM:
            conv_srm = self.SRMConv2D(input)
            ori_feats.append(conv_srm)

        x = self.init_conv(torch.cat(ori_feats, dim=1))

        r = x.clone()

        t = None  # self.time_mlp(time)

        h = []

        for idx, item in enumerate(self.downs):
            block1, block2, attn, downsample = item
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        x_cls = []
        if self.use_classification:
            # x_norm = x.mean([-2, -1])
            for idx in range(len(self.use_classification)):
                # x_cls = self.norm_class[0]()
                x_cls.append(self.middle_class_mlp[idx](x))
                # outputs.append(x_cls)
                # t = self.time_mlp(x_cls.detach())

        return x, r, t, h, middle_feats, x_cls

    def feature_extract_decoder(self, x, r, t, h, middle_feats, x_cls):
        for idx, item in enumerate(self.ups):
            block1, block2, attn, upsample = item
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            # if idx>1:
            # if self.use_hierarchical_class:
            #     middle_feats.append(self.norm_class[idx](x.mean([-2, -1])))
            # elif self.use_hierarchical_segment:
            #     middle_feats.append(self.upsamples[idx](x))
            # else:
            if idx!=len(self.ups)-1:
                middle_feats.append(self.bottleneck[idx](self.upsamples[idx](x)))
                # middle_feats.append(self.norm_class[idx](x.mean([-2, -1])))
                # middle_feats.append(x)
            x = upsample(x)

        # if self.use_hierarchical_class:
        #     middle_feats.append(self.norm_class[-1](x.mean([-2, -1])))
        # else:
        #     middle_feats.append(x)

        x = torch.cat((x, r), dim=1)
        x = self.up_res_block(x, t)
        middle_feats.append(x)
        ## varied last output: (x_cls), middle_feats, (hierarchical_class), (hierarchical_segment), (out)
        # outputs.append(middle_feats)
        return x, r, t, middle_feats, x_cls

    def forward(self, x, time=None, x_self_cond = None):
        num_tasks = len(self.use_hierarchical_class) + len(self.use_hierarchical_segment)
        hier_class_output, hier_seg_output, hier_post_feats, refined_feats, gates, outs = [], [], [[]*(num_tasks)], [], [], []
        x_out = None
        x, r, t, h, middle_feats, x_cls = self.feature_extract_encoder(x, x_self_cond)
        x, r, t, middle_feats, x_cls = self.feature_extract_decoder(x, r, t, h, middle_feats, x_cls)
        # cat_feats = torch.cat(middle_feats, dim=1)
        for i, feat in enumerate(middle_feats):
            for idx in range(len(self.use_hierarchical_class)+len(self.use_hierarchical_segment)):
                refined = self.refine_layers[idx](feat)
                ## predict score on the refined feature
                gate = self.gates[idx](refined)
                refined_feats.append(refined)
                gates.append(gate)

            for idx in range(len(self.use_hierarchical_class) + len(self.use_hierarchical_segment)):
                hier_feats_task = hier_post_feats[idx]
                out = 0
                for idx_gate in range(len(self.use_hierarchical_class) + len(self.use_hierarchical_segment)):
                    out += refined_feats[idx_gate]*gates[idx][:,idx_gate].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                hier_feats_task.append(out)


        # if self.use_hierarchical_class:
        #     middle_feats.append(self.norm_class[idx](x.mean([-2, -1])))
        # elif self.use_hierarchical_segment:
        #     middle_feats.append(self.upsamples[idx](x))

        if self.use_hierarchical_class:
            # middle_pool_feats = []
            # for idx in range(len(self.ups)):
            #     middle_pool_feats.append(self.norm_class[idx](middle_feats[idx].mean([-2, -1])))
            for idx in range(len(self.use_hierarchical_class)):
                hier_feats_task = hier_post_feats[idx]
                # feats_class = []
                # for feat in hier_feats_task:
                #     out = self.hierarchical_class_mlp[idx](feat)
                #     feats_class.append(out)
                # out = reduce(torch.cat(feats_class, dim=1), 'b c -> b 1', 'mean')
                feat_class = None
                for feat in hier_feats_task:
                    feat_class = feat if feat_class is None else feat_class + feat
                out = self.hierarchical_class_mlp[idx](feat_class)
                hier_class_output.append(out)

                # out = self.hierarchical_class_mlp[idx](hier_post_feats[idx])
                # hier_class_output.append(out)

        if self.use_hierarchical_segment:
            # middle_upsample_feats = []
            # for idx in range(len(self.ups)):
            #     middle_upsample_feats.append(self.upsamples[idx](middle_feats[idx]))
            for idx in range(len(self.use_hierarchical_segment)):
                hier_feats_task = hier_post_feats[idx+len(self.use_hierarchical_class)]
                # feats_seg = []
                # for feat in hier_feats_task:
                #     out = self.final_hierarchical_conv[idx](feat)
                #     feats_seg.append(out)
                # out = reduce(torch.cat(feats_seg,dim=1), 'b c h w -> b 1 h w', 'max')
                feat_seg = None
                for feat in hier_feats_task:
                    feat_seg = feat if feat_seg is None else feat_seg+feat
                out = self.final_hierarchical_conv[idx](feat_seg)
                hier_seg_output.append(out)

        if self.use_normal_output:
            for idx in range(len(self.use_normal_output)):
                out = self.final_conv[idx](middle_feats[-1])
                outs.append(out)

        return middle_feats, hier_class_output, hier_seg_output, x_cls, outs

    def forward_classification(self, x=None, time=None, x_self_cond = None, middle_feats=None):
        hier_class_output, hier_seg_output, hier_post_feats, refined_feats, gates, outs = [], [], [], [], [], []
        x_out = None
        x, r, t, h, middle_feats, x_cls = self.feature_extract_encoder(x, x_self_cond)

        return x_cls

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

# class Dataset(Dataset):
#     def __init__(
#         self,
#         folder,
#         image_size,
#         exts = ['jpg', 'jpeg', 'png', 'tiff'],
#         augment_horizontal_flip = False,
#         convert_image_to = None
#     ):
#         super().__init__()
#         self.folder = folder
#         self.image_size = image_size
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
#
#         maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
#
#         self.transform = T.Compose([
#             T.Lambda(maybe_convert_fn),
#             T.Resize(image_size),
#             T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
#             T.CenterCrop(image_size),
#             T.ToTensor()
#         ])
#
#     def __len__(self):
#         return len(self.paths)
#
#     def __getitem__(self, index):
#         path = self.paths[index]
#         img = Image.open(path)
#         return self.transform(img)

if __name__ == '__main__':
    model = Unet().cuda()
    input = torch.ones((3,3,128,128)).cuda()
    output = model(input, torch.zeros((1)).cuda())
    print(output.shape)

# trainer class

# class Trainer(object):
#     def __init__(
#         self,
#         diffusion_model,
#         folder,
#         *,
#         train_batch_size = 16,
#         gradient_accumulate_every = 1,
#         augment_horizontal_flip = True,
#         train_lr = 1e-4,
#         train_num_steps = 100000,
#         ema_update_every = 10,
#         ema_decay = 0.995,
#         adam_betas = (0.9, 0.99),
#         save_and_sample_every = 1000,
#         num_samples = 25,
#         results_folder = './results',
#         amp = False,
#         fp16 = False,
#         split_batches = True,
#         convert_image_to = None
#     ):
#         super().__init__()
#
#         self.accelerator = Accelerator(
#             split_batches = split_batches,
#             mixed_precision = 'fp16' if fp16 else 'no'
#         )
#
#         self.accelerator.native_amp = amp
#
#         self.model = diffusion_model
#
#         assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
#         self.num_samples = num_samples
#         self.save_and_sample_every = save_and_sample_every
#
#         self.batch_size = train_batch_size
#         self.gradient_accumulate_every = gradient_accumulate_every
#
#         self.train_num_steps = train_num_steps
#         self.image_size = diffusion_model.image_size
#
#         # dataset and dataloader
#
#         self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
#         dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
#
#         dl = self.accelerator.prepare(dl)
#         self.dl = cycle(dl)
#
#         # optimizer
#
#         self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
#
#         # for logging results in a folder periodically
#
#         if self.accelerator.is_main_process:
#             self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
#
#         self.results_folder = Path(results_folder)
#         self.results_folder.mkdir(exist_ok = True)
#
#         # step counter state
#
#         self.step = 0
#
#         # prepare model, dataloader, optimizer with accelerator
#
#         self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
#
#     def save(self, milestone):
#         if not self.accelerator.is_local_main_process:
#             return
#
#         data = {
#             'step': self.step,
#             'model': self.accelerator.get_state_dict(self.model),
#             'opt': self.opt.state_dict(),
#             'ema': self.ema.state_dict(),
#             'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
#             'version': __version__
#         }
#
#         torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
#
#     def load(self, milestone):
#         accelerator = self.accelerator
#         device = accelerator.device
#
#         data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
#
#         model = self.accelerator.unwrap_model(self.model)
#         model.load_state_dict(data['model'])
#
#         self.step = data['step']
#         self.opt.load_state_dict(data['opt'])
#         self.ema.load_state_dict(data['ema'])
#
#         if 'version' in data:
#             print(f"loading from version {data['version']}")
#
#         if exists(self.accelerator.scaler) and exists(data['scaler']):
#             self.accelerator.scaler.load_state_dict(data['scaler'])
#
#     def train(self):
#         accelerator = self.accelerator
#         device = accelerator.device
#
#         with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
#
#             while self.step < self.train_num_steps:
#
#                 total_loss = 0.
#
#                 for _ in range(self.gradient_accumulate_every):
#                     data = next(self.dl).to(device)
#
#                     with self.accelerator.autocast():
#                         loss = self.model(data)
#                         loss = loss / self.gradient_accumulate_every
#                         total_loss += loss.item()
#
#                     self.accelerator.backward(loss)
#
#                 accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
#                 pbar.set_description(f'loss: {total_loss:.4f}')
#
#                 accelerator.wait_for_everyone()
#
#                 self.opt.step()
#                 self.opt.zero_grad()
#
#                 accelerator.wait_for_everyone()
#
#                 self.step += 1
#                 if accelerator.is_main_process:
#                     self.ema.to(device)
#                     self.ema.update()
#
#                     if self.step != 0 and self.step % self.save_and_sample_every == 0:
#                         self.ema.ema_model.eval()
#
#                         with torch.no_grad():
#                             milestone = self.step // self.save_and_sample_every
#                             batches = num_to_groups(self.num_samples, self.batch_size)
#                             all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
#
#                         all_images = torch.cat(all_images_list, dim = 0)
#                         utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
#                         self.save(milestone)
#
#                 pbar.update(1)
#
#         accelerator.print('training complete')