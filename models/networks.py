import torch
import logging
# import .modules.discriminator_vgg_arch as SRGAN_arch
# from .modules.Inv_arch import *
# from .modules.Subnet_constructor import subnet
import math
import torch.nn as nn
import numpy as np
from collections import OrderedDict
logger = logging.getLogger('base')

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class NormalGenerator(nn.Module):
    def __init__(self, dims_in=[[3, 64, 64]], down_num=3, block_num=[4,4,4],out_channel=3):
        super(NormalGenerator, self).__init__()
        self.out_channel = out_channel
        operations = []

        current_dims = dims_in
        for i in range(down_num):
            b = HaarDownsampling(current_dims)
            # b = Squeeze()
            operations.append(b)
            current_dims[0][0] = current_dims[0][0] * 4
            current_dims[0][1] = current_dims[0][1] // 2
            current_dims[0][2] = current_dims[0][2] // 2
            for j in range(block_num[i]):
                b = ResBlock(current_dims[0][0],current_dims[0][0])
                # b = RNVPCouplingBlock(current_dims, subnet_constructor=DenseBlock, clamp=1.0)
                operations.append(b)
        block_num = block_num[:-1][::-1]
        block_num.append(0)
        for i in range(down_num):
            b = HaarUpsampling(current_dims)
            # b = Unsqueeze()
            operations.append(b)
            current_dims[0][0] = current_dims[0][0] // 4
            current_dims[0][1] = current_dims[0][1] * 2
            current_dims[0][2] = current_dims[0][2] * 2
            for j in range(block_num[i]):
                b = ResBlock(current_dims[0][0],current_dims[0][0])
                operations.append(b)

        # self.out_layer = nn.Conv2d(current_dims[0][0], out_channel, 1,1,0)
        # operations.append(self.out_layer)
        self.operations = nn.ModuleList(operations)
        # self.guassianize = Gaussianize(1)

    def forward(self, x):
        out = x

        for op in self.operations:
            out = op.forward(out)
        out = out[:, :self.out_channel, :, :]
        return out



class InpaintGenerator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=3, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()
        dim=16
        self.encoder_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=7, padding=0,bias=False),
            # nn.BatchNorm2d(dim, affine=True),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1,bias=False),
            nn.InstanceNorm2d(64),
            # nn.BatchNorm2d(dim, affine=True),
            nn.GELU(),
        )
        self.encoder_1 = nn.Sequential(

            nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=4, stride=2, padding=1,bias=False),
            # nn.BatchNorm2d(dim*2, affine=True),
            nn.InstanceNorm2d(dim*2),
            nn.GELU(),
            nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim*2),
            # nn.BatchNorm2d(dim*2, affine=True),
            nn.GELU(),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim*4, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(dim*4, affine=True),
            nn.InstanceNorm2d(dim*4),
            nn.GELU(),
            nn.Conv2d(in_channels=dim*4, out_channels=dim*4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim*4),
            # nn.BatchNorm2d(dim*4, affine=True),
            nn.GELU()
        )

        blocks = []
        for _ in range(residual_blocks): #residual_blocks
            block = ResnetBlock(dim*4, dilation=2, use_spectral_norm=False)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim*4*2, out_channels=dim*2, kernel_size=4, stride=2, padding=1,bias=False),
            # nn.BatchNorm2d(dim*2, affine=True),
            nn.InstanceNorm2d(dim*2),
            nn.GELU(),
            nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim*2),
            # nn.BatchNorm2d(dim*2, affine=True),
            nn.GELU(),
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim*2*2, out_channels=dim, kernel_size=4, stride=2, padding=1,bias=False),
            # nn.BatchNorm2d(dim, affine=True),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            # nn.BatchNorm2d(dim, affine=True),
            nn.GELU(),
        )

        self.decoder_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=dim*2, out_channels=out_channels, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        e0 = self.encoder_0(x)
        e1 = self.encoder_1(e0)
        e2 = self.encoder_2(e1)
        m = self.middle(e2)
        d2 = self.decoder_2(torch.cat((e2,m),dim=1))
        d1 = self.decoder_1(torch.cat((e1,d2),dim=1))
        x = self.decoder_0(torch.cat((e0, d1), dim=1))
        # x = (torch.tanh(x) + 1) / 2

        return x


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, use_spectral_norm=False):
        super(ResBlock, self).__init__()
        feature = channel_in
        if not use_spectral_norm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(channel_in, feature, kernel_size=3, padding=1,bias=False),
                nn.InstanceNorm2d(feature),
                # nn.BatchNorm2d(feature, affine=True),
                nn.GELU(),
            )
        else:
            self.conv_1 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=channel_in, out_channels=feature, kernel_size=3, padding=1,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.GELU(),
            )

        if not use_spectral_norm:
            self.conv2 = nn.Sequential(
                nn.Conv2d(feature, feature, kernel_size=3, padding=1,bias=False),
                nn.InstanceNorm2d(feature),
                # nn.BatchNorm2d(feature, affine=True),
                nn.GELU(),
            )
        else:
            self.conv_2 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=3, padding=1,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.GELU(),
            )

        if not use_spectral_norm:
            self.conv3 = nn.Sequential(
                nn.Conv2d(feature, feature, kernel_size=3, padding=1,bias=False),
                nn.InstanceNorm2d(feature),
                # nn.BatchNorm2d(feature, affine=True),
                nn.GELU(),
            )
        else:
            self.conv_3 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=3, padding=1,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.GELU(),
            )

        if not use_spectral_norm:
            self.conv4 = nn.Sequential(
                nn.Conv2d(feature, feature, kernel_size=3, padding=1,bias=False),
                nn.InstanceNorm2d(feature),
                # nn.BatchNorm2d(feature, affine=True),
                nn.GELU(),
            )
        else:
            self.conv_4 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=3, padding=1,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.GELU(),
            )

        self.conv5 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        residual = self.conv4(residual)
        input = torch.cat((x, residual), dim=1)
        out = self.conv5(input)
        return out

def symm_pad(im, padding):
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]

def reflect(x, minx, maxx):
    """ Reflects an array around two points making a triangular waveform that ramps up
    and down,  allowing for pad lengths greater than the input length """
    rng = maxx - minx
    double_rng = 2 * rng
    mod = np.fmod(x - minx, double_rng)
    normed_mod = np.where(mod < 0, mod + double_rng, mod)
    out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


class DG_discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True, use_SRM=False):
        super(DG_discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        dim = 256
        self.use_SRM = use_SRM
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),
            # spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=3, stride=1, padding=1,
            #                         bias=not use_spectral_norm), use_spectral_norm),
            # nn.GELU()

        )

        self.conv2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=dim , out_channels=dim , kernel_size=4,
                          stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),
            # spectral_norm(nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1,
            #                         bias=not use_spectral_norm), use_spectral_norm),
            # nn.GELU()
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim , out_channels=dim , kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),
            # spectral_norm(nn.Conv2d(in_channels=dim*4, out_channels=dim*4, kernel_size=3, stride=1, padding=1,
            #                         bias=not use_spectral_norm), use_spectral_norm),
            # nn.GELU()
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim , out_channels=dim , kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),
            # spectral_norm(nn.Conv2d(in_channels=dim*8, out_channels=dim*8, kernel_size=3, stride=1, padding=1,
            #                         bias=not use_spectral_norm), use_spectral_norm),
            # nn.GELU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=dim , out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):

        conv1 = self.conv1(x)
        # conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True,use_SRM=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        dim=32
        self.use_SRM = use_SRM
        if self.use_SRM:
            ## bayar conv
            self.BayarConv2D = nn.Conv2d(3, 3, 5, 1, padding=2, bias=False)
            self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
            self.bayar_mask[2, 2] = 0
            self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
            self.bayar_final[2, 2] = -1

            ## srm conv
            self.SRMConv2D = nn.Conv2d(3, 9, 5, 1, padding=2, bias=False)
            self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']

            ##SRM filters (fixed)
            for param in self.SRMConv2D.parameters():
                param.requires_grad = False

            in_channels = in_channels+12

        self.init_conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=4, stride=2,
                          padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True)

        )

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True)

        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(dim*2, out_channels=dim*4, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim*4, out_channels=dim*4, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim*4, out_channels=dim*8, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim*8, out_channels=dim*8, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim*8, out_channels=dim*16, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim*16, out_channels=dim*16, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=dim*16, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.relu = nn.ELU(inplace=True)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        if self.use_SRM:
            self.BayarConv2D.weight.data *= self.bayar_mask
            self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1),-1)
            self.BayarConv2D.weight.data += self.bayar_final
            conv_bayar = self.BayarConv2D(x[:,:3])
            conv_srm = self.SRMConv2D(x[:,:3])

            x = torch.cat((x,conv_srm, conv_bayar), dim=1)
            # x = self.relu(x)

        conv0 = self.init_conv(x)

        # conv1 = self.activation(conv1)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs , [conv1, conv2, conv3, conv4, conv5]

import torch.nn.functional as F
class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray

import torch.nn.functional as Functional
class Localizer(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=1, residual_blocks=2, init_weights=True, use_spectral_norm=True,
                 dim=16, use_sigmoid=False):
        super(Localizer, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.constrain_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)

        self.encoder_0 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=3, stride=1, padding=1),
                          use_spectral_norm),
            nn.ELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
                          use_spectral_norm),
            nn.ELU(),
        )

        self.encoder_1 = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(),
        )
        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU()
        )
        self.encoder_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.ELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1),
                          use_spectral_norm),
            nn.ELU()
        )

        blocks = []
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 4, dilation=2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_3 = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(in_channels=dim * 4 * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1),
                use_spectral_norm),
            nn.ELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1),
                          use_spectral_norm),
            nn.ELU(),
        )

        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 4 * 2, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(),
        )

        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2 * 2, out_channels=dim, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(),
        )

        self.decoder_0 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                          use_spectral_norm),
            nn.ELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
                          use_spectral_norm),
            nn.ELU(),
        )

        # self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, padding=0)
        self.decision_making = nn.Conv2d(in_channels=dim+dim+dim*2+dim*4, out_channels=out_channels, kernel_size=1, padding=0)

        if init_weights:
            self.init_weights()


    def forward(self, x):

        x = rgb2gray(x)
        x = self.constrain_conv(x)

        e0 = self.encoder_0(x)

        e1 = self.encoder_1(e0)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)
        m = self.middle(e3)
        d3 = self.decoder_3(torch.cat((e3, m), dim=1))
        d3_up = Functional.interpolate(
                                d3,
                                size=[x.shape[2], x.shape[3]],
                                mode='bilinear')
        d2 = self.decoder_2(torch.cat((e2, d3), dim=1))
        d2_up = Functional.interpolate(
            d2,
            size=[x.shape[2], x.shape[3]],
            mode='bilinear')
        d1 = self.decoder_1(torch.cat((e1, d2), dim=1))
        d1_up = Functional.interpolate(
            d1,
            size=[x.shape[2], x.shape[3]],
            mode='bilinear')
        x = self.decoder_0(torch.cat((e0, d1), dim=1))
        # x = self.final_conv(torch.cat((e_begin, x), dim=1))
        x = self.decision_making(torch.cat((x,d1_up,d2_up,d3_up), dim=1))
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x


class UnetResBlock(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=3, residual_blocks=8, init_weights=True, use_spectral_norm=True,
                 dim=32, use_sigmoid=False):
        super(UnetResBlock, self).__init__()

        self.use_sigmoid = use_sigmoid

        self.in_channels = in_channels

        self.init_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=self.in_channels, out_channels=dim, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        blocks = []
        blocks.append(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm))
        blocks.append(nn.ELU(inplace=True))
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 2, dilation=1, use_spectral_norm=use_spectral_norm)
            blocks.append(block)
        self.encoder_1 = nn.Sequential(*blocks)

        blocks = []
        blocks.append(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm))
        blocks.append(nn.ELU(inplace=True))
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 4, dilation=1, use_spectral_norm=use_spectral_norm)
            blocks.append(block)
        self.encoder_2 = nn.Sequential(*blocks)

        blocks = []
        blocks.append(
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 8, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm))
        blocks.append(nn.ELU(inplace=True))
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 8, dilation=1, use_spectral_norm=use_spectral_norm)
            blocks.append(block)
        self.encoder_3 = nn.Sequential(*blocks)

        blocks = []
        blocks.append(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 8 * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm))
        blocks.append(nn.ELU(inplace=True))
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 4, dilation=1, use_spectral_norm=use_spectral_norm)
            blocks.append(block)
        self.decoder_3 = nn.Sequential(*blocks)

        blocks = []
        blocks.append(
            spectral_norm(
                nn.ConvTranspose2d(in_channels=dim * 4 * 2, out_channels=dim * 2, kernel_size=4, stride=2, padding=1),
                use_spectral_norm))
        blocks.append(nn.ELU(inplace=True))
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 2, dilation=1, use_spectral_norm=use_spectral_norm)
            blocks.append(block)
        self.decoder_2 = nn.Sequential(*blocks)

        blocks = []
        blocks.append(
            spectral_norm(
                nn.ConvTranspose2d(in_channels=dim * 2 * 2, out_channels=dim, kernel_size=4, stride=2, padding=1),
                use_spectral_norm))
        blocks.append(nn.ELU(inplace=True))
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim, dilation=1, use_spectral_norm=use_spectral_norm)
            blocks.append(block)
        self.decoder_1 = nn.Sequential(*blocks)

        self.decoder_0 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, stride=1, padding=1),
                          use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, stride=1, padding=1),
                          use_spectral_norm),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=1, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):

        e0 = self.init_conv(x)

        e1 = self.encoder_1(e0)

        e2 = self.encoder_2(e1)

        e3 = self.encoder_3(e2)

        d3 = self.decoder_3(torch.cat((e3, m), dim=1))
        # d3_add = self.decoder_3_add(d3)
        # d3 = d3_add #self.clock * d2_add + (1 - self.clock) * d2
        d2 = self.decoder_2(torch.cat((e2, d3), dim=1))
        # d2_add = self.decoder_2_add(d2)
        # d2 = d2_add #self.clock * d2_add + (1 - self.clock) * d2
        d1 = self.decoder_1(torch.cat((e1, d2), dim=1))
        # d1_add = self.decoder_1_add(d1)
        # d1 = d1_add #self.clock * d1_add + (1 - self.clock) * d1

        d0_concat = torch.cat((e0, d1), dim=1)

        x = self.decoder_0(d0_concat)
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        if self.output_middle_feature:
            return x, middle_feat
        else:
            return x

class UNetDiscriminator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=1, residual_blocks=4, init_weights=True, use_spectral_norm=True,
                 use_SRM=True, dim=32, use_sigmoid=False, output_middle_feature=False):
        super(UNetDiscriminator, self).__init__()
        # dim = 32
        self.use_SRM = use_SRM
        self.use_sigmoid = use_sigmoid
        self.clock = 1
        self.in_channels = in_channels

        self.output_middle_feature = output_middle_feature
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.subtask = subtask
        # if self.use_SRM:

        # self.SRMConv2D = nn.Conv2d(in_channels, 9, 5, 1, padding=2, bias=False)
        # self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
        # ##SRM filters (fixed)
        # for param in self.SRMConv2D.parameters():
        #     param.requires_grad = False
        if self.use_SRM:
            self.BayarConv2D = nn.Conv2d(3, 3, 5, 1, padding=2, bias=False)
            self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
            self.bayar_mask[2, 2] = 0
            self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
            self.bayar_final[2, 2] = -1
            self.activation = nn.ELU(inplace=True)

        self.init_conv = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=self.in_channels, out_channels=dim, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )


        self.encoder_1 = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        self.encoder_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 8, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        blocks = []
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 8, dilation=2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 8 * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 4 * 2, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2, out_channels=dim * 2,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2 * 2, out_channels=dim, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.ConvTranspose2d(in_channels=dim, out_channels=dim,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        # if self.subtask!=0:
        #     self.mlp_subtask = sequential(
        #         torch.nn.AdaptiveAvgPool2d((1, 1)),
        #         torch.nn.Flatten(),
        #         torch.nn.Linear(dim * 2, dim * 2),
        #         nn.ReLU(),
        #         torch.nn.Linear(dim * 2, self.subtask),
        #         # nn.Sigmoid()
        #     )

        self.decoder_0 = nn.Sequential(
            # spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1),
            #               use_spectral_norm),
            # nn.ELU(),
            # nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=1, padding=0),
        )

        if init_weights:
            self.init_weights()

    def update_clock(self):
        self.clock = min(1.0, self.clock + 1e-4)

    def forward(self, x):
        # x = x.contiguous()
        ## **Bayar constraints**
        if self.use_SRM:
            self.BayarConv2D.weight.data *= self.bayar_mask
            self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
            self.BayarConv2D.weight.data += self.bayar_final

            # Symmetric padding
            # x = symm_pad(x, (2, 2, 2, 2))

            # conv_init = self.vanillaConv2D(x)
            conv_bayar = self.BayarConv2D(x[:,:3])
            # conv_srm = self.SRMConv2D(x)

            first_block = conv_bayar #torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
            e0 = self.activation(first_block)
            if self.in_channels>3:
                e0 = torch.cat((e0,x[:,3:]),dim=1)
        else:
            e0 = x
        e0 = self.init_conv(e0)

        # e0 = self.encoder_0(x)
        e1 = self.encoder_1(e0)
        # e1_add = self.encoder_1_add(e1)
        # e1 = e1_add #self.clock*e1_add+(1-self.clock)*e1
        e2 = self.encoder_2(e1)
        # e2_add = self.encoder_2_add(e2)
        # e2 = e2_add #self.clock * e2_add + (1 - self.clock) * e2
        e3 = self.encoder_3(e2)
        # e3_add = self.encoder_3_add(e3)
        # e3 = e3_add  # self.clock * e2_add + (1 - self.clock) * e2

        m = self.middle(e3)
        if self.output_middle_feature:
            middle_feat = self.avgpool(m)
            middle_feat = middle_feat.reshape(middle_feat.shape[0], -1)

        d3 = self.decoder_3(torch.cat((e3, m), dim=1))
        # d3_add = self.decoder_3_add(d3)
        # d3 = d3_add #self.clock * d2_add + (1 - self.clock) * d2
        d2 = self.decoder_2(torch.cat((e2, d3), dim=1))
        # d2_add = self.decoder_2_add(d2)
        # d2 = d2_add #self.clock * d2_add + (1 - self.clock) * d2
        d1 = self.decoder_1(torch.cat((e1, d2), dim=1))
        # d1_add = self.decoder_1_add(d1)
        # d1 = d1_add #self.clock * d1_add + (1 - self.clock) * d1

        d0_concat = torch.cat((e0, d1), dim=1)

        x = self.decoder_0(d0_concat)
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        if self.output_middle_feature:
            return x, middle_feat
        else:
            return x


class Conditional_Norm(nn.Module):
    def __init__(self, in_channels=64):
        super(Conditional_Norm, self).__init__()
        # out_channels = in_channels

        # self.res = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), True)
        # conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

        self.conv_sn_1 = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1), True)
        self.act = nn.ELU(inplace=True)
        self.conv_sn_2 = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1), True)

        self.shared = sequential(torch.nn.Linear(1, in_channels), nn.ReLU())
        self.to_gamma_1 = sequential(torch.nn.Linear(in_channels, in_channels), nn.Sigmoid())
        self.to_beta_1 = sequential(torch.nn.Linear(in_channels, in_channels), nn.Tanh())
        self.to_gamma_2 = sequential(torch.nn.Linear(in_channels, in_channels), nn.Sigmoid())
        self.to_beta_2 = sequential(torch.nn.Linear(in_channels, in_channels), nn.Tanh())

    def forward(self, x, label):
        actv = self.shared(label)
        gamma_1, beta_1 = self.to_gamma_1(actv).unsqueeze(-1).unsqueeze(-1), self.to_beta_1(actv).unsqueeze(-1).unsqueeze(-1)
        gamma_2, beta_2 = self.to_gamma_2(actv).unsqueeze(-1).unsqueeze(-1), self.to_beta_2(actv).unsqueeze(-1).unsqueeze(-1)

        x_1 = self.act(gamma_1 * self.conv_sn_1(x) + beta_1)
        x_2 = self.act(gamma_2 * self.conv_sn_1(x_1) + beta_2)
        return x + x_2

class SPADE_UNet(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=3, residual_blocks=4, init_weights=True, use_spectral_norm=True,
                  dim=16, use_sigmoid=False):
        super(SPADE_UNet, self).__init__()
        # dim = 32

        self.use_sigmoid = use_sigmoid
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.init_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=self.in_channels, out_channels=dim, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.encoder_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        self.encoder_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 8, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        blocks = []
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 8, dilation=2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 8 * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            # spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            # nn.ELU(inplace=True),
            # spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            # nn.ELU(inplace=True),
        )

        self.decoder_3_condition = Conditional_Norm(in_channels=dim * 4)

        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 4 * 2, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            # spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2, out_channels=dim * 2,  kernel_size=3, padding=1), use_spectral_norm),
            # nn.ELU(inplace=True),
            # spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            # nn.ELU(inplace=True),
        )

        self.decoder_2_condition = Conditional_Norm(in_channels=dim * 2)

        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2 * 2, out_channels=dim, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            # spectral_norm(nn.ConvTranspose2d(in_channels=dim, out_channels=dim,  kernel_size=3, padding=1), use_spectral_norm),
            # nn.ELU(inplace=True),
            # spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            # nn.ELU(inplace=True),
        )

        self.decoder_1_condition = Conditional_Norm(in_channels=dim)


        self.decoder_0 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=self.out_channels, kernel_size=1, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x_input, label):

        e0 = self.init_conv(x_input)

        # e0 = self.encoder_0(x)
        e1 = self.encoder_1(e0)
        # e1_add = self.encoder_1_add(e1)
        # e1 = e1_add #self.clock*e1_add+(1-self.clock)*e1
        e2 = self.encoder_2(e1)
        # e2_add = self.encoder_2_add(e2)
        # e2 = e2_add #self.clock * e2_add + (1 - self.clock) * e2
        e3 = self.encoder_3(e2)
        # e3_add = self.encoder_3_add(e3)
        # e3 = e3_add  # self.clock * e2_add + (1 - self.clock) * e2

        m = self.middle(e3)

        d3 = self.decoder_3(torch.cat((e3, m), dim=1))
        d3 = self.decoder_3_condition(d3, label)
        # d3_add = self.decoder_3_add(d3)
        # d3 = d3_add #self.clock * d2_add + (1 - self.clock) * d2
        d2 = self.decoder_2(torch.cat((e2, d3), dim=1))
        d2 = self.decoder_2_condition(d2, label)
        # d2_add = self.decoder_2_add(d2)
        # d2 = d2_add #self.clock * d2_add + (1 - self.clock) * d2
        d1 = self.decoder_1(torch.cat((e1, d2), dim=1))
        d1 = self.decoder_1_condition(d1, label)
        # d1_add = self.decoder_1_add(d1)
        # d1 = d1_add #self.clock * d1_add + (1 - self.clock) * d1
        x = self.decoder_0(torch.cat((e0, d1), dim=1))
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x+x_input

class QF_predictor(BaseNetwork):
    def __init__(self, in_channels=3, classes=6, residual_blocks=8, init_weights=True, use_spectral_norm=True,
                 use_SRM=True, with_attn=False, dim=3, use_sigmoid=False):
        super(QF_predictor, self).__init__()
        # dim = 32
        self.use_SRM = use_SRM
        self.use_sigmoid = use_sigmoid
        self.with_attn = with_attn

        # self.vanillaConv2D = nn.Conv2d(in_channels, 4, 5, 1, padding=2, bias=False)
        # self.SRMConv2D = nn.Conv2d(in_channels, 9, 5, 1, padding=2, bias=False)
        # self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
        # ##SRM filters (fixed)
        # for param in self.SRMConv2D.parameters():
        #     param.requires_grad = False
        #
        # self.BayarConv2D = nn.Conv2d(in_channels, 3, 5, 1, padding=2, bias=False)
        # self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        # self.bayar_mask[2, 2] = 0
        # self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        # self.bayar_final[2, 2] = -1
        # self.activation = nn.GELU()

        self.down1 = HaarDownsampling([[dim]])
        self.down2 = HaarDownsampling([[dim * 4]])
        self.down3 = HaarDownsampling([[dim * 16]])
        self.encoder_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 4),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 4),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 4),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 4),
            nn.GELU(),
        )
        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 16, out_channels=dim * 16, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 16),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 16, out_channels=dim * 16, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 16),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 16, out_channels=dim * 16, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 16),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 16, out_channels=dim * 16, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 16),
            nn.GELU()
        )

        self.encoder_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 64, out_channels=dim * 64, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 64),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 64, out_channels=dim * 64, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 64),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 64, out_channels=dim * 64, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 64),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 64, out_channels=dim * 64, kernel_size=3, padding=1),use_spectral_norm),
            # nn.BatchNorm2d(dim * 64),
            nn.GELU()
        )

        blocks = []
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 64, dilation=2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.qf_pred = sequential(
                                  torch.nn.AdaptiveAvgPool2d((1, 1)),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(dim * 64, dim * 64),
                                  nn.BatchNorm1d(dim * 64),
                                  nn.GELU(),
                                  # torch.nn.Linear(dim * 64, dim * 64),
                                  # nn.GELU(),
                                  torch.nn.Linear(dim * 64, classes),
                                  )


        if init_weights:
            self.init_weights()

    def forward(self, x):
        # x = x.contiguous()
        ## **Bayar constraints**

        # self.BayarConv2D.weight.data *= self.bayar_mask
        # self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        # self.BayarConv2D.weight.data += self.bayar_final
        #
        # conv_init = self.vanillaConv2D(x)
        # conv_bayar = self.BayarConv2D(x)
        # conv_srm = self.SRMConv2D(x)
        #
        # first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        # e0 = self.activation(first_block)

        x_down1 = self.down1(x)
        x_encoder_1 = self.encoder_1(x_down1)
        x_down2 = self.down2(x_encoder_1)
        x_encoder_2 = self.encoder_2(x_down2)
        x_down3 = self.down3(x_encoder_2)
        x_encoder_3 = self.encoder_3(x_down3)

        x_mid = self.middle(x_encoder_3)

        qf = self.qf_pred(x_mid)

        return qf



from .invertible_net import HaarDownsampling, HaarUpsampling
class JPEGGenerator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=3, residual_blocks=8, init_weights=True, use_spectral_norm=True, use_SRM=True, with_attn=False, additional_conv=False, dim=32):
        super(JPEGGenerator, self).__init__()
        dim = 32
        self.use_SRM = False #use_SRM
        self.with_attn = with_attn
        self.additional_conv=additional_conv

        self.down1 = HaarDownsampling([[in_channels]])
        self.down2 = HaarDownsampling([[in_channels*4]])
        self.down3 = HaarDownsampling([[in_channels*16]])

        self.up3 = HaarUpsampling([[in_channels*64]])
        self.up2 = HaarUpsampling([[in_channels * 16]])
        self.up1 = HaarUpsampling([[in_channels*4]])


        self.encoder_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
        )

        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels*16, out_channels=in_channels*16, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels*16, out_channels=in_channels*16, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels*16, out_channels=in_channels*16, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels*16, out_channels=in_channels*16, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
        )

        self.encoder_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels*64, out_channels=in_channels*64, kernel_size=3, stride=1,padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels*64, out_channels=in_channels*64, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels*64, out_channels=in_channels*64, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels*64, out_channels=in_channels*64, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
        )

        blocks = []
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(in_channels*64, dilation=2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels * 64*2, out_channels=in_channels * 64, kernel_size=3, stride=1, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels * 64, out_channels=in_channels * 64, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels * 64, out_channels=in_channels * 64, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels * 64, out_channels=in_channels * 64, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
        )

        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels * 16*2, out_channels=in_channels *16, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels *16, out_channels=in_channels *16, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels *16, out_channels=in_channels *16, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels *16, out_channels=in_channels *16, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
        )

        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels * 4*2, out_channels=in_channels * 4, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels * 4, out_channels=in_channels * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels * 4, out_channels=in_channels * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels * 4, out_channels=in_channels * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
        )

        if init_weights:
            self.init_weights()


        self.attn_1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels * 4, out_channels=in_channels * 4, kernel_size=7, padding=0),
        )

        self.attn_2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels * 16, out_channels=in_channels * 16, kernel_size=7, padding=0),
        )

        self.attn_3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels * 64, out_channels=in_channels * 64, kernel_size=7, padding=0),
        )

        self.qf_embed = nn.Sequential(torch.nn.Linear(1, 512),
                                   nn.ELU(),
                                   torch.nn.Linear(512, 512),
                                   nn.ELU(),
                                   torch.nn.Linear(512, 512),
                                   nn.ReLU()
                                   )
        self.to_gamma_3 = nn.Sequential(torch.nn.Linear(512,in_channels * 64), nn.Sigmoid())
        self.to_beta_3 = nn.Sequential(torch.nn.Linear(512, in_channels * 64), nn.Tanh())
        self.to_gamma_2 = nn.Sequential(torch.nn.Linear(512, in_channels * 16), nn.Sigmoid())
        self.to_beta_2 = nn.Sequential(torch.nn.Linear(512, in_channels * 16), nn.Tanh())
        self.to_gamma_1 = nn.Sequential(torch.nn.Linear(512, in_channels * 4), nn.Sigmoid())
        self.to_beta_1 = nn.Sequential(torch.nn.Linear(512, in_channels * 4), nn.Tanh())

    def forward(self, x, qf):

        qf_embedding = self.qf_embed(qf)
        gamma_3 = self.to_gamma_3(qf_embedding).unsqueeze(-1).unsqueeze(-1)
        beta_3 = self.to_beta_3(qf_embedding).unsqueeze(-1).unsqueeze(-1)
        gamma_2 = self.to_gamma_2(qf_embedding).unsqueeze(-1).unsqueeze(-1)
        beta_2 = self.to_beta_2(qf_embedding).unsqueeze(-1).unsqueeze(-1)
        gamma_1 = self.to_gamma_1(qf_embedding).unsqueeze(-1).unsqueeze(-1)
        beta_1 = self.to_beta_1(qf_embedding).unsqueeze(-1).unsqueeze(-1)

        x_down1 = self.down1(x)
        x_encoder_1 = self.encoder_1(x_down1)
        x_encoder_1_adaIn = (gamma_1) * self.attn_1(x_encoder_1) + beta_1
        x_down2 = self.down2(x_encoder_1)
        x_encoder_2 = self.encoder_2(x_down2)
        x_encoder_2_adIN = (gamma_2) * self.attn_2(x_encoder_2) + beta_2
        x_down3 = self.down3(x_encoder_2)
        x_encoder_3 = self.encoder_3(x_down3)
        x_encoder_3_adIN = (gamma_3) * self.attn_3(x_encoder_3) + beta_3
        x_mid = self.middle(x_encoder_3)

        d2 = self.decoder_3(torch.cat((x_mid,x_encoder_3_adIN),dim=1))
        d2_x = self.up3(d2)
        d1 = self.decoder_2(torch.cat((d2_x,x_encoder_2_adIN),dim=1))
        d1_x = self.up2(d1)
        y = self.decoder_1(torch.cat((d1_x, x_encoder_1_adaIn), dim=1))
        out = self.up1(y)

        return out, [x_mid, d2, d1]



class EdgeGenerator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=1, residual_blocks=8, use_spectral_norm=True, init_weights=True, dims_in=[[3, 64, 64]], down_num=3, block_num=[2, 2, 2]):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            # nn.InstanceNorm2d(64),
            # nn.BatchNorm2d(64, affine=True),
            nn.GELU(),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            # nn.InstanceNorm2d(128),
            # nn.BatchNorm2d(128, affine=True),
            nn.GELU(),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            # nn.InstanceNorm2d(256),
            # nn.BatchNorm2d(256, affine=True),
            nn.GELU()
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            # nn.InstanceNorm2d(128),
            # nn.BatchNorm2d(128, affine=True),
            nn.GELU(),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            # nn.InstanceNorm2d(64),
            # nn.BatchNorm2d(64, affine=True),
            nn.GELU(),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        # x = (torch.tanh(x) + 1) / 2
        return x

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        if use_spectral_norm:
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
                # nn.InstanceNorm2d(dim),
                nn.GELU(),

                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
                # nn.InstanceNorm2d(dim),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,bias=False),
                # nn.BatchNorm2d(dim, affine=True),
                nn.InstanceNorm2d(dim),
                nn.GELU(),

                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,bias=True),
            )


    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

if __name__ == '__main__':
    # input = torch.ones((1,3,256,256)).cuda()
    # # model = JPEGGenerator()
    # model = QF_predictor().cuda()
    # # output = model(input,qf=torch.tensor([[0.2]]))
    # output = model(input)
    # CE_loss = nn.CrossEntropyLoss().cuda()
    # loss = CE_loss(output,torch.tensor([0]).long().cuda())
    # print(output.shape)
    # print(loss)

    with torch.no_grad():
        from thop import profile
        # from lama_models.HWMNet import HWMNet
        begin = torch.cuda.memory_reserved()
        nin, nout = 3, 3
        X = torch.randn(1,nin, 512,512).cuda()
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.memory_reserved())
        # model = SKFF(in_channels=16)
        # X = [torch.randn(1, 16, 64, 64), torch.randn(1, 16, 64, 64), torch.randn(1, 16, 64, 64)]
        # print(X.shape)

        model = UNetDiscriminator().cuda()
        # model = HWMNet(in_chn=1, out_chn=1, wf=32, depth=4, subtask=0, style_control=False, use_dwt=False).cuda()

        # print(torch.cuda.memory_reserved())
        Y = model(X)
        end = torch.cuda.memory_reserved()
        print(Y.shape)
        print(f"memory: {(end-begin)/1024/1024}")
        # from torchstat import stat
        #
        # stat(model, (3, 512, 512))

        flops, params = profile(model, (X,))
        print(flops/1e9)
        print(params/1e6)




