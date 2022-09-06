from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


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

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2): # CBR
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res

# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)



class QFAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(QFAttention, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = (gamma)*self.res(x) + beta
        return x + res


class FBCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[32, 64, 128, 256], nb=4, act_mode='R', downsample_mode='strideconv',qf_classes=6,
                 upsample_mode='convtranspose'):
        super(FBCNN, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')
        self.nb = nb
        self.nc = nc
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = nn.ModuleList([
            downsample_block(nc[0], nc[1], bias=True, mode='2'),
            # *[QFAttention(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],

        ])
        self.m_down2 = nn.ModuleList([
            downsample_block(nc[1], nc[2], bias=True, mode='2'),
        # *[QFAttention(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        ])
        self.m_down3 = nn.ModuleList([
            downsample_block(nc[2], nc[2], bias=True, mode='2'),
        # *[QFAttention(192, 192, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        ])

        self.m_body_encoder = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        # self.m_body_decoder = sequential(
        #     *[ResBlock(192, 192, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = nn.ModuleList([
                                upsample_block(nc[2], nc[2], bias=True, mode='2'),
                                    # *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                                *[QFAttention(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        ])

        self.m_up2 = nn.ModuleList([
                                upsample_block(nc[2], nc[1], bias=True, mode='2'),
                                    # *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                                *[QFAttention(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        ])

        self.m_up1 = nn.ModuleList([
                                upsample_block(nc[1], nc[0], bias=True, mode='2'),
                                    # *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                                *[QFAttention(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        ])


        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')
        dim = 32
        self.qf_downsample = sequential(
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=4, stride=2,
                      padding=1, bias=True),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=4, stride=2,
                      padding=1, bias=True),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=dim*2, out_channels=dim*4, kernel_size=4, stride=2,
                      padding=1, bias=True),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=dim * 4, out_channels=dim * 8, kernel_size=4, stride=2,
                      padding=1, bias=True),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=dim * 8, out_channels=dim * 16, kernel_size=4, stride=2,
                      padding=1, bias=True),

        )
        # self.qf_pred = sequential(
        #                           torch.nn.AdaptiveAvgPool2d((1,1)),
        #                           torch.nn.Flatten(),
        #                           torch.nn.Linear(512, 512),
        #                           nn.GELU(),
        #                           torch.nn.Linear(512, 512),
        #                           nn.GELU(),
        #                           torch.nn.Linear(512, qf_classes),
        #                           nn.Sigmoid()
        #                         )
        #
        self.qf_embed = sequential(torch.nn.Linear(1, 512),
                                  nn.GELU(),
                                  torch.nn.Linear(512, 512),
                                  nn.GELU(),
                                  torch.nn.Linear(512, 512),
                                  nn.GELU()
                                )

        self.to_gamma_3 = sequential(torch.nn.Linear(512, nc[2]),nn.Sigmoid())
        self.to_beta_3 =  sequential(torch.nn.Linear(512, nc[2]),nn.Tanh())
        self.to_gamma_2 = sequential(torch.nn.Linear(512, nc[1]),nn.Sigmoid())
        self.to_beta_2 =  sequential(torch.nn.Linear(512, nc[1]),nn.Tanh())
        self.to_gamma_1 = sequential(torch.nn.Linear(512, nc[0]),nn.Sigmoid())
        self.to_beta_1 =  sequential(torch.nn.Linear(512, nc[0]),nn.Tanh())


    def forward(self, x, qf_input=None):
        qf_embedding = self.qf_embed(qf_input)  # if qf_input is not None else self.qf_embed(qf)
        gamma_3 = self.to_gamma_3(qf_embedding)
        beta_3 = self.to_beta_3(qf_embedding)
        gamma_2 = self.to_gamma_2(qf_embedding)
        beta_2 = self.to_beta_2(qf_embedding)
        gamma_1 = self.to_gamma_1(qf_embedding)
        beta_1 = self.to_beta_1(qf_embedding)

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1[0](x1)
        for i in range(self.nb):
            x2 = self.m_down1[i+1](x2)#, gamma_1,beta_1)

        x3 = self.m_down2[0](x2)
        for i in range(self.nb):
            x3 = self.m_down2[i+1](x3)#, gamma_2,beta_2)

        x4 = self.m_down3[0](x3)
        for i in range(self.nb):
            x4 = self.m_down3[i+1](x4)#, gamma_3,beta_3)


        x_m_1 = self.m_body_encoder(x4)
        # qf = self.qf_pred(x)
        # x = self.m_body_decoder(x)
        x = x_m_1 + x4

        x_m_2 = self.m_up3[0](x)
        for i in range(self.nb):
            x_m_2 = self.m_up3[i + 1](x_m_2, gamma_3,beta_3)
        # x = self.m_up3(x_m_1)
        x = x_m_2 + x3

        x_m_3 = self.m_up2[0](x)
        for i in range(self.nb):
            x_m_3 = self.m_up2[i + 1](x_m_3, gamma_2, beta_2)
        # x = self.m_up2(x_m_2)
        x = x_m_3 + x2

        x_m_4 = self.m_up1[0](x)
        for i in range(self.nb):
            x_m_4 = self.m_up1[i + 1](x_m_4, gamma_1, beta_1)
        # x = self.m_up1(x_m_3)
        x = x_m_4 + x1

        x = self.m_tail(x)
        x = x[..., :h, :w]
        # x_down = self.qf_downsample(x)
        # qf = self.qf_pred(x_down)

        return x, (x_m_1,x_m_2,x_m_3,x_m_4)

class rec_FBCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[32, 64, 128, 256], nb=4, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(rec_FBCNN, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')
        self.nb = nb
        self.nc = nc
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode='2'))
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], 192, bias=True, mode='2'))

        ############# Predict
        # self.m_head_A = conv(in_nc, nc[0], bias=True, mode='C')
        # self.m_down1_A = sequential(
        #     *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        #     downsample_block(nc[0], nc[1], bias=True, mode='2'))
        # self.m_down2_A = sequential(
        #     *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        #     downsample_block(nc[1], nc[2], bias=True, mode='2'))
        # self.m_down3_A = sequential(
        #     *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        #     downsample_block(nc[2], 192, bias=True, mode='2'))
        # self.m_body_encoder_A = sequential(
        #     *[ResBlock(192, 192, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])
        ####################

        self.m_body_encoder = sequential(
            *[ResBlock(192, 192, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_body_decoder = sequential(
            *[ResBlock(192, 192, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = nn.ModuleList([upsample_block(192, nc[2], bias=True, mode='2'),
                                  *[QFAttention(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_up2 = nn.ModuleList([upsample_block(nc[2], nc[1], bias=True, mode='2'),
                                  *[QFAttention(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_up1 = nn.ModuleList([upsample_block(nc[1], nc[0], bias=True, mode='2'),
                                  *[QFAttention(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])


        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')


        # self.qf_pred = sequential(*[ResBlock(192, 192, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        #                           torch.nn.AdaptiveAvgPool2d((1,1)),
        #                           torch.nn.Flatten(),
        #                           torch.nn.Linear(512, 512),
        #                           nn.GELU(),
        #                           torch.nn.Linear(512, 512),
        #                           nn.GELU(),
        #                           torch.nn.Linear(512, 1),
        #                           nn.Sigmoid()
        #                         )

        self.qf_embed = sequential(torch.nn.Linear(1, 512),
                                  nn.GELU(),
                                  torch.nn.Linear(512, 512),
                                  nn.GELU(),
                                  torch.nn.Linear(512, 512),
                                  nn.GELU()
                                )

        self.to_gamma_3 = sequential(torch.nn.Linear(512, nc[2]),nn.Sigmoid())
        self.to_beta_3 =  sequential(torch.nn.Linear(512, nc[2]),nn.Tanh())
        self.to_gamma_2 = sequential(torch.nn.Linear(512, nc[1]),nn.Sigmoid())
        self.to_beta_2 =  sequential(torch.nn.Linear(512, nc[1]),nn.Tanh())
        self.to_gamma_1 = sequential(torch.nn.Linear(512, nc[0]),nn.Sigmoid())
        self.to_beta_1 =  sequential(torch.nn.Linear(512, nc[0]),nn.Tanh())


    def forward(self, x, qf_input=None):

        # h, w = x.size()[-2:]
        # paddingBottom = int(np.ceil(h / 8) * 8 - h)
        # paddingRight = int(np.ceil(w / 8) * 8 - w)
        # x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        ############### We do not predict here. We generate JPEG images
        # qf = self.qf_pred(x)
        x = self.m_body_decoder(x)
        # print(qf_input.shape)
        qf_embedding = self.qf_embed(qf_input) # if qf_input is not None else self.qf_embed(qf)
        gamma_3 = self.to_gamma_3(qf_embedding)
        beta_3 = self.to_beta_3(qf_embedding)

        gamma_2 = self.to_gamma_2(qf_embedding)
        beta_2 = self.to_beta_2(qf_embedding)

        gamma_1 = self.to_gamma_1(qf_embedding)
        beta_1 = self.to_beta_1(qf_embedding)


        x = x + x4
        x = self.m_up3[0](x)
        for i in range(self.nb):
            x = self.m_up3[i+1](x, gamma_3,beta_3)

        x = x + x3

        x = self.m_up2[0](x)
        for i in range(self.nb):
            x = self.m_up2[i+1](x, gamma_2, beta_2)
        x = x + x2

        x = self.m_up1[0](x)
        for i in range(self.nb):
            x = self.m_up1[i+1](x, gamma_1, beta_1)

        x = x + x1
        x = self.m_tail(x)
        x = (torch.tanh(x) + 1) / 2
        # x = x[..., :h, :w]
        #
        # #### pred QF
        # h, w = x.size()[-2:]
        # paddingBottom = int(np.ceil(h / 8) * 8 - h)
        # paddingRight = int(np.ceil(w / 8) * 8 - w)
        # x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        # x1_1 = self.m_head_A(x)
        # x2_1 = self.m_down1_A(x1_1)
        # x3_1 = self.m_down2_A(x2_1)
        # x4_1 = self.m_down3_A(x3_1)
        # x_pred = self.m_body_encoder_A(x4_1)
        # qf = self.qf_pred(x_pred)

        return x


class MantraNet(nn.Module):
    def __init__(self, in_channel=3, eps=10 ** (-6)):
        super(MantraNet, self).__init__()

        self.eps = eps
        self.relu = nn.GELU()

        # ********** IMAGE MANIPULATION TRACE FEATURE EXTRACTOR *********

        ## Initialisation

        self.init_conv = nn.Conv2d(in_channel, 4, 5, 1, padding=0, bias=False)

        self.BayarConv2D = nn.Conv2d(in_channel, 3, 5, 1, padding=0, bias=False)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        self.bayar_mask[2, 2] = 0

        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        self.bayar_final[2, 2] = -1

        self.SRMConv2D = nn.Conv2d(in_channel, 9, 5, 1, padding=0, bias=False)
        self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']

        ##SRM filters (fixed)
        for param in self.SRMConv2D.parameters():
            param.requires_grad = False

        self.middle_and_last_block = nn.ModuleList([
            nn.Conv2d(16, 32, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, 1, padding=0)]
        )

        # # ********** LOCAL ANOMALY DETECTOR *********
        #
        # self.adaptation = nn.Conv2d(256, 64, 1, 1, padding=0, bias=False)
        #
        # self.sigma_F = nn.Parameter(torch.zeros((1, 64, 1, 1)), requires_grad=True)
        #
        # self.pool31 = nn.AvgPool2d(31, stride=1, padding=15, count_include_pad=False)
        # self.pool15 = nn.AvgPool2d(15, stride=1, padding=7, count_include_pad=False)
        # self.pool7 = nn.AvgPool2d(7, stride=1, padding=3, count_include_pad=False)
        #
        # self.convlstm = ConvLSTM(input_dim=64,
        #                          hidden_dim=8,
        #                          kernel_size=(7, 7),
        #                          num_layers=1,
        #                          batch_first=False,
        #                          bias=True,
        #                          return_all_layers=False)

        # self.end = nn.Sequential(nn.Conv2d(8, 1, 7, 1, padding=3), nn.Sigmoid())

        self.qf_pred = sequential(
                                  torch.nn.AdaptiveAvgPool2d((1,1)),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(256, 256),
                                  nn.GELU(),
                                  torch.nn.Linear(256, 256),
                                  nn.GELU(),
                                  torch.nn.Linear(256, 1),
                                  # nn.Sigmoid()
                                )

    def forward(self, x):
        B, nb_channel, H, W = x.shape

        if not (self.training):
            self.GlobalPool = nn.AvgPool2d((H, W), stride=1)
        else:
            if not hasattr(self, 'GlobalPool'):
                self.GlobalPool = nn.AvgPool2d((H, W), stride=1)

        # Normalization
        x = x / 255. * 2 - 1

        ## Image Manipulation Trace Feature Extractor

        ## **Bayar constraints**

        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final

        # Symmetric padding
        x = symm_pad(x, (2, 2, 2, 2))

        conv_init = self.init_conv(x)
        conv_bayar = self.BayarConv2D(x)
        conv_srm = self.SRMConv2D(x)

        first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        first_block = self.relu(first_block)

        last_block = first_block

        for layer in self.middle_and_last_block:

            if isinstance(layer, nn.Conv2d):
                last_block = symm_pad(last_block, (1, 1, 1, 1))

            last_block = layer(last_block)

        # # L2 normalization
        # last_block = F.normalize(last_block, dim=1, p=2)
        #
        # ## Local Anomaly Feature Extraction
        # X_adapt = self.adaptation(last_block)
        # X_adapt = batch_norm(X_adapt)
        #
        # # Z-pool concatenation
        # mu_T = self.GlobalPool(X_adapt)
        # sigma_T = torch.sqrt(self.GlobalPool(torch.square(X_adapt - mu_T)))
        # sigma_T = torch.max(sigma_T, self.sigma_F + self.eps)
        # inv_sigma_T = torch.pow(sigma_T, -1)
        # zpoolglobal = torch.abs((mu_T - X_adapt) * inv_sigma_T)
        #
        # mu_31 = self.pool31(X_adapt)
        # zpool31 = torch.abs((mu_31 - X_adapt) * inv_sigma_T)
        #
        # mu_15 = self.pool15(X_adapt)
        # zpool15 = torch.abs((mu_15 - X_adapt) * inv_sigma_T)
        #
        # mu_7 = self.pool7(X_adapt)
        # zpool7 = torch.abs((mu_7 - X_adapt) * inv_sigma_T)
        #
        # input_lstm = torch.cat(
        #     [zpool7.unsqueeze(0), zpool15.unsqueeze(0), zpool31.unsqueeze(0), zpoolglobal.unsqueeze(0)], axis=0)
        #
        # # Conv2DLSTM
        # _, output_lstm = self.convlstm(input_lstm)
        # output_lstm = output_lstm[0][0]

        final_output = self.qf_pred(last_block)

        return final_output


class Crop_predictor(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[32, 64, 128, 256], nb=4, act_mode='R', downsample_mode='strideconv', classes=6, crop_pred=False,
                 upsample_mode='convtranspose'):
        super(Crop_predictor, self).__init__()
        self.in_nc = in_nc
        self.nb = nb
        self.nc = nc
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))


        ############# Predict
        # self.init_conv = nn.Conv2d(3, 4, 5, 1, padding=2, bias=False)
        #
        # self.SRMConv2D = nn.Conv2d(3, 9, 5, 1, padding=2, bias=False)
        # self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
        # ##SRM filters (fixed)
        # for param in self.SRMConv2D.parameters():
        #     param.requires_grad = False

        self.BayarConv2D = nn.Conv2d(3, 3, 5, 1, padding=2, bias=False)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        self.bayar_mask[2, 2] = 0

        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        self.bayar_final[2, 2] = -1
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.m_head_A = conv(3, nc[0], bias=True, mode='C')
        self.m_down1_A = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2_A = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode='2'))
        self.m_down3_A = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], 192, bias=True, mode='2'))
        self.m_body_encoder_A = sequential(
            *[ResBlock(192, 192, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])
        ####################

        # # upsample
        # if upsample_mode == 'upconv':
        #     upsample_block = upsample_upconv
        # elif upsample_mode == 'pixelshuffle':
        #     upsample_block = upsample_pixelshuffle
        # elif upsample_mode == 'convtranspose':
        #     upsample_block = upsample_convtranspose
        # else:
        #     raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))


        self.to_img = nn.Conv2d(192, 1, 3, 1, padding=1, bias=False)
        self.qf_pred = sequential(
                                  # torch.nn.AdaptiveAvgPool2d((1, 1)),
                                  # torch.nn.Flatten(),
                                  torch.nn.Linear(32*32, 192),
                                  nn.ELU(inplace=True),
                                  torch.nn.Linear(192, 192),
                                  nn.ELU(inplace=True),
                                  torch.nn.Linear(192, classes),
                                  # nn.Sigmoid()
                                  )

    def forward(self, x):
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1),
                                                  -1)
        self.BayarConv2D.weight.data += self.bayar_final

        # conv_init = self.init_conv(x)
        conv_bayar = self.BayarConv2D(x)
        # conv_srm = self.SRMConv2D(x)
        # e0 = conv_bayar
        # first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        # e0 = self.relu(first_block)

        e0 = conv_bayar

        x1_1 = self.m_head_A(e0)
        x2_1 = self.m_down1_A(x1_1)
        x3_1 = self.m_down2_A(x2_1)
        x4_1 = self.m_down3_A(x3_1)
        x_pred = self.m_body_encoder_A(x4_1)

        img = self.to_img(x_pred)
        img_down = F.interpolate(img, size=[32,32], mode='bilinear')
        img_down = img_down.view(img.shape[0],32*32)
        qf = self.qf_pred(img_down)
        # img = F.interpolate(img, size=[x.shape[2],x.shape[3]], mode='bilinear')
        return img,qf


class QF_predictor(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[32, 64, 128, 256], nb=4, act_mode='R', downsample_mode='strideconv', classes=6, crop_pred=False,
                 upsample_mode='convtranspose'):
        super(QF_predictor, self).__init__()
        self.in_nc = in_nc
        self.nb = nb
        self.nc = nc
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        ############# Predict
        # self.SRMConv2D = nn.Conv2d(3, 9, 5, 1, padding=0, bias=False)
        # self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
        # self.init_conv = nn.Conv2d(3, 4, 5, 1, padding=0, bias=False)
        #
        # self.SRMConv2D = nn.Conv2d(3, 9, 5, 1, padding=0, bias=False)
        # self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
        # ##SRM filters (fixed)
        #
        # for param in self.SRMConv2D.parameters():
        #     param.requires_grad = False

        self.BayarConv2D = nn.Conv2d(3, 3, 5, 1, padding=0, bias=False)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        self.bayar_mask[2, 2] = 0

        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        self.bayar_final[2, 2] = -1
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ##SRM filters (fixed)

        # for param in self.SRMConv2D.parameters():
        #     param.requires_grad = False

        self.m_head_A = conv(3, nc[0], bias=True, mode='C')
        self.m_down1_A = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2_A = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode='2'))
        self.m_down3_A = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], 192, bias=True, mode='2'))
        self.m_body_encoder_A = sequential(
            *[ResBlock(192, 192, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])
        ####################

        # # upsample
        # if upsample_mode == 'upconv':
        #     upsample_block = upsample_upconv
        # elif upsample_mode == 'pixelshuffle':
        #     upsample_block = upsample_pixelshuffle
        # elif upsample_mode == 'convtranspose':
        #     upsample_block = upsample_convtranspose
        # else:
        #     raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))


        self.qf_pred = sequential(*[ResBlock(192, 192, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                              torch.nn.AdaptiveAvgPool2d((1,1)),
                              torch.nn.Flatten(),
                              torch.nn.Linear(192, 192),
                              nn.GELU(),
                              torch.nn.Linear(192, 192),
                              nn.GELU(),
                              torch.nn.Linear(192, classes),
                              # nn.Sigmoid()
                            )

    def forward(self, x):
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1),
                                                  -1)
        self.BayarConv2D.weight.data += self.bayar_final


        # Symmetric padding
        x = symm_pad(x, (2, 2, 2, 2))

        # conv_init = self.init_conv(x)
        conv_bayar = self.BayarConv2D(x)
        # conv_srm = self.SRMConv2D(x)
        e0 = conv_bayar
        # first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        # e0 = self.relu(first_block)
        # if self.in_nc==6:
        #     x1 = self.SRMConv2D(x[:,:3,:,:])
        #     x2 = self.SRMConv2D(x[:, 3:, :, :])
        #     x1_1 = self.m_head_A(torch.cat((x1,x2),dim=1))
        # else:
        #     x1 = self.SRMConv2D(x[:, :, :, :])
        x1_1 = self.m_head_A(e0)
        x2_1 = self.m_down1_A(x1_1)
        x3_1 = self.m_down2_A(x2_1)
        x4_1 = self.m_down3_A(x3_1)
        x_pred = self.m_body_encoder_A(x4_1)
        if self.crop_pred:
            img = self.to_img(x_pred)
            qf = self.qf_pred(x_pred)
            img = F.interpolate(img, size=[512,512], mode='bicubic')
            return img,qf
        else:
            qf = self.qf_pred(x_pred)
        # if self.tanh:
        #     return (torch.tanh(qf) + 1) / 2
        # else:
            return qf

class domain_generalization_predictor(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[32, 64, 128, 256], nb=4, act_mode='R', downsample_mode='strideconv', classes=3, upsample_mode='convtranspose'):
        super(domain_generalization_predictor, self).__init__()
        self.in_nc = in_nc
        self.nb = nb
        self.nc = nc
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))


        ############# Predict

        self.qf_pred = sequential(*[ResBlock(256,256, bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                              torch.nn.AdaptiveAvgPool2d((1,1)),
                              torch.nn.Flatten(),
                              torch.nn.Linear(256, 256),
                              nn.ELU(inplace=True),
                              torch.nn.Linear(256, 256),
                              nn.ELU(inplace=True),
                              torch.nn.Linear(256, classes),
                              nn.Sigmoid()
                            )

    def forward(self, x):


        qf = self.qf_pred(x)
        return qf


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