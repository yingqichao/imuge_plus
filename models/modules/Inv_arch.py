import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from .Subnet_constructor import DenseBlock, ResBlock


# class InvBlock(nn.Module):
#     def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
#         super(InvBlock, self).__init__()
#         # channel_num: 3
#         # channel_split_num: 1
#
#         self.split_len1 = channel_split_num  # 1
#         self.split_len2 = channel_num - channel_split_num  # 2
#
#         self.clamp = clamp
#
#         self.F = subnet_constructor(self.split_len2, self.split_len1)
#         self.G = subnet_constructor(self.split_len1, self.split_len2)
#         self.H = subnet_constructor(self.split_len1, self.split_len2)
#
#         in_channels = 3
#         self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
#         self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
#
#     def forward(self, x, rev=False):
#         if not rev:
#             # invert1x1conv
#             x, logdet = self.flow_permutation(x, logdet=0, rev=False)
#
#             # split to 1 channel and 2 channel.
#             x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
#
#             y1 = x1 + self.F(x2)  # 1 channel
#             self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
#             y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
#             out = torch.cat((y1, y2), 1)
#         else:
#             # split.
#             x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
#             self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
#             y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
#             y1 = x1 - self.F(y2)
#
#             x = torch.cat((y1, y2), 1)
#
#             # inv permutation
#             out, logdet = self.flow_permutation(x, logdet=0, rev=True)
#
#         return out

class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class HaarUpsampling(nn.Module):
    '''Uses Haar wavelets to merge 4 channels into one, with double the
    width and height.'''

    def __init__(self, dims_in):
        super().__init__()

        self.in_channels = dims_in[0][0] // 4
        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights *= 0.5
        self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if rev:
            return [F.conv2d(x[0], self.haar_weights,
                             bias=None, stride=2, groups=self.in_channels)]
        else:
            return [F.conv_transpose2d(x[0], self.haar_weights,
                                       bias=None, stride=2,
                                       groups=self.in_channels)]

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c//4, w*2, h*2
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac

class AttackNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3):
        super(AttackNet, self).__init__()
        # print("Attack Block_num:{}".format(block_num))
        # self.tanh = nn.Tanh()
        operations,operations_inverse = [],[]

        current_channel = channel_in
        down_num=2
        for i in range(down_num):
            ######## debug remove downsampling ########
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            ######## end debug ########################

            for j in range(4):
                b = DenseBlock(current_channel, current_channel)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

        ######## debug reversed upsampling layer ########
        self.with_reverse = True
        if self.with_reverse:
            current_channel = channel_in
            for i in range(down_num):
                ######## debug remove downsampling ########
                b = HaarDownsampling(current_channel)
                operations_inverse.append(b)
                current_channel *= 4
                ######## end debug ########################

                for j in range(4):
                    b = DenseBlock(current_channel, current_channel)
                    operations_inverse.append(b)

            self.operations_inverse = nn.ModuleList(operations_inverse)

    def forward(self, x, rev=False):
        out = x

        if not rev:
            for op in self.operations:
                out = op.forward(out)

            ###### debug inversed
            if self.with_reverse:
                for op in reversed(self.operations_inverse):
                    out = op.forward(out)

        else:
            ###### debug inversed
            if self.with_reverse:
                for op in self.operations_inverse:
                    out = op.forward(out)


            for op in reversed(self.operations):
                out = op.forward(out)


        return out

class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[8,8,8,8], down_num=2):
        super(InvRescaleNet, self).__init__()
        print("Block_num:{}".format(block_num))
        self.tanh = nn.Tanh()

        operations,operations_inverse = [],[]

        current_channel = channel_in
        for i in range(down_num):
            ######## debug remove downsampling ########
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            ######## end debug ########################

            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

        ######## debug reversed upsampling layer ########
        self.with_reverse = True
        if self.with_reverse:
            current_channel = channel_in
            for i in range(down_num):
                ######## debug remove downsampling ########
                b = HaarDownsampling(current_channel)
                operations_inverse.append(b)
                current_channel *= 4
                ######## end debug ########################

                for j in range(block_num[i]):
                    b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                    operations_inverse.append(b)

            self.operations_inverse = nn.ModuleList(operations_inverse)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                # if dist.get_rank()==0:
                #     print(out.shape)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            ###### debug inversed
            if self.with_reverse:
                for op in reversed(self.operations_inverse):
                    out = op.forward(out, not rev)
                    if cal_jacobian:
                        jacobian += op.jacobian(out, not rev)
        else:
            ###### debug inversed
            if self.with_reverse:
                for op in self.operations_inverse:
                    out = op.forward(out, not rev)
                    if cal_jacobian:
                        jacobian += op.jacobian(out, not rev)

            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        # out = self.tanh(out)
        if cal_jacobian:
            return out, jacobian
        else:
            return out

