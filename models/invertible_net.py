import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

clock_global = 1.0
augment_global = True

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class Squeeze(nn.Module):
    """ RealNVP squeezing operation layer (cf RealNVP section 3.6; Glow figure 2b):
    For each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of
    shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s × s × 4c tensor """
    def __init__(self):
        super().__init__()

    def forward(self, x, rev):
        x = x
        if not rev:
            B, C, H, W = x.shape
            x = x.reshape(B, C, H // 2, 2, W // 2, 2)  # factor spatial dim
            x = x.permute(0, 1, 3, 5, 2, 4)  # transpose to (B, C, 2, 2, H//2, W//2)
            x = x.reshape(B, 4 * C, H // 2, W // 2)  # aggregate spatial dim factors into channels
            return x
        else:
            B, C, H, W = x.shape
            x = x.reshape(B, C // 4, 2, 2, H, W)  # factor channel dim
            x = x.permute(0, 1, 4, 2, 5, 3)  # transpose to (B, C//4, H, 2, W, 2)
            x = x.reshape(B, C // 4, 2 * H, 2 * W)  # aggregate channel dim factors into spatial dims
            return x


class Unsqueeze(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, rev):
        x = x
        if not rev:
            B, C, H, W = x.shape
            x = x.reshape(B, C // 4, 2, 2, H, W)  # factor channel dim
            x = x.permute(0, 1, 4, 2, 5, 3)  # transpose to (B, C//4, H, 2, W, 2)
            x = x.reshape(B, C // 4, 2 * H, 2 * W)  # aggregate channel dim factors into spatial dims
            return x
        else:
            B, C, H, W = x.shape
            x = x.reshape(B, C, H // 2, 2, W // 2, 2)  # factor spatial dim
            x = x.permute(0, 1, 3, 5, 2, 4)  # transpose to (B, C, 2, 2, H//2, W//2)
            x = x.reshape(B, 4 * C, H // 2, W // 2)  # aggregate spatial dim factors into channels
            return x


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """
    def __init__(self, n_channels):
        super().__init__()
        self.net = DenseBlock(n_channels, 4*n_channels)  # computes the parameters of Gaussian
        self.clamp = 1.
        self.affine_eps = 0.0001

    def forward(self, x1, x2, rev=False):
        if not rev:
            h = self.net(x1)
            m, s = h[:, 0::2, :, :], h[:, 1::2, :, :]          # split along channel dims
            z2 = (x2 - m) / self.e(s)                # center and scale; log prob is computed at the model forward
            return z2
        else:
            z2 = x2
            h = self.net(x1)
            m, s = h[:, 0::2, :, :], h[:, 1::2, :, :]
            x2 = m + z2 * self.e(s)
            return x2

    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps


class RNVPCouplingBlock(nn.Module):
    '''
    Coupling Block following the RealNVP design.
    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    clamp:              Soft clamping for the multiplicative component. The amplification or attenuation
                        of each input dimension can be at most ±exp(clamp).
    '''

    def __init__(self, dims_in, subnet_constructor=None, clamp=1.0, clock=1):
        super().__init__()
        self.clock = clock
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.affine_eps = 0.0001
        # self.max_s = exp(clamp)
        # self.min_s = exp(-clamp)

        self.s1 = subnet_constructor(self.split_len1, self.split_len2, clock=self.clock)
        self.t1 = subnet_constructor(self.split_len1, self.split_len2, clock=self.clock)
        self.s2 = subnet_constructor(self.split_len2, self.split_len1, clock=self.clock)
        self.t2 = subnet_constructor(self.split_len2, self.split_len1, clock=self.clock)

    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            x2_c = x2
            s2, t2 = self.s2(x2_c), self.t2(x2_c)
            y1 = self.e(s2) * x1 + t2
            y1_c = y1
            s1, t1 = self.s1(y1_c), self.t1(y1_c)
            y2 = self.e(s1) * x2 + t1
            self.last_s = [s1, s2]
        else:
            # names of x1 and y1 are swapped!
            x1_c = x1
            s1, t1 = self.s1(x1_c), self.t1(x1_c)
            y2 = (x2 - t1) / self.e(s1)
            y2_c = y2
            s2, t2 = self.s2(y2_c), self.t2(y2_c)
            y1 = (x1 - t2) / self.e(s2)
            self.last_s = [s1, s2]

        return torch.cat((y1, y2), 1)


class HaarDownsampling(nn.Module):
    '''Uses Haar wavelets to split each channel into 4 channels, with half the
    width and height.'''

    def __init__(self, dims_in, order_by_wavelet=False, rebalance=1.):
        super().__init__()

        self.in_channels = dims_in[0][0]
        self.fac_fwd = 0.5 * rebalance
        self.fac_rev = 0.5 / rebalance
        self.haar_weights = torch.ones(4,1,2,2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

        self.permute = order_by_wavelet
        self.last_jac = None

        if self.permute:
            permutation = []
            for i in range(4):
                permutation += [i+4*j for j in range(self.in_channels)]

            self.perm = torch.LongTensor(permutation)
            self.perm_inv = torch.LongTensor(permutation)

            for i, p in enumerate(self.perm):
                self.perm_inv[p] = i

    def forward(self, x, rev=False):
        if not rev:
            # self.last_jac = self.elements / 4 * (np.log(16.) + 4 * np.log(self.fac_fwd))
            out = F.conv2d(x, self.haar_weights,
                           bias=None, stride=2, groups=self.in_channels)
            if self.permute:
                return out[:, self.perm] * self.fac_fwd
            else:
                return out * self.fac_fwd

        else:
            # self.last_jac = self.elements / 4 * (np.log(16.) + 4 * np.log(self.fac_rev))
            if self.permute:
                x_perm = x[:, self.perm_inv]
            else:
                x_perm = x

            return F.conv_transpose2d(x_perm * self.fac_rev, self.haar_weights,
                                     bias=None, stride=2, groups=self.in_channels)

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return self.last_jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c*4, w//2, h//2
        self.elements = c*w*h
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return (c2, w2, h2)


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
            return F.conv2d(x, self.haar_weights,
                             bias=None, stride=2, groups=self.in_channels)
        else:
            return F.conv_transpose2d(x, self.haar_weights,
                                       bias=None, stride=2,
                                       groups=self.in_channels)

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c//4, w*2, h*2
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return (c2, w2, h2)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True, clock=1):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.act = nn.ELU(inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

class ResBlock_light(nn.Module):
    def __init__(self, channel_in, channel_out,use_spectral_norm=False,init='xavier', dilation=1, clock=1):
        super(ResBlock_light, self).__init__()
        feature = 64
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel_in, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        # self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )

        self.conv5 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        global clock_global
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        residual = self.conv4(residual)
        input = torch.cat((x, residual), dim=1)
        out = self.conv5(input)

        return out

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out,use_spectral_norm=False,init='xavier', dilation=1, clock=1):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel_in, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        # self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=3, padding=1*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=5, padding=2*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        # self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=5, padding=2*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=5, padding=2*dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv4_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature, feature, kernel_size=5, padding=2 * dilation, dilation=dilation), use_spectral_norm),
            # nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ELU(inplace=True)
        )
        self.conv5 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)
        if init == 'xavier':
            initialize_weights_xavier([self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1], 0.1)
        else:
            initialize_weights([self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1], 0.1)

    def forward(self, x):
        global clock_global
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        residual = self.conv4(residual)

        temp = residual
        temp = self.conv1_1(temp)
        temp = self.conv2_1(temp)
        temp = self.conv3_1(temp)
        temp = self.conv4_1(temp)
        ## newly added
        residual = clock_global*temp+residual*(1-temp)
        input = torch.cat((x, residual), dim=1)
        out = self.conv5(input)

        return out


class Inveritible_Decolorization(nn.Module):
    def __init__(self, dims_in=[[3, 64, 64]], down_num=3, block_num=[8,8,8], subnet_constructor=DenseBlock):
        super(Inveritible_Decolorization, self).__init__()
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
                b = RNVPCouplingBlock(current_dims, subnet_constructor=subnet_constructor, clamp=1.0)
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
                b = RNVPCouplingBlock(current_dims, subnet_constructor=DenseBlock, clamp=1.0) # subnet_constructor
                operations.append(b)

        self.operations = nn.ModuleList(operations)
        # self.guassianize = Gaussianize(1)

    def forward(self, x, rev=False):
        out = x
        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
            # g, z = out[0][:, [0], :, :], out[0][:, 1:, :, :]
            # z = self.guassianize(x1=g, x2=z, rev=rev)
            # out = [torch.cat((g, z), dim=1)]
        else:
            # g, z = out[0][:, [0], :, :], out[0][:, 1:, :, :]
            # z = self.guassianize(x1=g, x2=z, rev=rev)
            # out = [torch.cat((g, z), dim=1)]
            for op in reversed(self.operations):
                out = op.forward(out, rev)
        return out

class Haar_UNet(nn.Module):
    def __init__(self, down_num=3, block_num=[1,1,1], subnet_constructor=ResBlock_light):
        super(Haar_UNet, self).__init__()

        operations_Up, operations_Down = [], []
        operations_Middle = []
        current_dims = [[3]]
        for i in range(down_num):
            b = HaarDownsampling(current_dims)
            operations_Down.append(b)
            current_dims[0][0] = current_dims[0][0] * 4
            for j in range(block_num[i]):
                b = subnet_constructor(current_dims[0][0],current_dims[0][0])
                operations_Down.append(b)

        ## middle layer
        for j in range(1):
            b = subnet_constructor(current_dims[0][0],current_dims[0][0],dilation=2)
            operations_Middle.append(b)

        block_num = block_num[:-1][::-1]
        block_num.append(0)

        for i in range(down_num):
            b = HaarUpsampling(current_dims)
            operations_Up.append(b)
            current_dims[0][0] = current_dims[0][0] // 4
            for j in range(block_num[i]):
                b = subnet_constructor(current_dims[0][0], current_dims[0][0])
                operations_Up.append(b)

        self.operations_up = nn.ModuleList(operations_Up)
        self.operations_down = nn.ModuleList(operations_Down)
        self.operations_middle = nn.ModuleList(operations_Middle)

        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        out = x
        for op in self.operations_down:
            out = op.forward(out)

        ## 20220626 added middle layer
        temp = out
        for op in self.operations_middle:
            temp = op.forward(temp)
        out = temp  # (1 - self.clock) * out + self.clock * temp

        for op in self.operations_up:
            out = op.forward(out)
        out = self.final_layer(out)
        return out


class Inveritible_Decolorization_CSVT(nn.Module):
    def __init__(self, dims_in=[[3, 64, 64]], down_num=3, block_num=[8,8,8], subnet_constructor=ResBlock, clock=1, augment=False,
                 middile_block=4, middle_block_constructor=ResBlock
                 ):
        super(Inveritible_Decolorization_CSVT, self).__init__()
        global clock_global
        clock_global = clock
        global augment_global
        augment_global = augment
        self.augment = augment
        self.clock = clock
        operations_Up, operations_Down = [], []
        operations_Middle = []

        current_dims = dims_in
        for i in range(down_num):
            b = HaarDownsampling(current_dims)
            # b = Squeeze()
            operations_Down.append(b)
            current_dims[0][0] = current_dims[0][0] * 4
            current_dims[0][1] = current_dims[0][1] // 2
            current_dims[0][2] = current_dims[0][2] // 2
            for j in range(block_num[i]):
                b = RNVPCouplingBlock(current_dims, subnet_constructor=subnet_constructor, clamp=1.0, clock=self.clock)
                operations_Down.append(b)

        ## 20220626 added middle layer
        ## middle layer
        if augment:
            for j in range(middile_block):
                b = RNVPCouplingBlock(current_dims, subnet_constructor=middle_block_constructor, clamp=1.0, clock=self.clock)
                operations_Middle.append(b)

        block_num = block_num[:-1][::-1]
        block_num.append(0)
        for i in range(down_num):
            b = HaarUpsampling(current_dims)
            # b = Unsqueeze()
            operations_Up.append(b)
            current_dims[0][0] = current_dims[0][0] // 4
            current_dims[0][1] = current_dims[0][1] * 2
            current_dims[0][2] = current_dims[0][2] * 2
            for j in range(block_num[i]):
                b = RNVPCouplingBlock(current_dims, subnet_constructor=subnet_constructor, clamp=1.0, clock=self.clock)
                operations_Up.append(b)

        self.operations_middle = nn.ModuleList(operations_Middle)
        self.operations_up = nn.ModuleList(operations_Up)
        self.operations_down = nn.ModuleList(operations_Down)
        # self.guassianize = Gaussianize(1)

    def update_clock(self):
        self.clock = min(1.0, self.clock + 1e-4)

    def forward(self, x, rev=False):
        out = x
        if not rev:

            for op in self.operations_down:
                temp = op.forward(out, rev)
                out = temp

            ## 20220626 added middle layer
            if self.augment:
                temp = out
                for op in self.operations_middle:
                    temp = op.forward(temp, rev)
                out = (1 - self.clock) * out + self.clock * temp

            for op in self.operations_up:
                temp = op.forward(out, rev)
                out = temp


            return out
            # g, z = out[0][:, [0], :, :], out[0][:, 1:, :, :]
            # z = self.guassianize(x1=g, x2=z, rev=rev)
            # out = [torch.cat((g, z), dim=1)]
        else:
            # g, z = out[0][:, [0], :, :], out[0][:, 1:, :, :]
            # z = self.guassianize(x1=g, x2=z, rev=rev)
            # out = [torch.cat((g, z), dim=1)]


            for op in reversed(self.operations_up):
                temp = op.forward(out, rev)
                out = temp

            out_middle = out.clone()

            ## 20220626 added middle layer
            temp = out
            for op in reversed(self.operations_middle):
                temp = op.forward(temp, rev)
            out = (1 - self.clock) * out + self.clock * temp


            for op in reversed(self.operations_down):
                temp = op.forward(out, rev)
                out = temp


            return out, out_middle


class Inveritible_Decolorization_PAMI(nn.Module):
    def __init__(self, dims_in=[[3, 64, 64]], down_num=3, block_num=[8,8,8], subnet_constructor=ResBlock, clock=1, augment=True,
                 middile_block=4, middle_block_constructor=DenseBlock
                 ):
        super(Inveritible_Decolorization_PAMI, self).__init__()
        global clock_global
        clock_global = clock
        global augment_global
        augment_global = augment
        self.augment = augment
        self.clock = clock
        operations_Up, operations_Down = [], []
        operations_Middle = []

        current_dims = dims_in
        for i in range(down_num):
            b = HaarDownsampling(current_dims)
            # b = Squeeze()
            operations_Down.append(b)
            current_dims[0][0] = current_dims[0][0] * 4
            current_dims[0][1] = current_dims[0][1] // 2
            current_dims[0][2] = current_dims[0][2] // 2
            for j in range(block_num[i]):
                b = RNVPCouplingBlock(current_dims, subnet_constructor=subnet_constructor, clamp=1.0, clock=self.clock)
                operations_Down.append(b)

        ## 20220626 added middle layer
        ## middle layer
        if augment:
            for j in range(middile_block):
                b = RNVPCouplingBlock(current_dims, subnet_constructor=middle_block_constructor, clamp=1.0, clock=self.clock)
                operations_Middle.append(b)

        block_num = block_num[:-1][::-1]
        block_num.append(0)
        for i in range(down_num):
            b = HaarUpsampling(current_dims)
            # b = Unsqueeze()
            operations_Up.append(b)
            current_dims[0][0] = current_dims[0][0] // 4
            current_dims[0][1] = current_dims[0][1] * 2
            current_dims[0][2] = current_dims[0][2] * 2
            for j in range(block_num[i]):
                b = RNVPCouplingBlock(current_dims, subnet_constructor=subnet_constructor, clamp=1.0, clock=self.clock)
                operations_Up.append(b)

        if augment:
            self.operations_middle = nn.ModuleList(operations_Middle)
        self.operations_up = nn.ModuleList(operations_Up)
        self.operations_down = nn.ModuleList(operations_Down)
        # self.guassianize = Gaussianize(1)

    def update_clock(self):
        self.clock = min(1.0, self.clock + 1e-4)

    def forward(self, x, rev=False):
        out = x
        if not rev:

            for op in self.operations_down:
                temp = op.forward(out, rev)
                out = temp

            ## 20220626 added middle layer
            if self.augment:
                temp = out
                for op in self.operations_middle:
                    temp = op.forward(temp, rev)
                out = (1 - self.clock) * out + self.clock * temp

            for op in self.operations_up:
                temp = op.forward(out, rev)
                out = temp


            return out
            # g, z = out[0][:, [0], :, :], out[0][:, 1:, :, :]
            # z = self.guassianize(x1=g, x2=z, rev=rev)
            # out = [torch.cat((g, z), dim=1)]
        else:
            # g, z = out[0][:, [0], :, :], out[0][:, 1:, :, :]
            # z = self.guassianize(x1=g, x2=z, rev=rev)
            # out = [torch.cat((g, z), dim=1)]


            for op in reversed(self.operations_up):
                temp = op.forward(out, rev)
                out = temp

            out_middle = out.clone()

            ## 20220626 added middle layer
            if self.augment:
                temp = out
                for op in reversed(self.operations_middle):
                    temp = op.forward(temp, rev)
                out = (1 - self.clock) * out + self.clock * temp


            for op in reversed(self.operations_down):
                temp = op.forward(out, rev)
                out = temp


            return out, out_middle


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    img_name = "/media/ruizhao/programs/datasets/Denoising/testset/Kodak24/kodim04.png"

    image_c = cv2.imread(img_name, cv2.IMREAD_COLOR)[:, :, ::-1] / 255
    image_c = cv2.resize(image_c, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
    tensor_c = torch.from_numpy(image_c.transpose((2, 0, 1)).astype(np.float32))

    batch_c = tensor_c.unsqueeze(0)

    net = Inveritible_Decolorization()
    net.eval()
    net = net.cuda()

    with torch.no_grad():
        batch_x = net(x=[batch_c.cuda()], rev=False)
        batch_y = net(x=batch_x, rev=True)[0]

    print((batch_y.cpu() - batch_c).sum())

    plt.figure(0)
    plt.imshow(image_c)
    plt.show()
    r = batch_y.cpu().numpy()[0, :, :, :]
    plt.figure(1)
    plt.imshow(r.transpose((1, 2, 0)))
    plt.show()
    print("done")
