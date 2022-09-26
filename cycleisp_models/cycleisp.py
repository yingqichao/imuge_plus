import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_gaussian_kernel(kernel_size=21, sigma=5, channels=3):
    #if not kernel_size: kernel_size = int(2*np.ceil(2*sigma)+1)
    #print("Kernel is: ",kernel_size)
    #print("Sigma is: ",sigma)
    padding = kernel_size//2
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter, padding


##########################################################################ssss

def mosaic(images):
    """Extracts RGGB Bayer planes from RGB image."""
    # import pdb;pdb.set_trace()
    shape = images.shape
    red = images[:, 0, 0::2, 0::2]
    green_red = images[:, 1, 0::2, 1::2]
    green_blue = images[:, 1, 1::2, 0::2]
    blue = images[:, 2, 1::2, 1::2]
    images = torch.stack((red, green_red, green_blue, blue), dim=1)
    # images = tf.reshape(images, (shape[0] // 2, shape[1] // 2, 4))
    return images


##########################################################################

def conv(in_channels, out_channels, kernel_size, bias=True, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


##########################################################################


## Dual Attention Block (DAB)
class DAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True)):

        super(DAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        self.body = nn.Sequential(*modules_body)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


##########################################################################


## Recursive Residual Group (RRG)
class RRG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, num_dab):
        super(RRG, self).__init__()
        modules_body = []
        modules_body = [
            DAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act) \
            for _ in range(num_dab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
################ Color Correction Network  #############################
##########################################################################

class CCM(nn.Module):
    def __init__(self,  conv=conv):
        super(CCM, self).__init__()
        input_nc  = 3
        output_nc = 64

        num_rrg = 2
        num_dab = 2
        n_feats = 64
        kernel_size = 3
        reduction = 8

        sigma = 12 ## GAUSSIAN_SIGMA

        act =nn.PReLU(n_feats)


        modules_head = [conv(input_nc, n_feats, kernel_size = kernel_size, stride = 1)]

        modules_downsample = [nn.MaxPool2d(kernel_size=2)]
        self.downsample = nn.Sequential(*modules_downsample)

        modules_body = [
            RRG(
                conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab) \
            for _ in range(num_rrg)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(act)

        modules_tail = [conv(n_feats, output_nc, kernel_size),nn.Sigmoid()]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.blur, self.pad = get_gaussian_kernel(sigma=sigma)


    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x = self.blur(x)
        x = self.head(x)
        # x = self.downsample(x)
        x = self.body(x)
        x = self.tail(x)
        return x


##########################################################################
##########################   RAW2RGB Network  ############################
##########################################################################
class Raw2Rgb(nn.Module):
    def __init__(self, conv=conv):
        super(Raw2Rgb, self).__init__()

        ############# CCM ###################
        input_nc = 3
        output_nc = 64

        num_rrg = 2
        num_dab = 2
        n_feats = 64
        kernel_size = 3
        reduction = 8

        sigma = 12  ## GAUSSIAN_SIGMA

        act = nn.PReLU(n_feats)

        modules_head = [conv(input_nc, n_feats, kernel_size=kernel_size, stride=1)]

        # modules_downsample = [nn.MaxPool2d(kernel_size=2)]
        # self.downsample = nn.Sequential(*modules_downsample)

        modules_body = [
            RRG(
                conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab) \
            for _ in range(num_rrg)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(act)

        modules_tail = [conv(n_feats, output_nc, kernel_size), nn.Sigmoid()]

        self.head_CCM = nn.Sequential(*modules_head)
        self.body_CCM = nn.Sequential(*modules_body)
        self.tail_CCM = nn.Sequential(*modules_tail)
        self.blur, self.pad = get_gaussian_kernel(sigma=sigma)


        ######################################


        input_nc  = 3
        output_nc = 3

        num_rrg = 3
        num_dab =5
        n_feats = 64
        kernel_size = 3
        reduction = 8

        act =nn.PReLU(n_feats)

        modules_head = [conv(input_nc, n_feats, kernel_size = kernel_size, stride = 1)]

        modules_body = [
            RRG(
                conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab) \
            for _ in range(num_rrg)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(act)

        modules_tail = [conv(n_feats, n_feats, kernel_size), act]
        modules_tail_rgb = [conv(n_feats, output_nc, kernel_size=1)]#, nn.Sigmoid()]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.tail_rgb = nn.Sequential(*modules_tail_rgb)

        conv1x1 = [conv(n_feats*2, n_feats, kernel_size=1)]
        self.conv1x1 = nn.Sequential(*conv1x1)


    def forward(self, rgb, raw, ccm_feat=None):
        ######### CCM ###################
        x = F.pad(rgb, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x = self.blur(x)
        x = self.head_CCM(x)
        # x = self.downsample(x)
        x = self.body_CCM(x)
        ccm_feat = self.tail_CCM(x)
        ################################
        x = self.head(raw)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
        body_out = x.clone()
        x = x * ccm_feat          ## Attention
        x = x + body_out
        x = self.body[-1](x)
        x = self.tail(x)
        x = self.tail_rgb(x)
        # x = nn.functional.pixel_shuffle(x, 2)
        return x
