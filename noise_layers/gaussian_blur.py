
import torch.nn as nn

import torch
from utils.metrics import PSNR
import math
class GaussianBlur(nn.Module):
    '''
    Adds random noise to a tensor.'''

    def __init__(self, kernel_size=5):
        super(GaussianBlur, self).__init__()
        # self.device = config.device
        # self.kernel_size = kernel_size
        self.psnr = PSNR(255.0).cuda()
        self.name = "G_Blur"

    def get_gaussian_kernel(self, kernel_size=5, sigma=2, channels=3):
        # kernel_size = self.kernel_size
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        padding = int((kernel_size-1)/2)
        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, padding=padding, groups=channels, bias=False)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

        return self.gaussian_filter

    def forward(self, tensor, kernel_size=5):
        self.name = "GaussianBlur"
        result = tensor
        for kernel in [3,5,7]:
            gaussian_layer = self.get_gaussian_kernel(kernel).cuda()
            blur_result = gaussian_layer(tensor)
            psnr = self.psnr(self.postprocess(blur_result), self.postprocess(tensor)).item()
            if psnr>=28:
                return blur_result, kernel
        ## if none of the above satisfy psnr>30, we abandon the attack
        # print("abandoned gaussian blur, we cannot find a suitable kernel that satisfy PSNR>=25")
        return result, 0

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

