import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rawpy
# x = torch.Tensor([[[
#     [0, 0, 0, 0],
#     [0, 1, 0, 1],
#     [0, 0, 0, 0],
#     [0, 1, 0, 1]]]])
#
#
# pad = nn.ReflectionPad2d(padding=1)
# print(pad(x))


class Bayer_demosaic():
    def __init__(self, width_height):
        self.width_height = width_height
        """
        'RGGB': 0,
        'GBRG': 1,
        'BGGR': 2,
        'GRBG': 3
        """
        self.kernel_RAW_k0 = torch.tensor([[[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]], device="cuda",
                                          requires_grad=False).unsqueeze(0)
        self.kernel_RAW_k1 = torch.tensor([[[0, 0], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [0, 0]]], device="cuda",
                                          requires_grad=False).unsqueeze(0)
        self.kernel_RAW_k2 = torch.tensor([[[0, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, 0]]], device="cuda",
                                          requires_grad=False).unsqueeze(0)
        self.kernel_RAW_k3 = torch.tensor([[[0, 1], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [1, 0]]], device="cuda",
                                          requires_grad=False)
        expand_times = int(self.width_height // 2)
        self.kernel_RAW_k0 = self.kernel_RAW_k0.repeat(1, 1, expand_times, expand_times)
        self.kernel_RAW_k1 = self.kernel_RAW_k1.repeat(1, 1, expand_times, expand_times)
        self.kernel_RAW_k2 = self.kernel_RAW_k2.repeat(1, 1, expand_times, expand_times)
        self.kernel_RAW_k3 = self.kernel_RAW_k3.repeat(1, 1, expand_times, expand_times)

        self.conv_R = torch.tensor([[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]]).unsqueeze(0)
        self.conv_G = torch.tensor([[[0., 0.25, 0.], [0.25, 1.0, 0.25], [0., 0.25, 0.]]]).unsqueeze(0)
        self.conv_B = torch.tensor([[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]]).unsqueeze(0)
        conv_RGB = torch.cat([self.conv_R, self.conv_G, self.conv_B], dim=0)
        conv_weight = nn.Parameter(conv_RGB, requires_grad=False).cuda()
        self.bilinear_conv_weight = conv_weight
        self.pad = nn.ReflectionPad2d(padding=1).cuda()

    def bilinear_demosaic(self, input_x, bayer_pattern):
        batch_size = input_x.shape[0]
        used_kernels = [getattr(self, f"kernel_RAW_k{bayer_pattern[idx].item()}") for idx in range(batch_size)]
        used_kernels = torch.cat(used_kernels, dim=0)
        transfer_input = input_x.repeat(1, 3, 1, 1) * used_kernels
        padding_input = self.pad(transfer_input)
        result = F.conv2d(padding_input, self.bilinear_conv_weight, stride=1, padding=0, groups=3)
        return result


# def construct_mask():
#     pass
# de_raw = imageio.imread('./test_deraw.png')
# me = imageio.imread('./test_bil.png')
# exit(0)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
bayers = torch.tensor([0], dtype=torch.int).cuda()
origin = rawpy.imread('/ssd/FiveK_Dataset/NIKON_D700/DNG/a0024-_DSC8932.dng')
raw0 = np.load('/ssd/FiveK_Dataset/NIKON_D700/RAW_UINT16/a0024-_DSC8932.npz')
cwb = raw0['wb']
raw0 = raw0['raw']
cwb = cwb[:3]
cwb = cwb / cwb.max()
import colour_demosaicing
raw = (raw0 / 16383)[:512, :512]
raw = raw.astype(np.float32)
de_raw = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, 'RGGB')
de_raw = (de_raw * 255).astype(np.uint8)
imageio.imwrite('./test_deraw.png', de_raw)
input_x = torch.tensor(raw).unsqueeze(0).unsqueeze(0).cuda()
bilinear = Bayer_demosaic(512)
ret = bilinear.bilinear_demosaic(input_x, bayers)
ret = ret.squeeze(0).permute(1, 2, 0).cpu().numpy()
ret = ret * 255
ret = ret.astype(np.uint8)
imageio.imwrite('./test_bil.png', ret)
print('1')