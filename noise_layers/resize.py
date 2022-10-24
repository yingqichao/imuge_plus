import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from utils.metrics import PSNR

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_range=(0.5,1.5), interpolation_method='bilinear', opt=None):
        super(Resize, self).__init__()
        self.name = "Resize"
        self.psnr = PSNR(255.0).cuda()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method
        self.psnr_thresh = 28 if opt is None else opt['minimum_PSNR_caused_by_attack']


    def forward(self, noised_image, resize_ratio=None):

        self.name = "Resize"
        original_width, original_height = noised_image.shape[2], noised_image.shape[3]
        if resize_ratio is None:
            resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)

            newWidth, newHeight = int(resize_ratio*original_width), int(resize_ratio*original_height)
        else:
            newWidth, newHeight = resize_ratio

        while True:

            out = F.interpolate(
                                        noised_image,
                                        size=[newWidth, newHeight],
                                        mode=self.interpolation_method)

            recover = F.interpolate(
                                        out,
                                        size=[original_width, original_height],
                                        # scale_factor=(1/resize_ratio, 1/resize_ratio),
                                        mode=self.interpolation_method)

            # resize_back = F.interpolate(
            #     noised_image,
            #     size=[original_width, original_height],
            #     recompute_scale_factor=True,
            #     mode='nearest')
            recover = torch.clamp(recover, 0, 1)
            psnr = self.psnr(self.postprocess(recover), self.postprocess(noised_image)).item()
            if psnr>=self.psnr_thresh:
                break
            else:
                resize_ratio = (int(random_float(0.7, 1.5) * original_width),
                                int(random_float(0.7, 1.5) * original_width))
                newWidth, newHeight = resize_ratio

        return recover, resize_ratio

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
