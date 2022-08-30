import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

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
    def __init__(self, resize_ratio_range=(0.5,1.5), interpolation_method='bilinear'):
        super(Resize, self).__init__()
        self.name = "Resize"

        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method


    def forward(self, noised_image, resize_ratio=None):

        self.name = "Resize"
        original_width, original_height = noised_image.shape[2], noised_image.shape[3]
        if resize_ratio is None:
            resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)

        newWidth, newHeight = int(resize_ratio*original_width), int(resize_ratio*original_height)
        # resize_ratio = 0.5
        # noised_image = noised_and_cover[0]
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

        return recover
