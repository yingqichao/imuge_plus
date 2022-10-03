import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os
# from turbojpeg import TurboJPEG
from PIL import Image
# from jpeg2dct.numpy import load, loads
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import albumentations as A
import copy

class LQDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt, dataset_opt, is_train=True):
        super(LQDataset, self).__init__()
        self.is_train = is_train
        self.opt = opt
        self.dataset_opt = dataset_opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.GT_size = self.dataset_opt['GT_size']

        self.paths_GT, _ = util.get_image_paths(dataset_opt['dataroot_GT'])
        self.paths_MASKS, _ = util.get_image_paths(dataset_opt['dataroot_LQ'])

        assert self.paths_GT, 'Error: GT path is empty.'

        self.random_scale_list = [1]

        # self.jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')


    def __getitem__(self, index):

        # scale = self.dataset_opt['scale']

        # get GT image
        GT_path = self.paths_GT[index]
        GT_MASK_path = self.paths_MASKS[index]

        # img_GT = util.read_img(GT_path)
        img_GT = cv2.imread(GT_path, cv2.IMREAD_COLOR)
        # img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]
        mask_GT = cv2.imread(GT_MASK_path, cv2.IMREAD_GRAYSCALE)

        # img_GT = self.transform(image=copy.deepcopy(img_GT))["image"]

        img_GT = img_GT.astype(np.float32) / 255.
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        # some images have 4 channels
        if img_GT.shape[2] > 3:
            img_GT = img_GT[:, :, :3]
        mask_GT = mask_GT.astype(np.float32) / 255.


        ###### directly resize instead of crop
        # img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
        #                     interpolation=cv2.INTER_LINEAR)

        orig_height, orig_width, _ = img_GT.shape
        H, W, _ = img_GT.shape

        mask_GT = torch.from_numpy(np.ascontiguousarray(mask_GT)).float()


        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]


        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()

        return (img_GT, mask_GT)

    def __len__(self):
        return len(self.paths_GT)

