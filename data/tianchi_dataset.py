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
from jpeg2dct.numpy import load, loads
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import torchvision.transforms.functional as F

class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self):
        dataroot_GT = '/home/groupshare/forgery_round1_train_20220217/train/img'
        super(LQGTDataset, self).__init__()
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.mask_filepath= '/home/groupshare/processed'
        self.paths_GT, self.sizes_GT = util.get_image_paths(dataroot_GT)
        print(len(self.paths_GT))
        assert self.paths_GT, 'Error: GT path is empty.'



    def __getitem__(self, index):

        # get GT image
        imgs_GT = []
        GT_path = self.paths_GT[index]
        filepath, tempfilepath = os.path.split(GT_path)
        img_GT = util.read_img(GT_path, cv2.IMREAD_COLOR)
        img_GT = cv2.resize(np.copy(img_GT), (512, 512),
                            interpolation=cv2.INTER_LINEAR)
        imgs_GT.append(img_GT)

        jpeg_path = os.path.join(self.mask_filepath,tempfilepath)
        jpeg_path = jpeg_path[:-4]+".png"
        # print(jpeg_path)
        img_GT = util.read_img(jpeg_path, cv2.IMREAD_GRAYSCALE)
        img_GT = cv2.resize(np.copy(img_GT), (512, 512),
                            interpolation=cv2.INTER_LINEAR)
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        imgs_GT.append(img_GT)

        out_list = []
        for idx in range(len(imgs_GT)):
            img_GT = imgs_GT[idx]

            # BGR to RGB, HWC to CHW, numpy to tensor
            if img_GT.shape[2] == 3:
                img_GT = img_GT[:, :, [2, 1, 0]]

            img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
            out_list.append(img_GT)

        return out_list, torch.tensor([0]).long()

    ########### how to read
    # imgs, label = batch
    # img, mask = imgs[0].cuda(), imgs[1].cuda()
    ### construct the dataset
    # from data.LQ_dataset import LQGTDataset as D
    # train_set = D()

    def __len__(self):
        return len(self.paths_GT)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
