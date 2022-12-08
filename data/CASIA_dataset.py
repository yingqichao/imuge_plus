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

class CASIA_dataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt, dataset_opt, is_train=True, dataset="CASIA1"):
        super(CASIA_dataset, self).__init__()
        self.is_train = is_train
        self.opt = opt
        self.dataset_opt = dataset_opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.GT_size = self.dataset_opt['GT_size']
        self.dataset_name = dataset

        self.GT_folder = {
            'CASIA1': [
                '/groupshare/CASIA1/CASIA 1.0 dataset/Original Tp/Tp/Sp'
            ],
            'CASIA2': [
                '/groupshare/CASIA2/Tp'
            ],
            'Defacto': [
                '/groupshare/Defacto/splicing_1_img/img',
                '/groupshare/Defacto/inpainting_img/img',
                '/groupshare/Defacto/copymove_img/img',
            ],
        }
        self.mask_folder = {
            'CASIA1': [
                '/groupshare/CASIA1/CASIA 1.0 groundtruth/Sp'
            ],
            'CASIA2': [
                '/groupshare/CASIA2/CASIA 2 Groundtruth'
            ],
            'Defacto': [
                '/groupshare/Defacto/splicing_1_annotations/probe_mask',
                '/groupshare/Defacto/inpainting_annotations/probe_mask',
                '/groupshare/Defacto/copymove_annotations/probe_mask',
            ],
        }

        self.paths_GT, self.paths_mask = [], []
        GT_items, mask_items = self.GT_folder[self.dataset_name], self.mask_folder[self.dataset_name]

        attack_list = {0}
        for idx in range(len(GT_items)):
            if idx in attack_list:
                GT_path, _ = util.get_image_paths(GT_items[idx])
                mask_path, _ = util.get_image_paths(mask_items[idx])
                # GT_path = sorted(GT_path)
                # mask_path = sorted(mask_path)

                dataset_len = len(GT_path)
                dataset_mask_len = len(mask_path)
                print(f"len image {dataset_len}")
                print(f"len mask {dataset_mask_len}")
                num_train_val_split = int(dataset_len*0.85)
                self.paths_GT += (GT_path[:num_train_val_split] if self.is_train else GT_path[num_train_val_split:])
                self.paths_mask += (mask_path[:num_train_val_split] if self.is_train else mask_path[num_train_val_split:])

        # self.dataset_len = len(self.paths_GT)
        # self.train_val_split = int(self.dataset_len*0.85)
        # if self.is_train:
        #     ### dataset split 85:15
        #     self.paths_GT = self.paths_GT[:self.train_val_split]
        #     self.paths_mask = self.paths_mask[:self.train_val_split]
        # else:
        #     attack_idx = 0
        #
        #     self.paths_GT = self.paths_GT[self.train_val_split:]
        #     self.paths_mask = self.paths_mask[self.train_val_split:]

            # self.paths_GT, self.paths_mask = [], []
            #
            # GT_items_this_attack, _ = util.get_image_paths(self.GT_folder[self.dataset_name][attack_idx])
            # GT_items_this_attack = set(GT_items_this_attack)
            # # mask_items_this_attack, _ = util.get_image_paths(self.mask_folder[self.dataset_name][attack_idx])
            # # mask_items_this_attack = set(mask_items_this_attack)
            #
            # ### split three attacks.
            # for idx in range(len(backup_paths_GT)):
            #     if backup_paths_GT[idx] in GT_items_this_attack:
            #         self.paths_GT += backup_paths_GT[idx]
            #         self.paths_mask += backup_paths_mask[idx]


        self.transform_just_resize = A.Compose(
            [
                A.Resize(always_apply=True, height=self.GT_size, width=self.GT_size)
            ]
        )

        assert self.paths_GT, 'Error: GT path is empty.'



    def __getitem__(self, index):

        # scale = self.dataset_opt['scale']

        # get GT image
        GT_path = self.paths_GT[index]

        # img_GT = util.read_img(GT_path)
        img_GT = cv2.imread(GT_path, cv2.IMREAD_COLOR)
        img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]
        img_GT = self.transform_just_resize(image=copy.deepcopy(img_GT))["image"]
        img_GT = img_GT.astype(np.float32) / 255.
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        # some images have 4 channels
        if img_GT.shape[2] > 3:
            img_GT = img_GT[:, :, :3]
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()

        mask_path = self.paths_mask[index]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        # mask = util.channel_convert(mask.shape[2], self.dataset_opt['color'], [mask])[0]
        mask = self.transform_just_resize(image=copy.deepcopy(mask))["image"]
        mask = mask.astype(np.float32) / 255.
        # if img_GT.ndim == 2:
        #     mask = np.expand_dims(mask, axis=2)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

        # print(f"{self.is_train} GT_path: {GT_path}")
        # print(f"{self.is_train} mask_path: {mask_path}")

        return (img_GT, mask)

    def __len__(self):
        return len(self.paths_GT)

    # def to_tensor(self, img):
    #     img = Image.fromarray(img)
    #     img_t = F.to_tensor(img).float()
    #     return img_t
