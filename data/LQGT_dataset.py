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

class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt, dataset_opt, is_train=True):
        super(LQGTDataset, self).__init__()
        self.is_train = is_train
        self.opt = opt
        self.dataset_opt = dataset_opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.GT_size = self.dataset_opt['GT_size']
        # self.jpeg_filepath=['/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_10',
        #                     '/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_30',
        #                     '/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_50',
        #                     '/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_70',
        #                     '/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_90']
        self.paths_GT, self.sizes_GT = util.get_image_paths(dataset_opt['dataroot_GT'])
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(p=0.5),
                A.OneOf(
                    [
                        A.CLAHE(always_apply=False, p=0.25),
                        A.RandomBrightnessContrast(always_apply=False, p=0.25),
                        A.Equalize(always_apply=False, p=0.25),
                        A.RGBShift(always_apply=False, p=0.25),
                    ]
                ),
                A.OneOf(
                    [
                        A.ImageCompression(always_apply=False, quality_lower=60, quality_upper=100, p=0.25),
                        A.MedianBlur(always_apply=False, p=0.25),
                        A.GaussianBlur(always_apply=False, p=0.25),
                        A.MotionBlur(always_apply=False, p=0.25),
                        # A.GaussNoise(always_apply=False, p=0.2),
                        # A.ISONoise(always_apply=False, p=0.2)
                    ]
                ),
                A.Resize(always_apply=True, height=self.GT_size, width=self.GT_size)
            ]
        )
        self.transform_just_resize = A.Compose(
            [
                A.Resize(always_apply=True, height=self.GT_size, width=self.GT_size)
            ]
        )

        assert self.paths_GT, 'Error: GT path is empty.'

        self.random_scale_list = [1]

        # self.jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')


    def __getitem__(self, index):

        # scale = self.dataset_opt['scale']

        # get GT image
        GT_path = self.paths_GT[index]

        # img_GT = util.read_img(GT_path)
        img_GT = cv2.imread(GT_path, cv2.IMREAD_COLOR)
        img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]

        if not self.is_train:
            img_GT = self.transform_just_resize(image=copy.deepcopy(img_GT))["image"]
        else:
            img_GT = self.transform(image=copy.deepcopy(img_GT))["image"]

        img_GT = img_GT.astype(np.float32) / 255.
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        # some images have 4 channels
        if img_GT.shape[2] > 3:
            img_GT = img_GT[:, :, :3]


        # filepath, tempfilepath = os.path.split(GT_path)

        # load_jpeg = False
        # if load_jpeg:
        #     index = np.random.randint(0,5)
        #     # index = 0
        #     if index==0:
        #         QF=0.1
        #     elif index==1:
        #         QF=0.3
        #     elif index == 2:
        #         QF = 0.5
        #     elif index == 3:
        #         QF = 0.7
        #     else: #if index==4:
        #         QF= 0.9
        #     jpeg_path = os.path.join(self.jpeg_filepath[index],tempfilepath)
        #
        #     label = int((QF*10)/2)
        #     # label = torch.zeros(5)
        #     # label[int((QF*10)/2)] = 1
        #     # print(jpeg_path)
        #
        #     ######### if exist jpeg path
        #     img_jpeg_GT = util.read_img(jpeg_path)
        #     ######### else
        #     # img_jpeg_GT = util.read_img(GT_path)


        # img_jpeg_GT = util.channel_convert(img_jpeg_GT.shape[2], self.dataset_opt['color'], [img_jpeg_GT])[0]


        ###### directly resize instead of crop
        # img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
        #                     interpolation=cv2.INTER_LINEAR)

        orig_height, orig_width, _ = img_GT.shape
        H, W, _ = img_GT.shape

        img_gray = rgb2gray(img_GT)
        sigma = 2 #random.randint(1, 4)


        canny_img = canny(img_gray, sigma=sigma, mask=None)
        canny_img = canny_img.astype(np.float)
        canny_img = self.to_tensor(canny_img)



        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            # img_jpeg_GT = img_jpeg_GT[:, :, [2, 1, 0]]
            # img_LQ = img_LQ[:, :, [2, 1, 0]]


        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        # canny_img = torch.from_numpy(np.ascontiguousarray(canny_img)).float()
        # img_jpeg_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_jpeg_GT, (2, 0, 1)))).float()
        # img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        # if LQ_path is None:
        #     LQ_path = GT_path

        return (img_GT, 0, canny_img if canny_img is not None else img_GT.clone())

    def __len__(self):
        return len(self.paths_GT)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
