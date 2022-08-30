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

    def __init__(self, opt, dataset_opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.dataset_opt = dataset_opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.jpeg_filepath=['/home/qcying/jpeg_test2017/10',
                            '/home/qcying/jpeg_test2017/30',
                            '/home/qcying/jpeg_test2017/50',
                            '/home/qcying/jpeg_test2017/70',
                            '/home/qcying/jpeg_test2017/90']
        self.paths_GT, self.sizes_GT = util.get_image_paths(dataset_opt['dataroot_GT'])

        assert self.paths_GT, 'Error: GT path is empty.'

        self.random_scale_list = [1]

        # self.jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')


    def __getitem__(self, index):

        scale = self.dataset_opt['scale']
        GT_size = self.dataset_opt['GT_size']

        # get GT image
        img_jpeg_GT = []
        GT_path = self.paths_GT[index]
        filepath, tempfilepath = os.path.split(GT_path)

        img_jpeg_GT.append(util.read_img(GT_path))

        for idx in range(0,5):
            jpeg_path = os.path.join(self.jpeg_filepath[idx],tempfilepath)
            img_jpeg_GT.append(util.read_img(jpeg_path))

        out_list = []
        for idx in range(len(img_jpeg_GT)):
            img_GT = img_jpeg_GT[idx]
            img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]

            ###### directly resize instead of crop
            if idx==0:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)

            # H, W,_ = img_GT.shape
            # if H<GT_size or W<GT_size:
            #     img_GT = cv2.resize(np.copy(img_GT), (GT_size, W),
            #                                             interpolation=cv2.INTER_LINEAR)
            # if W<GT_size:
            #     img_GT = cv2.resize(np.copy(img_GT), (H, GT_size),
            #                                             interpolation=cv2.INTER_LINEAR)
            H, W, _ = img_GT.shape
            # randomly crop
            # rnd_h = int(random.randint(0, max(0, H - GT_size))/8)*8
            # rnd_w = int(random.randint(0, max(0, W - GT_size))/8)*8
            # img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            orig_height, orig_width, _ = img_GT.shape
            # H, W, _ = img_GT.shape

            # BGR to RGB, HWC to CHW, numpy to tensor
            if img_GT.shape[2] == 3:
                img_GT = img_GT[:, :, [2, 1, 0]]

            img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
            out_list.append(img_GT)

        return out_list, torch.tensor([0,1,2,3,4,5]).long()

    def __len__(self):
        return len(self.paths_GT)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
