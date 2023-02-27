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

class TianchiDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt, dataset_opt, is_train=True, with_au=False, with_mask=True,
                 split=True, filter=False):
        super(TianchiDataset, self).__init__()
        self.filter = filter
        self.min_rate, self.max_rate = 0.01, 0.4
        self.ban_list = set()
        self.is_train = "train" if is_train else "test"
        self.opt = opt
        self.dataset_opt = dataset_opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.GT_size = self.dataset_opt['GT_size']
        self.dataset_name = ["tianchi"]
        self.with_au = with_au
        self.with_mask = with_mask
        self.split = split

        print(f"Using {self.dataset_name} with_au {with_au} with_mask {with_mask} is_train {is_train}")

        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )

        self.GT_folder = {
            'train':{
                'tianchi': [
                    '/groupshare/tianchi2023/train/train/tampered/imgs',
                    # '/groupshare/tianchi2023/train/automatic/train/tampered/imgs',
                ],
            },
            'test': {
                'tianchi': [
                    '/groupshare/tianchi2023/test/imgs',
                    # '/groupshare/tianchi2023/train/automatic/train/tampered/imgs',
                ],
            },
        }
        self.mask_folder = {
            'tianchi': [
                '/groupshare/tianchi2023/train/train/tampered/masks',
                # '/groupshare/tianchi2023/train/automatic/train/tampered/masks',
            ],
        }


        self.paths_GT, self.codebook = {}, [] # list is less dependable if we need to load mask
        for i, this_dataset in enumerate(self.dataset_name):
            GT_items = self.GT_folder[self.is_train][this_dataset]
            for idx in range(len(GT_items)):

                GT_path, new_codes = util.get_filename_from_images(GT_items[idx],sep1='.',sep2=None,prefix=str(idx))

                # GT_path = sorted(GT_path)
                # mask_path = sorted(mask_path)
                dataset_len = len(GT_path)
                print(f"len image {dataset_len}")

                self.paths_GT.update(GT_path)
                new_codes = new_codes

                self.codebook += new_codes

        if self.with_mask:
            self.paths_mask = {}
            for i, this_dataset in enumerate(self.dataset_name):
                mask_items = self.mask_folder[this_dataset]
                for idx in range(len(mask_items)):

                    mask_path, _ = util.get_filename_from_images(mask_items[idx],sep1='.',sep2=None,prefix=str(idx))

                    dataset_mask_len = len(mask_path)
                    print(f"len mask {dataset_mask_len}")


                    self.paths_mask.update(mask_path)


            assert len(self.paths_GT)==len(self.paths_mask), f'Mask和Image的总数不匹配！{len(self.paths_GT)} {len(self.paths_mask)}请检查'

        if self.with_au:
            self.au_folder = {
                'tianchi': [
                    '/groupshare/tianchi2023/train/train/untampered'
                ],
            }

            ### au folder
            self.au_GT = []
            for i, this_dataset in enumerate(self.dataset_name):
                au_items = self.au_folder[this_dataset]
                for idx in range(len(au_items)):
                    au_path, _ = util.get_image_paths(au_items[idx])
                    au_dataset_len = len(au_path)
                    print(f"Au len image {au_dataset_len}")

                    num_train_val_split = int(au_dataset_len * 0.85)
                    self.au_GT += (au_path[:num_train_val_split] if self.is_train=='train' else au_path[num_train_val_split:])

        assert self.paths_GT, 'Error: GT path is empty.'



    def __getitem__(self, index):

        return_list = []
        # scale = self.dataset_opt['scale']

        # get GT image
        valid=False
        while not valid:
            if not self.filter:
                valid = True
            filename = self.codebook[index]
            GT_path = self.paths_GT[filename]
            if GT_path in self.ban_list:
                index = np.random.randint(0,len(self.paths_mask))
                continue

            if self.with_mask:
                mask_path = self.paths_mask[filename]
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.uint8) * 255
                # mask = util.channel_convert(mask.shape[2], self.dataset_opt['color'], [mask])[0]
                ## todo: augment is disabled
                # mask = self.transform_just_resize(image=copy.deepcopy(mask))["image"]
                H, W = mask.shape
                rescale = max(H,W)/1024
                if rescale>1:
                    mask = cv2.resize(np.copy(mask), (int(W/rescale), int(H/rescale)),
                                        interpolation=cv2.INTER_LINEAR)
                mask = mask.astype(np.float32) / 255.
                # if img_GT.ndim == 2:
                #     mask = np.expand_dims(mask, axis=2)
                mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
                if 'nist16' in GT_path:
                    # nist16的mask是反过来的，未篡改为白色
                    mask = 1.0 - mask
                rate = torch.mean(mask)
                if self.filter and (rate<self.min_rate or rate>self.max_rate):
                    self.ban_list.add(GT_path)
                    index = np.random.randint(0, len(self.paths_mask))
                    continue

            valid = True
            # img_GT = util.read_img(GT_path)
            img_GT = cv2.imread(GT_path, cv2.IMREAD_COLOR)
            # img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]
            ## todo: augment is disabled
            H, W, _ = img_GT.shape
            rescale = max(H, W) / 1024
            if rescale > 1:
                img_GT = cv2.resize(np.copy(img_GT), (int(W / rescale), int(H / rescale)),
                                  interpolation=cv2.INTER_LINEAR)
            img_GT = self.transform(image=copy.deepcopy(img_GT))["image"]
            img_GT = img_GT.astype(np.float32) / 255.
            if img_GT.ndim == 2:
                img_GT = np.expand_dims(img_GT, axis=2)
            # some images have 4 channels
            if img_GT.shape[2] > 3:
                img_GT = img_GT[:, :, :3]
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_GT = img_GT[:, :, [2, 1, 0]]

            img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
            return_list.append(img_GT)
            return_list.append(GT_path[GT_path.rfind('/')+1:])

            if self.with_mask:
                return_list.append(mask)
                return_list.append(mask_path[mask_path.rfind('/')+1:])

        # if  self.with_au:
        #     index = np.random.randint(0,len(self.au_GT))
        #     GT_path = self.au_GT[index]
        #
        #     img_au_GT = cv2.imread(GT_path, cv2.IMREAD_COLOR)
        #     # img_au_GT = util.channel_convert(img_au_GT.shape[2], self.dataset_opt['color'], [img_au_GT])[0]
        #     img_au_GT = self.transform_just_resize(image=copy.deepcopy(img_au_GT))["image"]
        #     img_au_GT = img_au_GT.astype(np.float32) / 255.
        #     if img_au_GT.ndim == 2:
        #         img_au_GT = np.expand_dims(img_au_GT, axis=2)
        #     # some images have 4 channels
        #     if img_au_GT.shape[2] > 3:
        #         img_au_GT = img_au_GT[:, :, :3]
        #     # BGR to RGB, HWC to CHW, numpy to tensor
        #     img_au_GT = img_au_GT[:, :, [2, 1, 0]]
        #
        #     img_au_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_au_GT, (2, 0, 1)))).float()
        #
        #     return_list.append(img_au_GT)

        return tuple(return_list)
        # if self.with_au and self.with_mask:
        #     return img_GT, mask, img_au_GT
        # elif self.with_au and not self.with_mask:
        #     return img_GT, img_au_GT
        # elif not self.with_au and self.with_mask:
        #     return img_GT, mask
        # else:
        #     return img_GT

    def __len__(self):
        return len(self.paths_GT)

    # def to_tensor(self, img):
    #     img = Image.fromarray(img)
    #     img_t = F.to_tensor(img).float()
    #     return img_t
if __name__ == '__main__':
    import torchvision
    def print_this_image(image, filename):
        '''
            the input should be sized [C,H,W], not [N,C,H,W]
        '''
        camera_ready = image.unsqueeze(0)
        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                     filename, nrow=1,
                                     padding=0, normalize=False)
    dataset_opt = {'color':'RGB', 'GT_size':256}
    # 单元测试
    train_set = TianchiDataset(None, dataset_opt,)
    test_image, test_mask= train_set[0]
    print_this_image(test_image, './image.png')
    print_this_image(test_mask, './mask.png')