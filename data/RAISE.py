import os

import imageio
import rawpy
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random


def center_crop(patch_size, input_raw, target_rgb):
    # raw输入是1通道
    raw_channel_1 = (len(input_raw.shape) == 2)
    if raw_channel_1:
        h, w = input_raw.shape
    else:
        h, w, _ = input_raw.shape
    x1 = int(round((w - patch_size) / 2.))
    y1 = int(round((h - patch_size) / 2.))
    x1 = x1 - x1 % 2
    y1 = y1 - y1 % 2
    if raw_channel_1:
        patch_input_raw = input_raw[y1:y1 + patch_size, x1:x1 + patch_size]
    else:
        patch_input_raw = input_raw[y1:y1 + patch_size, x1:x1 + patch_size, :]
    if raw_channel_1:
        patch_target_rgb = target_rgb[y1:y1 + patch_size, x1:x1 + patch_size, :]
    else:
        patch_target_rgb = target_rgb[y1 * 2: y1 * 2 + patch_size * 2, x1 * 2: x1 * 2 + patch_size * 2, :]
    return patch_input_raw, patch_target_rgb


def random_crop(patch_size, input_raw, target_rgb):
    # raw输入是1通道h
    raw_channel_1 = (len(input_raw.shape) == 2)
    if raw_channel_1:
        h, w = input_raw.shape
    else:
        h, w, _ = input_raw.shape
    rnd_h = random.randint(0, max(0, h - patch_size))
    rnd_w = random.randint(0, max(0, w - patch_size))
    rnd_h = rnd_h - rnd_h % 2
    rnd_w = rnd_w - rnd_w % 2

    if raw_channel_1:
        patch_input_raw = input_raw[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size]
    else:
        patch_input_raw = input_raw[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]

    if raw_channel_1:
        # 是流模型说明输入输出维度一致 都是 H W 3
        patch_target_rgb = target_rgb[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
    else:
        # 不是流模型 输入大小为 H/2 W/2 3 输出大小为H W 3
        patch_target_rgb = target_rgb[rnd_h * 2:rnd_h * 2 + patch_size * 2, rnd_w * 2:rnd_w * 2 + patch_size * 2, :]

    return patch_input_raw, patch_target_rgb


class Raise(Dataset):
    def __init__(self, data_root, stage='test', patch_size=512):
        self.data_root = data_root
        self.stage = stage
        data_list = self.load()
        self.data_list = data_list
        self.len_data = len(data_list)
        self.patch_size = patch_size

    def load(self):
        file_txt = os.path.join(self.data_root, f'{self.stage}.txt')
        with open(file_txt, "r") as f:
            data_list = [i.strip() for i in f.readlines()]
        # for i in range(len(rgb_files)):
        #     rgb_path = rgb_files[i]
        #     npz_path = npz_files[i]
        #     assert rgb_path.split('/')[-1][:-4] in npz_path

        # split dataset with ratio 85:15 as train set and test set
        # import random
        # random.shuffle(npz_files)
        # split_len = int(len(npz_files) * 0.85)
        # train_list = npz_files[:split_len]
        # test_list = npz_files[split_len:]
        # with open(train_file, "w") as f_train:
        #     for train_item in train_list:
        #         f_train.write(train_item.split('/')[-1][:-4] + '\n')
        # with open(test_file, "w") as f_test:
        #     for test_item in test_list:
        #         f_test.write(test_item.split('/')[-1][:-4] + '\n')

        return data_list

    def get_bayer_pattern(self, flip_val, bayer):
        if flip_val == 5:
            bayer_pattern = (bayer + 1) % 4
        elif flip_val == 3:
            bayer_pattern = (bayer + 2) % 4
        elif flip_val == 6:
            bayer_pattern = (bayer + 3) % 4
        else:
            bayer_pattern = bayer
        return bayer_pattern

    def norm_raw(self, raw, black_level, white_level):
        assert black_level.max() == 0 and black_level.min() == 0
        img = raw / white_level
        return img


    def __getitem__(self, index):
        file_name = self.data_list[index]

        raw_npz_path = os.path.join(self.data_root, 'NPZ', file_name+'.npz')
        rgb_path = os.path.join(self.data_root, 'RGB_NPZ', file_name+'.npz')

        raw_file = np.load(raw_npz_path)
        # rgb = imageio.imread(rgb_path)
        # rgb = cv2.imread(rgb_path)
        rgb = np.load(rgb_path)['rgb']
        # assert rgb.shape[0] == raw_file['raw_img'].shape[0]
        # save rgb image as npz to improve the running speed
        raw_img = raw_file['raw_img']

        cwb = raw_file['cwb']
        cwb = cwb[:3]
        cwb = cwb / cwb.max()
        bayer = raw_file['bayer'].item()
        black_level = raw_file['black_level']
        white_level = raw_file['white_level'].item()
        flip_val = raw_file['flip_val'].item()

        bayer_pattern = self.get_bayer_pattern(flip_val, bayer)

        if self.stage == 'test':
            input_raw_img, target_rgb_img = center_crop(self.patch_size, raw_img, rgb)
        elif self.stage == 'train':
            input_raw_img, target_rgb_img = random_crop(self.patch_size, raw_img, rgb)
        else:
            input_raw_img = raw_img
            target_rgb_img = rgb

        input_raw_img = self.norm_raw(input_raw_img, black_level, white_level)
        target_rgb_img = target_rgb_img / 255
        input_raw_img = np.expand_dims(input_raw_img, axis=2)

        input_raw_img = torch.Tensor(input_raw_img).permute(2, 0, 1)
        target_rgb_img = torch.Tensor(target_rgb_img).permute(2, 0, 1)

        sample = {
            'input_raw': input_raw_img,
            'target_rgb': target_rgb_img,
            'file_name': file_name,
            'camera_whitebalance': cwb,
            'bayer_pattern': bayer_pattern,
            'camera_name': 'NIKON'
        }
        return sample


    def __len__(self):
        return self.len_data


if __name__ == '__main__':
    # with open('/groupshare/raise/test.txt', 'r') as f:
    #     data = [i.strip() for i in f.readlines()]
    # with open('/groupshare/raise_crop/crop_test.txt', 'w') as f:
    #     for d in data:
    #         for i in range(10):
    #             f.write((d+'_'+str(i)+'\n'))
    # exit(0)
    print('test here')
    print('wanna to test RAISE dataset')
    data_root = '/groupshare/raise_crop'
    stage = 'crop_test' # crop_train crop_test
    cur_dataset = Raise(data_root, stage=stage)
    for i in range(len(cur_dataset)):
        item = cur_dataset[i]