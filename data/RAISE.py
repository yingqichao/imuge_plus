import os

import imageio
import rawpy
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
import pickle


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
        self.data_list, self.metadata_list = self.load()
        self.len_data = len(self.data_list)
        self.patch_size = patch_size
        np.seterr(divide='ignore', invalid='ignore')

    def load(self):
        file_txt = os.path.join(self.data_root, f'{self.stage}.txt')
        with open(file_txt, "r") as f:
            data_list = [i.strip() for i in f.readlines()]

        if self.stage == 'crop_test':
            random.shuffle(data_list)
            len_data = len(data_list)
            data_list = data_list[:int(len_data*0.1)]
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
        file_pickle = os.path.join(self.data_root, 'metadata.pickle')
        with open(file_pickle, 'rb') as f:
            metadata_list = pickle.load(f)
        return data_list, metadata_list

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


class FiveKTest(Dataset):
    def __init__(self, data_root, camera='Canon', patch_size=512, stage='test'):
        self.data_root = data_root
        self.camera = camera
        self.stage = stage
        self.data_list, self.camera_list, self.metadata_list = self.load()
        self.len_data = len(self.data_list)
        self.patch_size = patch_size


    def load(self):
        file_txt = os.path.join(self.data_root, f'{self.camera}_{self.stage}.txt')
        with open(file_txt, "r") as f:
            content = f.readlines()
            data_list = [i.strip().split(' ')[0] for i in content]
            camera_list = [i.strip().split(' ')[-1] for i in content]

        file_pickle = os.path.join(self.data_root, f'{self.camera}_metadata.pickle')
        with open(file_pickle, 'rb') as f:
            metadata_list = pickle.load(f)

        return data_list, camera_list, metadata_list

        # file_pickle = os.path.join(self.data_root, 'metadata.pickle')
        # with open(file_pickle, 'rb') as f:
        #     metadata_list = pickle.load(f)
        # return data_list, metadata_list

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

    def norm_raw(self, img, black_level, white_level):
        assert len(black_level) == 4
        # print(black_level)
        if len(set(black_level)) > 1:
            # todo: 需要加上全局判断
            norm_black_level = sum(black_level) / len(black_level)
        else:
            # 黑电平的值一致
            norm_black_level = black_level[0]
        img = img - norm_black_level
        img[img < 0] = 0
        img = img / (white_level - norm_black_level)
        return img
        # assert black_level.max() == 0 and black_level.min() == 0
        # img = raw / white_level
        # return img


    def __getitem__(self, index):
        file_name = self.data_list[index]

        raw_npz_path = os.path.join(self.data_root, self.camera, 'RAW', file_name+'.npz')
        rgb_path = os.path.join(self.data_root, self.camera,  'RGB', file_name+'.png')

        raw_file = np.load(raw_npz_path)
        # rgb = imageio.imread(rgb_path)
        # rgb = cv2.imread(rgb_path)
        rgb = imageio.imread(rgb_path)
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


        input_raw_img = self.norm_raw(raw_img, black_level, white_level)
        target_rgb_img = rgb / 255
        input_raw_img = np.expand_dims(input_raw_img, axis=2)

        input_raw_img = torch.Tensor(input_raw_img).permute(2, 0, 1)
        target_rgb_img = torch.Tensor(target_rgb_img).permute(2, 0, 1)

        sample = {
            'input_raw': input_raw_img,
            'target_rgb': target_rgb_img,
            'file_name': file_name,
            'camera_whitebalance': cwb,
            'bayer_pattern': bayer_pattern,
            'camera_name': self.camera_list[index]
        }
        return sample


    def __len__(self):
        return self.len_data


"""
这个函数用来为数据集生成对应的列表文件， 主要针对的是FiveK数据集
"""
def generate_dataset_file(key= 'Canon', stage='test'):
    # with open('/groupshare/raise/test.txt', 'r') as f:
    #     data = [i.strip() for i in f.readlines()]
    # with open('/groupshare/raise_crop/crop_test.txt', 'w') as f:
    #     for d in data:
    #         for i in range(10):
    #             f.write((d + '_' + str(i) + '\n'))

    data_root = '/ssd/FiveK_Dataset'
    with open('./camera.txt', 'r') as f:
        cameras = [i.strip() for i in f.readlines()]
    use_camera_list = []
    for camera in cameras:
        if key in camera:
            use_camera_list.append(camera)
    dng_files = []
    camera_files = []
    for camera_name in use_camera_list:
        test_file_path = os.path.join(data_root, f"{camera_name}_{stage}.txt")
        with open(test_file_path, 'r') as fin:
            cur_dng_files = [i.strip() for i in fin.readlines()]
        result_dng_files = []
        for ll in range(4):
            result_dng_files += [i + f'_{ll}' for i in cur_dng_files]
        dng_files = dng_files + result_dng_files
        camera_files = camera_files + [camera_name] * len(result_dng_files)
    assert len(camera_files) == len(dng_files)
    with open(f'/ssd/FiveK_{stage}/{key}_{stage}.txt', 'w') as fout:
        for dd in range(len(dng_files)):
            fout.write(dng_files[dd] + ' ' + camera_files[dd]+ '\n')



if __name__ == '__main__':
    # generate_dataset_file('Canon', stage='train')
    # exit(0)
    print('wanna to test RAISE dataset')
    data_root = '/groupshare/raise_crop'
    stage = 'crop_train'  # crop_train crop_test
    train_set = Raise(data_root, stage=stage)
    exit(0)
    a = FiveKTest('/ssd/FiveK_test', 'Canon', stage='test')
    for i in range(len(a)):
        item = a[i]
    exit(0)
    print('test here')
    print('wanna to test RAISE dataset')
    data_root = '/groupshare/raise_crop'
    stage = 'crop_test' # crop_train crop_test
    cur_dataset = Raise(data_root, stage=stage)
    for i in range(len(cur_dataset)):
        item = cur_dataset[i]