import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
import rawpy
import glob
import os
import numpy as np
import random
import imageio
import cv2
# from imageio import imread
from tqdm import tqdm
import time


# DNG：原本的raw文件
# RAW: 去马赛克之后的raw文件
# RGB：ISP管道输出的RGB图像
# stage: train 或 test
class FiveKDataset_total(Dataset):
    """
    FiveKDataset
    dataset_root：数据集的路径,
    camera_name：相机名,
    stage：train or test,
    patch_size：cropped rgb image size,
    data_mode='RAW',
    file_nums：训练时，展示验证集图像结果的数目,
    rgb_scale：False or True 代表rgb图像norm的方式，设置为True时，rgb图像将会rescale到[-1, 1],
    npz_uint16：False or True 是否使用uint16格式存储的raw图像
    """

    def __init__(self, dataset_roots: list, camera_names: list, stage, patch_size=256, data_mode='RAW',
                 file_nums=200, npz_uint16=True):
        ####################################################################################################
        # todo: Settings
        # todo:
        ####################################################################################################
        self.data_mode = data_mode
        self.npz_uint16 = npz_uint16
        self.stage = stage
        self.patch_size = patch_size
        ####################################################################################################
        # todo: Adding support of reading multiple camera settings
        # todo:
        ####################################################################################################
        self.metadata_list = {}
        self.bayer_list = {}
        self.raw_files, self.rgb_files, self.camera_name_per_image = [], [], []
        for idx in range(len(dataset_roots)):
            dataset_root = dataset_roots[idx]
            camera_name = camera_names[idx]
            # self.dataset_root = dataset_root
            # self.camera_name = camera_name

            dataset_file = os.path.join(dataset_root, camera_name + '_' + stage + '.txt')
            new_raw_files, new_rgb_files = self.load(dataset_root=dataset_root, file_name=dataset_file,
                                                     camera_name=camera_name)
            if stage != 'train':
                # if len(new_raw_files) > 100:
                #     print(camera_name)
                new_raw_files = new_raw_files[:file_nums]
                new_rgb_files = new_rgb_files[:file_nums]
            # else:
            #     new_raw_files = new_raw_files[file_nums:]
            #     new_rgb_files = new_rgb_files[file_nums:]
            assert len(new_raw_files) == len(new_rgb_files)

            self.raw_files = self.raw_files + new_raw_files
            self.rgb_files = self.rgb_files + new_rgb_files
            self.camera_name_per_image = self.camera_name_per_image + ([camera_name] * len(new_raw_files))

            new_metadata_dict, bayer_metadata_dict = self.load_metadata(dataset_root=dataset_root, dataset_file=dataset_file,
                                                   camera_name=camera_name)
            self.metadata_list.update(new_metadata_dict)
            self.bayer_list.update(bayer_metadata_dict)

    # 这个函数用来通过文件载入数据集
    def load(self, *, dataset_root, file_name, camera_name):
        input_raws = []
        target_rgbs = []

        with open(file_name, "r") as f:
            valid_camera_list = [line.strip() for line in f.readlines()]

        for i, name in enumerate(valid_camera_list):
            full_name = os.path.join(dataset_root, camera_name)
            if self.data_mode == 'RAW':
                raw_folder_name = 'RAW_UINT16' if self.npz_uint16 else 'RAW'
                input_raws.append(os.path.join(full_name, raw_folder_name, name + '.npz'))
            else:
                input_raws.append(os.path.join(full_name, 'DNG', name + '.dng'))
            target_rgbs.append(os.path.join(full_name, 'RGB_PNG', name + '.png'))

        return input_raws, target_rgbs

    def load_metadata(self, *, dataset_root, dataset_file, camera_name):
        import pickle
        # 载入metadata
        with open(os.path.join(dataset_root, camera_name, 'metadata.pickle'), 'rb') as fin:
            data = pickle.load(fin)
        # 载入bayer pattern
        with open(os.path.join(dataset_root, 'bayer.pickle'), 'rb') as fin:
            bayer_data = pickle.load(fin)
        with open(dataset_file, "r") as f:
            valid_camera_list = [line.strip() for line in f.readlines()]
        return dict([(key, data[key]) for key in valid_camera_list]), \
               dict([(key, bayer_data[key]) for key in valid_camera_list])

    def pack_raw(self, raw):
        # 两个相机都是RGGB
        H, W = raw.shape[0], raw.shape[1]
        raw = np.expand_dims(raw, axis=2)
        R = raw[0:H:2, 0:W:2, :]
        Gr = raw[0:H:2, 1:W:2, :]
        Gb = raw[1:H:2, 0:W:2, :]
        B = raw[1:H:2, 1:W:2, :]
        G_avg = (Gr + Gb) / 2
        out = np.concatenate((R, G_avg, B), axis=2)
        return out

    def visualize_raw(self, raw, bayer_pattern, wb):
        # 不同bayer pattern对应的坐标
        coords = [
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 0], [0, 0], [1, 1], [0, 1]],
            [[1, 1], [0, 1], [1, 0], [0, 0]],
            [[0, 1], [0, 0], [1, 1], [1, 0]]
        ]
        cur_coords = coords[bayer_pattern]
        wb = wb[:3]
        wb = wb / wb.max()
        wb_inv = [1 / i for i in wb]
        # 两个相机都是RGGB
        # im = np.expand_dims(raw, axis=2)
        H, W = raw.shape[0], raw.shape[1]
        v_im = np.zeros([H, W, 1], dtype=np.float64)
        v_im[0:H:2, 0:W:2, 0] = raw[cur_coords[0][0]:H:2, cur_coords[0][1]:W:2] * wb[0] * wb_inv[0]
        v_im[0:H:2, 1:W:2, 0] = raw[cur_coords[1][0]:H:2, cur_coords[1][1]:W:2] * wb[1] * wb_inv[1]
        v_im[1:H:2, 0:W:2, 0] = raw[cur_coords[2][0]:H:2, cur_coords[2][1]:W:2] * wb[1] * wb_inv[1]
        v_im[1:H:2, 1:W:2, 0] = raw[cur_coords[3][0]:H:2, cur_coords[3][1]:W:2] * wb[2] * wb_inv[2]
        # v_im = np.zeros([H, W, 3], dtype=np.uint16)
        # v_im[0:H:2, 0:W:2, 0] = raw[cur_coords[0][0]:H:2, cur_coords[0][1]:W:2]
        # v_im[0:H:2, 1:W:2, 1] = raw[cur_coords[1][0]:H:2, cur_coords[1][1]:W:2]
        # v_im[1:H:2, 0:W:2, 1] = raw[cur_coords[2][0]:H:2, cur_coords[2][1]:W:2]
        # v_im[1:H:2, 1:W:2, 2] = raw[cur_coords[3][0]:H:2, cur_coords[3][1]:W:2]
        return v_im

    def get_bayer_pattern(self, flip_val, bayer):
        """
        Args: flip_val
        Returns:
            0 means RGGB
            1 means GBRG
            2 means BGGR
            3 means GRBG
        """
        bayer_patterns = {
            'RGGB': 0,
            'GBRG': 1,
            'BGGR': 2,
            'GRBG': 3
        }
        origin_bayer = bayer_patterns[bayer]
        if flip_val == 5:
            bayer_pattern = (origin_bayer + 1) % 4
        elif flip_val == 3:
            bayer_pattern = (origin_bayer + 2) % 4
        elif flip_val == 6:
            bayer_pattern = (origin_bayer + 3) % 4
        else:
            bayer_pattern = origin_bayer
        return bayer_pattern

    def __getitem__(self, index):
        raw_path = self.raw_files[index]
        rgb_path = self.rgb_files[index]
        camera_name = self.camera_name_per_image[index]
        file_name = raw_path.split("/")[-1].split(".")[0]
        meta = self.metadata_list[file_name]
        metadata = meta['metadata']
        bayer = self.bayer_list[file_name]

        target_rgb_img = imageio.imread(rgb_path)
        assert self.data_mode == 'RAW'
        raw = np.load(raw_path)
        input_raw_img = raw['raw']
        if self.stage != 'train':
            input_raw_img, target_rgb_img = center_crop(self.patch_size, input_raw_img, target_rgb_img,
                                                        not self.npz_uint16)
        else:
            input_raw_img, target_rgb_img = random_crop(self.patch_size, input_raw_img, target_rgb_img,
                                                        not self.npz_uint16)

        white_level = metadata['white_level'][0]
        black_level = metadata['black_level']


        target_rgb_img = target_rgb_img / 255
        input_raw_img = self.norm_raw(input_raw_img, white_level, black_level)
        # visualize_raw = input_raw_img.copy()
        input_raw_img = np.expand_dims(input_raw_img, axis=2)

        input_raw_img = torch.Tensor(input_raw_img).permute(2, 0, 1)
        target_rgb_img = torch.Tensor(target_rgb_img).permute(2, 0, 1)


        flip_val = meta['flip_val']
        wb = raw['wb']
        wb = wb[:3]
        wb = wb / wb.max()
        bayer_pattern = self.get_bayer_pattern(flip_val, bayer)
        # visualize_raw = self.visualize_raw(visualize_raw, bayer_pattern, wb)
        # visualize_raw = torch.Tensor(visualize_raw).permute(2, 0, 1)
        # todo:取出要用的metadata
        sample = {'input_raw': input_raw_img,
                  'target_rgb': target_rgb_img,
                  'file_name': file_name,
                  'color_matrix': meta['color_matrix'],
                  # example:
                  # array([[1.8083024, -0.9273899, 0.1190875, 0.],
                  #        [-0.01726297, 1.3148121, -0.29754913, 0.],
                  #        [0.05812025, -0.34984493, 1.2917247, 0.]], dtype=float32), '
                  'rgb_xyz_matrix': meta['rgb_xyz_matrix'],
                  # example:
                  # tensor([[[0.6347, -0.0479, -0.0972],
                  #          [-0.8297, 1.5954, 0.2480],
                  #          [-0.1968, 0.2131, 0.7649],
                  #          [0.0000, 0.0000, 0.0000]]])
                  # 'camera_whitebalance': meta['camera_whitebalance'],
                  'camera_whitebalance': wb,
                  # example (batch size=2):
                  # [tensor([2.1602, 1.5434], dtype=torch.float64), tensor([1., 1.], dtype=torch.float64), tensor([1.3457, 2.0000],
                  # dtype=torch.float64), tensor([0., 0.], dtype=torch.fl
                  ###### 0,1,2,3
                  'bayer_pattern': bayer_pattern,
                  'camera_name': camera_name,
                  # 'visualize_raw': visualize_raw
                  }
        return sample




    def __len__(self):
        return len(self.raw_files)

    def norm_raw(self, img, white_level, black_level):
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


class FiveKDataset_skip(Dataset):
    """
    FiveKDataset
    dataset_root：数据集的路径,
    camera_name：相机名,
    stage：train or test,
    patch_size：rgb image size // 2 (transfer raw 2*2 pattern into 1 pixel),
    data_mode='RAW',
    uncond_p：条件输入为空（全1图像）的概率,
    file_nums：训练时，展示验证集图像结果的数目,
    rgb_scale：False or True 代表rgb图像norm的方式，设置为True时，rgb图像将会rescale到[-1, 1],
    npz_uint16：False or True 是否使用uint16格式存储的raw图像
    """

    def __init__(self, dataset_roots: list, camera_names: list, stage, patch_size=256, data_mode='RAW', uncond_p=0.2,
                 file_nums=100, rgb_scale=False, npz_uint16=True, use_metadata=True):
        ####################################################################################################
        # todo: Settings
        # todo:
        ####################################################################################################
        self.data_mode = data_mode
        self.uncond_p = uncond_p
        self.rgb_scale = rgb_scale
        self.npz_uint16 = npz_uint16
        self.use_metadata = use_metadata
        self.stage = stage
        self.patch_size = patch_size
        self.gamma = True
        ####################################################################################################
        # todo: Adding support of reading multiple camera settings
        # todo:
        ####################################################################################################
        self.metadata_list = {}
        self.raw_files, self.rgb_files, self.camera_name_per_image = [], [], []
        for idx in range(len(dataset_roots)):
            dataset_root = dataset_roots[idx]
            camera_name = camera_names[idx]
            # self.dataset_root = dataset_root
            # self.camera_name = camera_name

            dataset_file = os.path.join(dataset_root, camera_name + '_' + stage + '.txt')
            new_raw_files, new_rgb_files = self.load(dataset_root=dataset_root, file_name=dataset_file,
                                                     camera_name=camera_name)
            if stage != 'train':
                new_raw_files = new_raw_files[:file_nums]
                new_rgb_files = new_rgb_files[:file_nums]
            # else:
            #     new_raw_files = new_raw_files[file_nums:]
            #     new_rgb_files = new_rgb_files[file_nums:]
            assert len(new_raw_files) == len(new_rgb_files)

            self.raw_files = self.raw_files + new_raw_files
            self.rgb_files = self.rgb_files + new_rgb_files
            self.camera_name_per_image = self.camera_name_per_image + ([camera_name] * len(new_raw_files))

            if self.use_metadata:
                new_metadata_dict = self.load_metadata(dataset_root=dataset_root, dataset_file=dataset_file,
                                                       camera_name=camera_name)
                self.metadata_list.update(new_metadata_dict)

    # 这个函数用来通过文件载入数据集
    def load(self, *, dataset_root, file_name, camera_name):
        input_raws = []
        target_rgbs = []

        with open(file_name, "r") as f:
            valid_camera_list = [line.strip() for line in f.readlines()]

        for i, name in enumerate(valid_camera_list):
            full_name = os.path.join(dataset_root, camera_name)
            if self.data_mode == 'RAW':
                raw_folder_name = 'RAW_UINT16' if self.npz_uint16 else 'RAW'
                input_raws.append(os.path.join(full_name, raw_folder_name, name + '.npz'))
            else:
                input_raws.append(os.path.join(full_name, 'DNG', name + '.dng'))
            target_rgbs.append(os.path.join(full_name, 'RGB_PNG', name + '.png'))

        return input_raws, target_rgbs

    def load_metadata(self, *, dataset_root, dataset_file, camera_name):
        import pickle
        with open(os.path.join(dataset_root, camera_name, 'metadata.pickle'), 'rb') as fin:
            data = pickle.load(fin)
        with open(dataset_file, "r") as f:
            valid_camera_list = [line.strip() for line in f.readlines()]
        return dict([(key, data[key]) for key in valid_camera_list])

    def pack_raw(self, raw):
        # 两个相机都是RGGB
        H, W = raw.shape[0], raw.shape[1]
        raw = np.expand_dims(raw, axis=2)
        R = raw[0:H:2, 0:W:2, :]
        Gr = raw[0:H:2, 1:W:2, :]
        Gb = raw[1:H:2, 0:W:2, :]
        B = raw[1:H:2, 1:W:2, :]
        G_avg = (Gr + Gb) / 2
        out = np.concatenate((R, G_avg, B), axis=2)
        return out

    def visualize_raw(self, raw, bayer_pattern, wb):
        # 不同bayer pattern对应的坐标
        coords = [
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 0], [0, 0], [1, 1], [0, 1]],
            [[1, 1], [0, 1], [1, 0], [0, 0]],
            [[0, 1], [0, 0], [1, 1], [1, 0]]
        ]
        cur_coords = coords[bayer_pattern]
        wb = wb[:3]
        wb = wb / wb.max()
        wb_inv = [1 / i for i in wb]
        # 两个相机都是RGGB
        # im = np.expand_dims(raw, axis=2)
        H, W = raw.shape[0], raw.shape[1]
        v_im = np.zeros([H, W, 1], dtype=np.float64)
        v_im[0:H:2, 0:W:2, 0] = raw[cur_coords[0][0]:H:2, cur_coords[0][1]:W:2] * wb[0] * wb_inv[0]
        v_im[0:H:2, 1:W:2, 0] = raw[cur_coords[1][0]:H:2, cur_coords[1][1]:W:2] * wb[1] * wb_inv[1]
        v_im[1:H:2, 0:W:2, 0] = raw[cur_coords[2][0]:H:2, cur_coords[2][1]:W:2] * wb[1] * wb_inv[1]
        v_im[1:H:2, 1:W:2, 0] = raw[cur_coords[3][0]:H:2, cur_coords[3][1]:W:2] * wb[2] * wb_inv[2]
        # v_im = np.zeros([H, W, 3], dtype=np.uint16)
        # v_im[0:H:2, 0:W:2, 0] = raw[cur_coords[0][0]:H:2, cur_coords[0][1]:W:2]
        # v_im[0:H:2, 1:W:2, 1] = raw[cur_coords[1][0]:H:2, cur_coords[1][1]:W:2]
        # v_im[1:H:2, 0:W:2, 1] = raw[cur_coords[2][0]:H:2, cur_coords[2][1]:W:2]
        # v_im[1:H:2, 1:W:2, 2] = raw[cur_coords[3][0]:H:2, cur_coords[3][1]:W:2]
        return v_im

    def get_bayer_pattern(self, flip_val):
        """
        Args: flip_val
        Returns:
            0 means RGGB
            1 means GBRG
            2 means BGGR
            3 means GRBG
        """
        bayer_pattern = 0
        if flip_val == 5:
            bayer_pattern = 1
        elif flip_val == 3:
            bayer_pattern = 2
        elif flip_val == 6:
            bayer_pattern = 3
        return bayer_pattern

    def __getitem__(self, index):
        raw_path = self.raw_files[index]
        rgb_path = self.rgb_files[index]
        camera_name = self.camera_name_per_image[index]
        target_rgb_img = imageio.imread(rgb_path)
        assert self.data_mode == 'RAW'
        raw = np.load(raw_path)
        input_raw_img = raw['raw']
        if self.stage != 'train':
            input_raw_img, target_rgb_img = center_crop(self.patch_size, input_raw_img, target_rgb_img,
                                                        not self.npz_uint16)
        else:
            input_raw_img, target_rgb_img = random_crop(self.patch_size, input_raw_img, target_rgb_img, not self.npz_uint16)
        norm_value = 4095 if camera_name == 'Canon_EOS_5D' else 16383

        target_rgb_img = self.norm_img(target_rgb_img, 255, self.rgb_scale)
        input_raw_img = self.norm_img(input_raw_img, max_value=norm_value)
        # visualize_raw = input_raw_img.copy()
        input_raw_img = np.expand_dims(input_raw_img, axis=2)

        input_raw_img = torch.Tensor(input_raw_img).permute(2, 0, 1)
        target_rgb_img = torch.Tensor(target_rgb_img).permute(2, 0, 1)

        if random.random() < self.uncond_p:
            # null label
            input_raw_img = torch.ones_like(input_raw_img)
        file_name = raw_path.split("/")[-1].split(".")[0]
        if self.use_metadata:
            meta = self.metadata_list[file_name]
            # print(meta)
            flip_val = meta['flip_val']
            wb = raw['wb']
            wb = wb[:3]
            wb = wb / wb.max()
            bayer_pattern = self.get_bayer_pattern(flip_val)
            # visualize_raw = self.visualize_raw(visualize_raw, bayer_pattern, wb)
            # visualize_raw = torch.Tensor(visualize_raw).permute(2, 0, 1)
            # todo:取出要用的metadata
            sample = {'input_raw': input_raw_img,
                      'target_rgb': target_rgb_img,
                      'file_name': file_name,
                      'color_matrix': meta['color_matrix'],
                      # example:
                      # array([[1.8083024, -0.9273899, 0.1190875, 0.],
                      #        [-0.01726297, 1.3148121, -0.29754913, 0.],
                      #        [0.05812025, -0.34984493, 1.2917247, 0.]], dtype=float32), '
                      'rgb_xyz_matrix': meta['rgb_xyz_matrix'],
                      # example:
                      # tensor([[[0.6347, -0.0479, -0.0972],
                      #          [-0.8297, 1.5954, 0.2480],
                      #          [-0.1968, 0.2131, 0.7649],
                      #          [0.0000, 0.0000, 0.0000]]])
                      # 'camera_whitebalance': meta['camera_whitebalance'],
                      'camera_whitebalance': wb,
                      # example (batch size=2):
                      # [tensor([2.1602, 1.5434], dtype=torch.float64), tensor([1., 1.], dtype=torch.float64), tensor([1.3457, 2.0000],
                      # dtype=torch.float64), tensor([0., 0.], dtype=torch.fl
                      ###### 0,1,2,3
                      'bayer_pattern': bayer_pattern,
                      'camera_name': camera_name,
                      # 'visualize_raw': visualize_raw
                      }

        else:
            sample = {'input_raw': input_raw_img,
                      'target_rgb': target_rgb_img,
                      'file_name': file_name, }
        return sample

    def __len__(self):
        return len(self.raw_files)

    def norm_img(self, img, max_value, scale_minus1=False):
        if scale_minus1:
            half_value = max_value / 2
            img = img / half_value - 1
            # scaled to [-1, 1]
        else:
            img = img / float(max_value)
        return img


def unflip(raw_img, flip):
    if flip == 3:
        raw_img = np.rot90(raw_img, k=2)
    elif flip == 5:
        raw_img = np.rot90(raw_img, k=3)
    elif flip == 6:
        raw_img = np.rot90(raw_img, k=1)
    else:
        pass
    return raw_img


# np.rot90(img, k):将矩阵逆时针旋转90*k度以后返回，k取负数时表示顺时针旋转
def flip(raw_img, flip):
    if flip == 3:
        raw_img = np.rot90(raw_img, k=2)
    elif flip == 5:
        raw_img = np.rot90(raw_img, k=1)
    elif flip == 6:
        raw_img = np.rot90(raw_img, k=3)
    else:
        pass
    return raw_img


# camera_crop_coords = {
#     'Canon_EOS_5D': {
#         'x_coord': [1168, 1168 + 512, 1168 + 512 * 2, 1168 + 512 * 3],
#         'y_coord': [432, 432 + 512, 432 + 512 * 2, 432 + 512 * 3]
#     },
#     'NIKON_D700': {
#         'x_coord': [1120, 1120 + 512, 1120 + 512 * 2, 1120 + 512 * 3],
#         'y_coord': [400, 400 + 512, 400 + 512 * 2, 400 + 512 * 3]
#     }
# }
#
#
# # 输入ndarray
# def crop_raw_img(raw_image, flip_val, camera_name):
#     raw_image = flip(raw_image, flip_val)
#     shape = raw_image.shape
#     h, w = shape
#     x_coord = camera_crop_coords[camera_name]['x_coord']
#     y_coord = camera_crop_coords[camera_name]['y_coord']
#     coords = []
#     results = []
#     for i in range(len(x_coord)):
#         for j in range(len(y_coord)):
#             coords.append([x_coord[i], y_coord[j]])
#     for k in range(len(coords)):
#         x, y = coords[k]
#         if h > w:
#             de_raw = raw_image[x:x + 512, y:y + 512]
#         else:
#             de_raw = raw_image[y:y + 512, x:x + 512]
#         de_raw = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(de_raw, 'RGGB')
#         results.append(de_raw)
#     return results
#
#
# def crop_rgb_img(rgb_image, camera_name):
#     shape = rgb_image.shape
#     h, w = shape[:2]
#     x_coord = camera_crop_coords[camera_name]['x_coord']
#     y_coord = camera_crop_coords[camera_name]['y_coord']
#     coords = []
#     results = []
#     for i in range(len(x_coord)):
#         for j in range(len(y_coord)):
#             coords.append([x_coord[i], y_coord[j]])
#     for k in range(len(coords)):
#         x, y = coords[k]
#         if h > w:
#             # box = (x, y, x+512, y+512)
#             de_raw = rgb_image[x:x + 512, y:y + 512]
#         else:
#             # box = (y, x, y+512, x+512)
#             de_raw = rgb_image[y:y + 512, x:x + 512]
#         results.append(de_raw)
#     return results
#
#
# def merge_images(out_image, gt_image):
#     return np.hstack([out_image, gt_image])
#
#
# # 此处之前踩过大坑，一般来说图像读取的维度都是H W C 但是Image.open读取的结果是W H C 服气
# def crop_image():
#     dataset_root = '/home/groupshare/invISP'
#     camera_name = 'Canon_EOS_5D'
#     raw_path = glob.glob(dataset_root + '/' + camera_name + '/DNG' + '/*.dng')
#     rgb_root = dataset_root + '/' + camera_name + '/RGB/'
#     deraw_root = dataset_root + '/' + camera_name + '/RAW/'
#     # dng_paths = glob.glob(dataset_root+'/'+camera_name+'/DNG'+'/*.dng')
#     output_root = '/home/groupshare/invISP_crop/'
#     output_path = output_root + '/' + camera_name + '/'
#     # files = os.listdir(output_path+'RGB')
#     # list = []
#     for file_path in tqdm(raw_path):
#         file_name = file_path.split('/')[-1].split('.')[0]
#         rgb_path = rgb_root + file_name + '.jpg'
#         # deraw_path = deraw_root + file_name + '.npz'
#         raw = rawpy.imread(file_path)
#         flip_val = raw.sizes.flip
#         cwb = raw.camera_whitebalance
#         raw_img = raw.raw_image_visible
#         if camera_name == 'Canon_EOS_5D':
#             raw_img = np.maximum(raw_img - 127.0, 0)
#         cropped_raw_images = crop_raw_img(raw_img, flip_val, camera_name)
#         # raw_shape = raw_img.shape
#         # rgb_image = imageio.imread(rgb_path)
#         rgb_image = cv2.imread(rgb_path)
#         cropped_images = crop_rgb_img(rgb_image, camera_name)
#         for l in range(len(cropped_images)):
#             cropped_rgb_image = cropped_images[l]
#             cropped_raw_image = cropped_raw_images[l]
#             # m_images = merge_images(cropped_raw_image, cropped_rgb_image)
#             cropped_raw_file_path = output_path + '/RAW/' + file_name + '_' + str(l) + '.npz'
#             np.savez(cropped_raw_file_path, raw=cropped_raw_image, cwb=cwb)
#             cropped_rgb_file_path = output_path + '/RGB/' + file_name + '_' + str(l) + '.jpg'
#             cv2.imwrite(cropped_rgb_file_path, cropped_rgb_image)


def crop_a_image(raw_image, target_rgb, patch_size, nums, flow=False, select_samples=2):
    """
    image: B H W C
    patch_size: cropped image size
    nums: the sqrt of number of cropped image
    """
    new_h = patch_size * nums
    idx_list = [i for i in range(nums * nums)]
    random.shuffle(idx_list)
    idx_list = idx_list[:select_samples]
    cropped_raw, cropped_rgb = center_crop(new_h, raw_image, target_rgb, flow=False)
    input_raw = []
    target_rgb = []
    for idx in idx_list:
        i = int(idx // nums)
        j = int(idx % nums)
        start_i = i * patch_size
        start_j = j * patch_size
        input_raw.append(cropped_raw[start_i:start_i + patch_size, start_j:start_j + patch_size, :])
        if flow:
            target_rgb.append(
                cropped_rgb[start_i:start_i + patch_size, start_j:start_j + patch_size, :])
        else:
            target_rgb.append(
                cropped_rgb[start_i * 2:start_i * 2 + patch_size * 2, start_j * 2:start_j * 2 + patch_size * 2, :])
    return np.concatenate(input_raw, axis=2), np.concatenate(target_rgb, axis=2)


# 这里用来分割数据集
# 85:15的比例划分得到训练集:测试集
def split_dataset(dataset_root, camera_name):
    dataset_path = os.path.join(dataset_root, camera_name)
    rgb_files = sorted(os.listdir(os.path.join(dataset_path, "RGB")))
    val_list = []
    train_file = os.path.join(dataset_root, camera_name + '_train.txt')
    test_file = os.path.join(dataset_root, camera_name + '_test.txt')
    for rgb_file in rgb_files:
        val_list.append(rgb_file.split('.')[0])
    random.shuffle(val_list)
    split_len = int(len(val_list) * 0.85)
    train_list = val_list[:split_len]
    test_list = val_list[split_len:]
    with open(train_file, "w") as f_train:
        for train_item in train_list:
            f_train.write(train_item + '\n')
    with open(test_file, "w") as f_test:
        for test_item in test_list:
            f_test.write(test_item + '\n')


def center_crop(patch_size, input_raw, target_rgb, flow=True):
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
    if flow or raw_channel_1:
        patch_target_rgb = target_rgb[y1:y1 + patch_size, x1:x1 + patch_size, :]
    else:
        patch_target_rgb = target_rgb[y1 * 2: y1 * 2 + patch_size * 2, x1 * 2: x1 * 2 + patch_size * 2, :]
    return patch_input_raw, patch_target_rgb


def random_crop(patch_size, input_raw, target_rgb, flow=True):
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

    if flow or raw_channel_1:
        # 是流模型说明输入输出维度一致 都是 H W 3
        patch_target_rgb = target_rgb[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
    else:
        # 不是流模型 输入大小为 H/2 W/2 3 输出大小为H W 3
        patch_target_rgb = target_rgb[rnd_h * 2:rnd_h * 2 + patch_size * 2, rnd_w * 2:rnd_w * 2 + patch_size * 2, :]

    return patch_input_raw, patch_target_rgb


def random_rotate(input_raw, target_rgb, possiblity=0.5):
    if random.random() < possiblity:
        idx = random.randint(0, 4)
        input_raw = np.rot90(input_raw, k=idx)
        target_rgb = np.rot90(target_rgb, k=idx)

    return input_raw, target_rgb


def random_flip(input_raw, target_rgb, possiblity=0.5):
    if random.random() < possiblity:
        idx = random.randint(0, 1)
        input_raw = np.flip(input_raw, axis=idx).copy()
        target_rgb = np.flip(target_rgb, axis=idx).copy()

    return input_raw, target_rgb


def aug(input_raw, target_rgb):
    # 把随机crop换成center_crop
    # input_raw, target_rgb = center_crop(patch_size, input_raw, target_rgb, not npz_uint16)

    # input_raw, target_rgb = random_crop(patch_size, input_raw, target_rgb, not npz_uint16)
    input_raw, target_rgb = random_rotate(input_raw, target_rgb)
    input_raw, target_rgb = random_flip(input_raw, target_rgb)
    return input_raw, target_rgb


def aug_joint(patch_size, input_raw, target_rgb):
    input_raw, target_rgb = random_crop(patch_size, input_raw, target_rgb, False)
    input_raw, target_rgb = random_rotate(input_raw, target_rgb)
    input_raw, target_rgb = random_flip(input_raw, target_rgb)
    return input_raw, target_rgb


def load_data(
        *,
        data_dir,
        camera_name,
        stage,
        batch_size,
        patch_size,
        data_mode,
        uncond_p,
        deterministic,
        num_workers,
        file_nums=10,
        rgb_scale=False,
        npz_uint16=True
):
    # deterministic 代表是否打乱数据集
    if not data_dir:
        raise ValueError('unspecified data directory')
    dataset = FiveKDataset_skip(data_dir, camera_name, stage, patch_size, data_mode, uncond_p, file_nums, rgb_scale,
                                npz_uint16)
    # print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=deterministic, num_workers=num_workers,
                            drop_last=True,
                            pin_memory=False)
    return dataloader
    # while True:
    #     yield from dataloader


# 这个函数用来生成用uint16格式存储的npz文件
def data_process_npz(dataset_root, camera_name):
    camera_raw_path = os.path.join(dataset_root, camera_name, 'DNG')
    dng_paths = sorted(glob.glob(camera_raw_path + '/*.dng'))
    for path in tqdm(dng_paths):
        filename = os.path.basename(path)
        output_path = os.path.join(dataset_root, camera_name, 'RAW_UINT16')
        output_npz_path = os.path.join(output_path, filename.replace('.dng', '.npz'))
        if os.path.exists(output_npz_path):
            continue
        raw = rawpy.imread(path)
        raw_image = raw.raw_image_visible
        cwb = raw.camera_whitebalance
        flip_val = raw.sizes.flip
        if camera_name == 'Canon_EOS_5D':
            raw_image = np.maximum(raw_image - 127.0, 0)
        raw_image = flip(raw_image, flip_val)
        raw_image = raw_image.astype(np.uint16)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        np.savez(output_npz_path, raw=raw_image, wb=cwb)


class test_crop():
    def __init__(self):
        pass

    def center_crop(self, input_raw, target_rgb, patch_size):
        """
        input_raw: [H, W, 1]
        target_rgb: [H, W, 3]
        return: [patch_size, patch_size, 1], [patch_size, patch_size, 3]
        """
        h, w = input_raw.shape
        new_h, new_w = h // 2, w // 2
        patch_size = patch_size // 2
        x1 = int(round((new_w - patch_size) / 2.))
        y1 = int(round((new_h - patch_size) / 2.))
        patch_input_raw = input_raw[y1 * 2:y1 * 2 + patch_size * 2, x1 * 2:x1 * 2 + patch_size * 2]

        patch_target_rgb = target_rgb[y1 * 2:y1 * 2 + patch_size * 2, x1 * 2: x1 * 2 + patch_size * 2, :]
        return self.get_sub_images(patch_input_raw, patch_target_rgb)

    def get_sub_images(self, input_raw, target_rgb):
        nums = 2
        patch_size = 512
        raws = []
        rgbs = []
        idx_list = [i for i in range(nums * nums)]
        for idx in idx_list:
            i = int(idx // nums)
            j = int(idx % nums)
            start_i = i * patch_size
            start_j = j * patch_size
            raws.append(input_raw[start_i:start_i + patch_size, start_j:start_j + patch_size])
            rgbs.append(target_rgb[start_i:start_i + patch_size, start_j:start_j + patch_size, :])

        return raws, rgbs


# 这个函数用来根据npz文件裁剪得到中央的16张256*256子图
def data_process_crop_npz(dataset_root, camera_name, new_root):
    camera_raw_path = os.path.join(dataset_root, camera_name, 'RAW_UINT16')
    npz_paths = sorted(glob.glob(camera_raw_path + '/*.npz'))
    output_raw_path = os.path.join(new_root, camera_name, 'RAW_UINT16')
    if not os.path.exists(output_raw_path):
        os.makedirs(output_raw_path, exist_ok=True)
    output_rgb_path = os.path.join(new_root, camera_name, 'RGB')
    if not os.path.exists(output_rgb_path):
        os.makedirs(output_rgb_path, exist_ok=True)
    crop_util = test_crop()
    for path in tqdm(npz_paths):
        file_name = os.path.basename(path)
        rgb_name = file_name.replace('.npz', '.jpg')
        rgb_path = os.path.join(dataset_root, camera_name, 'RGB', rgb_name)

        raw = np.load(path)
        raw_img = raw['raw']
        wb = raw['wb']
        rgb = imageio.imread(rgb_path)
        raw_subimages, rgb_subimages = crop_util.center_crop(raw_img, rgb, 1024)
        for j in range(len(raw_subimages)):
            subraw = raw_subimages[j]
            subrgb = rgb_subimages[j]
            subraw_filename = file_name.replace('.npz', f'_{j}.npz')
            subrgb_filename = file_name.replace('.npz', f'_{j}.jpg')
            np.savez(os.path.join(output_raw_path, subraw_filename), raw=subraw, wb=wb)
            imageio.imwrite(os.path.join(output_rgb_path, subrgb_filename), subrgb)


# 这个函数用来跳着采样，来降低数据集里图像的大小
def data_process_skip(dataset_root, camera_name, new_root):
    def pack_raw(raw):
        # 两个相机都是RGGB
        H, W = raw.shape[0], raw.shape[1]
        raw = np.expand_dims(raw, axis=2)
        R = raw[0:H:2, 0:W:2, :]
        Gr = raw[0:H:2, 1:W:2, :]
        Gb = raw[1:H:2, 0:W:2, :]
        B = raw[1:H:2, 1:W:2, :]
        out = np.concatenate((R, Gr, Gb, B), axis=2)
        return out

    camera_raw_path = os.path.join(dataset_root, camera_name, 'RAW_UINT16')
    npz_paths = sorted(glob.glob(camera_raw_path + '/*.npz'))
    output_raw_path = os.path.join(new_root, camera_name, 'RAW_UINT16')
    if not os.path.exists(output_raw_path):
        os.makedirs(output_raw_path, exist_ok=True)
    output_rgb_path = os.path.join(new_root, camera_name, 'RGB_PNG')
    if not os.path.exists(output_rgb_path):
        os.makedirs(output_rgb_path, exist_ok=True)
    # crop_util = test_crop()
    for path in tqdm(npz_paths):
        file_name = os.path.basename(path)
        rgb_name = file_name.replace('.npz', '.png')
        rgb_path = os.path.join(dataset_root, camera_name, 'RGB_PNG', rgb_name)

        raw = np.load(path)
        raw_img = raw['raw']
        wb = raw['wb']
        rgb = imageio.imread(rgb_path)
        rgb_h, rgb_w, _ = rgb.shape
        raw_h, raw_w = raw_img.shape
        assert raw_h == rgb_h and raw_w == rgb_w
        rgb_h = rgb_h - (rgb_h % 8)
        rgb_w = rgb_w - (rgb_w % 8)
        rgb_resize = rgb[0:rgb_h:4, 0:rgb_w:4, :]
        raw_img = raw_img[0:rgb_h, 0:rgb_w]
        imageio.imwrite(os.path.join(output_rgb_path, rgb_name), rgb_resize)
        p_raw = pack_raw(raw_img)
        raw_resize = p_raw[0:rgb_h:4, 0:rgb_w:4, :]
        new_H, new_W = raw_resize.shape[0], raw_resize.shape[1]
        v_im = np.zeros([new_H * 2, new_W * 2], dtype=np.uint16)
        v_im[0:new_H * 2:2, 0:new_W * 2:2] = raw_resize[0:new_H, 0:new_W, 0]
        v_im[0:new_H * 2:2, 1:new_W * 2:2] = raw_resize[0:new_H, 0:new_W, 1]
        v_im[1:new_H * 2:2, 0:new_W * 2:2] = raw_resize[0:new_H, 0:new_W, 2]
        v_im[1:new_H * 2:2, 1:new_W * 2:2] = raw_resize[0:new_H, 0:new_W, 3]
        np.savez(os.path.join(output_raw_path, file_name), raw=v_im, wb=wb)


# 这个函数用来更新metadata.pickle 目前是加入了flip_val
def set_metadata_pickle(dataset_root, camera_name):
    import pickle
    with open(os.path.join(dataset_root, camera_name, 'metadata.pickle'), 'rb') as fin:
        data = pickle.load(fin)
    default_root = '/ssd/invISP/'
    for file_name in tqdm(data.keys()):
        dng_path = os.path.join(default_root, camera_name, 'DNG', file_name + '.dng')
        raw = rawpy.imread(dng_path)
        data[file_name]['metadata']['black_level'] = raw.black_level_per_channel
    with open(os.path.join(dataset_root, camera_name, 'metadata.pickle'), 'wb') as out:
        pickle.dump(data, out)


# def test_skip_downsample(dataset_root, camera_name):
#     stage = 'test'
#     dataset = FiveKDataset(dataset_root, camera_name, stage, 32, 'RAW', 0, 10, True, True)
#     rgb_path = os.path.join(dataset_root, camera_name, 'RGB')
#     for i in range(len(dataset)):
#         item = dataset[i]
#         file_name = item['file_name']
#         rgb_file_path = os.path.join(rgb_path, file_name + '.jpg')
#         rgb = imageio.imread(rgb_file_path)
#         H, W, _ = rgb.shape
#         print(H, W)
#         Resize = rgb[1:H:4, 1:W:4, :]
#         imageio.imwrite(f'./test_{i}.jpg', Resize)

# from wand.image import Image as WandImage
# from wand.api import library as wandlibrary
# from scipy.ndimage import zoom as scizoom
# from io import BytesIO
# import wand.color as WandColor
# # Extend wand.image.Image class to include method signature
# class MotionImage(WandImage):
#     def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
#         wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
#
# def clipped_zoom(img, zoom_factor):
#     h = img.shape[0]
#     # ceil crop height(= crop width)
#     # np.ceil 向上取整
#     ch = int(np.ceil(h / float(zoom_factor)))
#
#     top = (h - ch) // 2
#     img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
#     # trim off any extra pixels
#     trim_top = (img.shape[0] - h) // 2
#
#     return img[trim_top:trim_top + h, trim_top:trim_top + h]
#
# def snow(x, severity=1):
#     c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
#          (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
#          (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
#          (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
#          (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
#
#     x = np.array(x, dtype=np.float32) / 255.
#     snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome
#     # np.newaxis 在该位置新增一个维度
#     snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
#     snow_layer[snow_layer < c[3]] = 0
#
#     snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
#     output = BytesIO()
#     snow_layer.save(output, format='PNG')
#     snow_layer = MotionImage(blob=output.getvalue())
#
#     snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
#
#     snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
#                               cv2.IMREAD_UNCHANGED) / 255.
#     snow_layer = snow_layer[..., np.newaxis]
#
#     x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(512, 512, 1) * 1.5 + 0.5)
#     return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
# 这里用来处理数据集 如裁剪等
# 数据集可用 data_mode=RAW表明是对插值后的图像
if __name__ == '__main__':
    from data.pipeline import rawpy_tensor2image
    from data.pipeline import pipeline_tensor2image
    import os
    import pickle

    dataset_root = '/ssd/invISP_skip/'
    camera_name = 'Canon_EOS_5D'
    os.makedirs('./test/', exist_ok=True)
    # dng_files = os.listdir(os.path.join(dataset_root, camera_name, 'DNG'))
    # with open(os.path.join('/ssd/invISP_skip/', camera_name, 'metadata.pickle'), 'rb') as f:
    #     metadatas = pickle.load(f)
    # for dng_file_name in dng_files:
    #     if not dng_file_name.endswith('.dng'):
    #         continue
    #     dng_file_path = os.path.join(dataset_root, camera_name, 'DNG', dng_file_name)
    #     raw_image_visible = rawpy.imread(dng_file_path).raw_image_visible
    #     target_rgb_path = os.path.join(dataset_root, camera_name, 'RGB_PNG', dng_file_name.replace('.dng', '.png'))
    #     target_rgb = imageio.imread(target_rgb_path)
    #     fn = dng_file_name.split('.')[-2]
    #     metadata = metadatas[fn]
    #     flip_val = metadata['flip_val']
    #     raw_image_visible = flip(raw_image_visible, flip_val)
    #     rawf, rgbf = random_crop(512, raw_image_visible, target_rgb)
    #     rawf = unflip(rawf, flip_val)
    #     n_rgb = pipeline_tensor2image(raw_image=rawf, metadata=metadata['metadata'], input_stage='raw')
    #     numpy_rgb = (n_rgb * 255).astype(np.uint8)
    #     numpy_rgb = np.concatenate([numpy_rgb, rgbf], axis=1)
    #     print(dng_file_name)
    #     PIL.Image.fromarray(numpy_rgb).save(os.path.join('./test/', f'{fn}.png'), subsampling=1)
    # exit(0)
    # data_process_npz(dataset_root, camera_name)
    # test_image_downsample(dataset_root, camera_name)
    # exit(0)
    # data_process_npz(dataset_root, camera_name)
    # exit(0)
    train_set = FiveKDataset_skip([dataset_root], [camera_name], stage='train', rgb_scale=False, uncond_p=0.,
                                  patch_size=512, use_metadata=True)
    dataloader = DataLoader(train_set, batch_size=4, shuffle=False, num_workers=4,
                            drop_last=True,
                            pin_memory=False)
    start = time.time()

    for i, value in enumerate(dataloader):
        # print(value)
        file_name = value['file_name']
        print(file_name[0], value['bayer_pattern'][0])
        # metadata = train_set.metadata_list[file_name[0]]

        input_raw = value['input_raw'][0]
        # 测试 my own pipeline
        # metadata = train_set.metadata_list[file_name[0]]
        # flip_val = metadata['flip_val']
        # metadata = metadata['metadata']
        # # 在metadata中加入要用的flip_val和camera_name
        # metadata['flip_val'] = flip_val
        # metadata['camera_name'] = camera_name
        #
        # # print(metadata)
        # input_raw = input_raw.permute(1, 2, 0).squeeze(2)
        # numpy_rgb = pipeline_tensor2image(raw_image=input_raw, metadata=metadata, input_stage='raw')
        # numpy_rgb = (numpy_rgb * 255).astype(np.uint8)

        # 测试rawpy
        # ###########################################################################
        # # todo：用法 直接传入raw图，template可以传入rawpy对象，也可以是file_name
        # ###########################################################################
        numpy_rgb = rawpy_tensor2image(raw_image=input_raw, template=file_name[0], camera_name=camera_name[0],
                                       patch_size=512)
        target_rgb = value['target_rgb'][0].permute(1, 2, 0).cpu().numpy() * 255
        target_rgb = target_rgb.astype(np.uint8)
        # print(target_rgb[0])
        # print('--split--')
        # print(numpy_rgb[0])
        numpy_rgb = np.concatenate([numpy_rgb, target_rgb], axis=1)
        PIL.Image.fromarray(numpy_rgb).save(os.path.join('./test/', f'testrawpy_{i}.png'), subsampling=1)
        # exit(0)
        # numpy_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(numpy_rgb, (2, 0, 1)))).float()
        # print(f"numpy_rgb:{numpy_rgb.shape}")
        # if i>10:
        #     break

    end_time = time.time()
    print((end_time - start))
