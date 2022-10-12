import os

from torch.utils.data import Dataset
import scipy.io as rawio
import h5py
import cv2
from tqdm import tqdm
import numpy as np
import random
import torch
from typing import List
import imageio
import pickle

BAYER_PATTERN = {
    "GP": "BGGR",
    "IP": "RGGB",
    "S6": "GRBG",
    "N6": "BGGR",
    "G4": "BGGR"
}

def get_file_list(dir, file_list, tag=None):
    if os.path.isdir(dir):
        for s in os.listdir(dir):
            new_dir = os.path.join(dir, s)
            # if not os.path.isdir(new_dir):
            #     print(dir, len(os.listdir(dir)))
            #     break
            get_file_list(new_dir, file_list, tag)
    else:
        if tag is None:
            file_list.append(dir)
        elif tag in dir:
            file_list.append(dir)
    return file_list


def get_cropped_area(file_path):
    metadata = rawio.loadmat(file_path)['metadata']
    if 'ActiveArea' in metadata.dtype.names:
        test = metadata['ActiveArea']
        print(file_path, test)
    else:
        if 'extra' in metadata.dtype.names:
            if 'ActiveArea' in metadata['extra'].dtype.names:
                test = metadata['extra']['ActiveArea']
                # print(file_path, test)
            else:
                assert 0 == 1
        elif 'SubIFDs' in metadata.dtype.names:
            test = metadata['SubIFDs'][0][0][0][0]['ActiveArea']
            # print(file_path, test)
        else:
            assert 0 == 1
    return test


def get_phone(file_name):
    for key in BAYER_PATTERN.keys():
        if key in file_name:
            return key
    return ''


# 获取拜耳格式
def read_bayer_pattern(file_name):
    for key in BAYER_PATTERN.keys():
        if key in file_name:
            return BAYER_PATTERN[key]
    return ''

def get_cfa_pattern_mat(metadata, field_lists, cur_phone):
    rgb_str = 'rgb'
    phone = get_phone(str(metadata['Filename'].item()))
    if phone != cur_phone:
        print(phone)
    if 'UnknownTags' in field_lists:
        ut = metadata['UnknownTags']
        cfa_index = ut.item()[1].item()[2][0].tolist()
    else:
        print('here', metadata['Filename'].item())
    cfa_pattern = ''
    for i in cfa_index:
        cfa_pattern += rgb_str[i]
    print(cfa_pattern)
    return phone

# 获取MAT中的图像数据
def read_raw(file_path):
    data = h5py.File(file_path, 'r')
    test = data['x']
    return np.array(test)

def read_metadata(file_path):
    metadata = rawio.loadmat(file_path)['metadata']
    # print(metadata.dtype.names)
    filename = metadata['Filename'][0][0][0]
    h, w = metadata['Height'][0][0][0][0], metadata['Width'][0][0][0][0]
    if 'Orientation' in metadata.dtype.names:
        flip = metadata['Orientation'][0][0][0][0]
    else:
        flip = 0
    bayer_pattern = read_bayer_pattern(filename)
    wb = metadata['AsShotNeutral'].item()[0].tolist()
    black_level = metadata['BlackLevel'].item()[0].tolist()
    white_level = metadata['WhiteLevel'].item()[0].tolist()
    return flip, bayer_pattern, h, w, filename, wb, black_level, white_level

# 把raw的路径换成对应srgb的路径
def replace_raw_path(raw_path, before_str, after_str, suffix):
    tmp = raw_path.replace(before_str, after_str)
    srgb_path = tmp[:-4] + suffix
    return srgb_path


def random_crop(patch_size, input_raw, target_rgb):
    h, w, _ = input_raw.shape
    h = int(h/2)
    w = int(w/2)
    rnd_h = random.randint(0, max(0, h - patch_size))
    rnd_w = random.randint(0, max(0, w - patch_size))
    patch_input_raw = input_raw[rnd_h * 2:rnd_h * 2 + patch_size * 2, rnd_w * 2:rnd_w * 2 + patch_size * 2, :]
    patch_target_rgb = target_rgb[rnd_h * 2:rnd_h * 2 + patch_size * 2, rnd_w * 2:rnd_w * 2 + patch_size * 2, :]

    return patch_input_raw, patch_target_rgb

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


class SIDDSceneInstance():
    def __init__(self, base_dir: str, scene_instance_name: str):
        self.scene_instance_name = scene_instance_name
        values = self.scene_instance_name.split('_')
        self.scene_instance_id = values[0]
        self.scene_id = values[1]
        self.smartphone = values[2]
        self.iso = int(values[3])
        self.shutter_speed = int(values[4])
        self.cct = int(values[5])
        self.luminance = values[6]
        self.bayer = BAYER_PATTERN[self.smartphone]

        self.visible = os.path.isdir(os.path.join(base_dir, 'data', self.scene_instance_id + '_GT_RAW'))

        if self.visible:
            self.gt_raw_dir = self.get_dir(os.path.join(base_dir, 'data'), self.scene_instance_id + '_GT_RAW')
            self.gt_srgb_dir = self.get_dir(os.path.join(base_dir, 'data_srgb'), self.scene_instance_id + '_GT_RAW')
            self.metadata_dir = self.get_dir(os.path.join(base_dir, 'metadata'),
                                             self.scene_instance_id + '_METADATA_RAW')
            gt_raw_num = len(os.listdir(self.gt_raw_dir))
            gt_srgb_num = len(os.listdir(self.gt_srgb_dir))

            assert gt_raw_num == gt_srgb_num
            self.img_num = gt_srgb_num
        else:
            self.img_num = 0

    def get_dir(self, dir_path, folder_name):
        while True:
            new_path = os.path.join(dir_path, folder_name)
            if os.path.isdir(new_path):
                dir_path = new_path
            else:
                break
        return dir_path

    def get_files(self, number):
        srgb_files = os.listdir(self.gt_srgb_dir)
        if number < self.img_num:
            random.shuffle(srgb_files)
            srgb_names = srgb_files[:number]
        else:
            srgb_names = srgb_files
        raw = [os.path.join(self.gt_raw_dir, i.replace('_tone.png', '.MAT')) for i in srgb_names]
        srgb = [os.path.join(self.gt_srgb_dir, i) for i in srgb_names]
        metadata = [os.path.join(self.metadata_dir, i.replace('_tone.png', '.MAT').replace('GT', 'METADATA'))
                    for i in srgb_names]
        return raw, srgb, metadata
        # while(os.path.isdir(1))


    def match(self, visible: bool=None, scene_id: str or List[str]=None, \
            smartphone: str or List[str]=None, iso: int or List[int]=None, \
            cct: int or List[int]=None, luminance: str or List[str]=None) -> bool:
        """
        匹配scene instance特征，返回是否匹配
        :param visible: 是否可见（部分scene instance隐藏用于benchmark）
        :param scene_id: scene id, (001, 002, ..., 010)
        :param smartphone: smartphone code, (GP, IP, S6, N6, G4)
        :param iso: ISO
        :param cct: cct, (3200, 4400, 5500)
        :param luminance: luminance, (L, N, H)
        """
        if self.img_num == 0:
            return False

        if type(scene_id) == str:
            scene_id = [scene_id]

        if type(smartphone) == str:
            smartphone = [smartphone]

        if type(iso) == int:
            iso = [iso]

        if type(cct) == int:
            cct = [cct]

        if type(luminance) == str:
            luminance = [luminance]

        return (visible is None or self.visible == visible) \
            and (scene_id is None or self.scene_id in scene_id) \
            and (smartphone is None or self.smartphone in smartphone) \
            and (iso is None or self.iso in iso) \
            and (cct is None or self.cct in cct) \
            and (luminance is None or self.luminance in luminance)


def get_scene_instance(path: str) -> List[str]:
    """
    读取scene instance文件
    :param path: scene instance file path
    :return:
    """
    with open(path, "r") as f:
        scene_instance_list = f.read().splitlines()
    return scene_instance_list


class SIDD(Dataset):
    def __init__(self, base_dir, file_txt, number=10, patch_size=512):
        self.base_dir = base_dir
        source_file_path = os.path.join(base_dir, file_txt)
        # scene_instance_name_list = get_scene_instance(source_file_path)
        # self.scene_list = [SIDDSceneInstance(self.base_dir, scene_instance_name)
        #                    for scene_instance_name in scene_instance_name_list]
        # self.cur_list = self.filter(number)
        with open(os.path.join(base_dir, file_txt), 'rb') as f:
            data = pickle.load(f)
        self.raw_list = data['raw_list']
        self.srgb_list = data['srgb_list']
        self.meta_list = data['meta_list']
        self.len_data = len(self.raw_list)
        # self.imio = imio('RGB', 'HWC')
        self.patch_size = patch_size



    def filter(self, number):
        """
        用于过滤一些scene instance
        :return:
        """
        cur_list = {'raw':[], 'srgb':[], 'meta':[]}
        for scene in self.scene_list:
            if scene.match(visible=True):
                raw, srgb, meta = scene.get_files(number)
                cur_list['raw'] += raw
                cur_list['srgb'] += srgb
                cur_list['meta'] += meta
        return cur_list

    def visualize_raw(self, raw, pattern='RGGB'):
        # 两个相机都是RGGB
        # im = np.expand_dims(raw, axis=2)
        H, W = raw.shape[0], raw.shape[1]
        m = {'R': 0, 'G': 1, 'B': 2}
        v_im = np.zeros([H, W, 3], dtype=np.float32)
        v_im[0:H:2, 0:W:2, m[pattern[0]]] = raw[0:H:2, 0:W:2]
        v_im[0:H:2, 1:W:2, m[pattern[1]]] = raw[0:H:2, 1:W:2]
        v_im[1:H:2, 0:W:2, m[pattern[2]]] = raw[1:H:2, 0:W:2]
        v_im[1:H:2, 1:W:2, m[pattern[3]]] = raw[1:H:2, 1:W:2]
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
        raw_path = self.raw_list[index]
        srgb_path = self.srgb_list[index]
        metadata = self.meta_list[index]

        flip_value = metadata['flip_val']
        bayer = metadata['bayer']
        wb = metadata['wb']
        black_level = metadata['black_level']
        white_level = metadata['white_level']
        filename = metadata['filename']

        raw = read_raw(raw_path)
        # flip_value, bayer, h, w, filename, wb, black_level, white_level = read_metadata(metadata_path)
        # return flip_value, bayer, h, w, filename, wb, black_level, white_level, raw_path, srgb_path
        raw = raw.transpose()
        raw = fix_orientation(raw, flip_value)
        srgb = imageio.imread(srgb_path)
        raw, srgb = center_crop(self.patch_size, raw, srgb, False)
        raw = np.expand_dims(raw, axis=2)
        srgb = srgb / 255

        input_raw_img = torch.Tensor(raw).permute(2, 0, 1)
        target_rgb_img = torch.Tensor(srgb).permute(2, 0, 1)

        wb_inv = [1/i for i in wb]
        wb_inv = [wb_inv[j] / max(wb_inv) for j in range(len(wb_inv))]
        bayer_pattern = self.get_bayer_pattern(flip_value, bayer)

        return {
            'input_raw': input_raw_img,
            'target_rgb': target_rgb_img,
            'camera_whitebalance': np.array(wb_inv),
            'bayer_pattern': bayer_pattern,
            'file_name': filename,
            'camera_name': 'test_smartphone'
        }
        # bayer = read_bayer_pattern(filename)
        # raw = self.visualize_raw(raw, bayer)
        # srgb = self.imio.read(srgb_path)
        # assert raw.shape == srgb.shape
        # raw, srgb = random_crop(self.patch_size, raw, srgb)
        #
        # target_raw = raw.copy()
        # srgb = srgb / 255
        # raw = torch.Tensor(raw).permute(2, 0, 1)
        # target_raw = torch.Tensor(target_raw).permute(2, 0, 1)
        # srgb = torch.Tensor(srgb).permute(2, 0, 1)

        # return {
        #     'input_raw': raw,
        #     'target_rgb': srgb,
        #     'target_raw': target_raw
        #
        # }

    def __len__(self):
        return self.len_data
# class SIDD_Dataset(Dataset):
#     def __init__(self, dataset_root, patch_size=256):
#         data_root = os.path.join(dataset_root, 'data')
#         metadata_root = os.path.join(dataset_root, 'metadata')
#         self.data_root = data_root
#         self.meta_root = metadata_root
#         self.raw_list = get_file_list(data_root, [], 'RAW')
#         self.srgb_list = [replace_raw_path(i, 'RAW', 'SRGB', '.PNG') for i in self.raw_list]
#         self.meta_list = [replace_raw_path(i, 'GT', 'METADATA', '.MAT') for i in self.raw_list]
#         self.len_data = len(self.raw_list)
#         self.imio = imio('RGB', 'HWC')
#         self.patch_size = patch_size
#
#
#     def visualize_raw(self, raw, pattern='RGGB'):
#         # 两个相机都是RGGB
#         # im = np.expand_dims(raw, axis=2)
#         H, W = raw.shape[0], raw.shape[1]
#         m = {'R': 0, 'G': 1, 'B': 2}
#         v_im = np.zeros([H, W, 3], dtype=np.float32)
#         v_im[0:H:2, 0:W:2, m[pattern[0]]] = raw[0:H:2, 0:W:2]
#         v_im[0:H:2, 1:W:2, m[pattern[1]]] = raw[0:H:2, 1:W:2]
#         v_im[1:H:2, 0:W:2, m[pattern[2]]] = raw[1:H:2, 0:W:2]
#         v_im[1:H:2, 1:W:2, m[pattern[3]]] = raw[1:H:2, 1:W:2]
#         return v_im
#
#     def __getitem__(self, index):
#         # 先测试一下shape是否相同
#         raw_path = self.raw_list[index]
#         raw_img = read_raw(raw_path)
#         meta_path_w = self.meta_list[index]
#         meta_path = meta_path_w.replace(self.data_root, self.meta_root)
#         flip_val, bayer, h, w, filename = read_metadata(meta_path)
#         # if flip_val == 1:
#         #     raw_img = raw_img.transpose()
#         # else:
#         #     # todo
#         #     assert 0 == 1
#         # raw_shape = raw_img.shape
#         # raw_img = self.visualize_raw(raw_img, bayer)
#         # print(bayer)
#         rgb_path = self.srgb_list[index]
#         rgb_img = self.imio.read(rgb_path)
#         # rgb_shape = rgb_img.shape
#         # raw_img, rgb_img = random_crop(int(self.patch_size/2), raw_img, rgb_img)
#         # rgb_img = rgb_img / 255
#         # raw_img = torch.Tensor(raw_img).permute(2, 0, 1)
#         # rgb_img = torch.Tensor(rgb_img).permute(2, 0, 1)
#         return {
#             'raw_img': raw_img,
#             'rgb_img': rgb_img,
#             'height': h,
#             'width': w,
#             'filename': filename,
#             'flip_value': flip_val,
#             'raw_path': raw_path
#         }
#
#
#         # print(raw_shape, rgb_shape, flip_val)
#
#
#
#
#     def __len__(self):
#         return self.len_data

def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


class SIDD_Dataset(Dataset):
    def __init__(self, dataset_root, patch_size=256):
        data_root = os.path.join(dataset_root, 'data_npz')
        # metadata_root = os.path.join(dataset_root, 'metadata')
        self.data_root = data_root
        # self.meta_root = metadata_root
        self.raw_list = get_file_list(data_root, [], 'RAW')
        # self.srgb_list = [replace_raw_path(i, 'RAW', 'SRGB', '.PNG') for i in self.raw_list]
        # self.meta_list = [replace_raw_path(i, 'GT', 'METADATA', '.MAT') for i in self.raw_list]
        self.len_data = len(self.raw_list)
        self.imio = imio('RGB', 'HWC')
        self.patch_size = patch_size


    def visualize_raw(self, raw, pattern='RGGB'):
        # 两个相机都是RGGB
        # im = np.expand_dims(raw, axis=2)
        H, W = raw.shape[0], raw.shape[1]
        m = {'R': 0, 'G': 1, 'B': 2}
        v_im = np.zeros([H, W, 3], dtype=np.float32)
        v_im[0:H:2, 0:W:2, m[pattern[0]]] = raw[0:H:2, 0:W:2]
        v_im[0:H:2, 1:W:2, m[pattern[1]]] = raw[0:H:2, 1:W:2]
        v_im[1:H:2, 0:W:2, m[pattern[2]]] = raw[1:H:2, 0:W:2]
        v_im[1:H:2, 1:W:2, m[pattern[3]]] = raw[1:H:2, 1:W:2]
        return v_im

    def __getitem__(self, index):
        # 先测试一下shape是否相同
        raw_path = self.raw_list[index]
        # raw_img = read_raw(raw_path)
        rgb_path = raw_path.replace('data_npz', 'data_srgb').replace('.npz', '_tone.png')
        data = np.load(raw_path)
        # return {
        #     'raw_img': raw_img,
        #     'rgb_img': rgb_img,
        #     'height': h,
        #     'width': w,
        #     'filename': filename,
        #     'flip_value': flip_val,
        #     'raw_path': raw_path
        # }
        raw_img = data['raw_img']
        rgb_img = self.imio.read(rgb_path)
        # rgb_img = data['rgb_img']
        h, w = int(data['height']), int(data['width'])
        filename = str(data['filename'])
        flip_val = int(data['flip_value'])
        # meta_path_w = self.meta_list[index]
        # meta_path = meta_path_w.replace(self.data_root, self.meta_root)
        # flip_val, bayer, h, w, filename = read_metadata(meta_path)
        raw_img = raw_img.transpose()
        raw_img = fix_orientation(raw_img, flip_val)
        # if flip_val != 1:
        #     print('here', flip_val)
        # raw_shape = raw_img.shape
        bayer = read_bayer_pattern(filename)
        raw_img = self.visualize_raw(raw_img, bayer)
        # print(bayer)
        # rgb_path = self.srgb_list[index]
        # rgb_img = self.imio.read(rgb_path)
        # rgb_shape = rgb_img.shape
        raw_img, rgb_img = random_crop(int(self.patch_size/2), raw_img, rgb_img)
        target_raw = raw_img.copy()
        rgb_img = rgb_img / 255
        raw_img = torch.Tensor(raw_img).permute(2, 0, 1)
        target_raw = torch.Tensor(target_raw).permute(2, 0, 1)
        rgb_img = torch.Tensor(rgb_img).permute(2, 0, 1)
        return {
            'input_raw': raw_img,
            'target_rgb': rgb_img,
            'target_raw': target_raw,
            'height': h,
            'width': w,
            'file_name': filename,
            'flip_value': flip_val,
            'raw_path': raw_path,
            'camera_name': 'test_smartphone'
        }


        # print(raw_shape, rgb_shape, flip_val)




    def __len__(self):
        return self.len_data


# 用来测试数据集类是否有误
if __name__ == '__main__':
    dataset_root = '/groupshare/SIDD_xxhu/'
    txt_file = 'meta.pickle'
    dataset = SIDD('/groupshare/SIDD_xxhu/',txt_file)
    len_dataset = len(dataset)
    print(len_dataset)
    raw_list = []
    srgb_list = []
    meta_list = []
    for i in tqdm(range(len_dataset)):
        item = dataset[i]
        # try:
        #     flip_value, bayer, h, w, filename, wb, black_level, white_level, raw_path, srgb_path = dataset[i]
        # except Exception as e:
        #     continue
        # if flip_value in [0, 1, 3, 6]:
        #     raw_list.append(raw_path)
        #     srgb_list.append(srgb_path)
        #     meta_list.append(
        #         {
        #             'flip_val': flip_value,
        #             'bayer': bayer,
        #             'filename': filename,
        #             'wb': wb,
        #             'black_level': black_level,
        #             'white_level': white_level,
        #         }
        #     )
    # print(len(raw_list))
    # result = {
    #     'raw_list': raw_list,
    #     'srgb_list': srgb_list,
    #     'meta_list': meta_list
    # }
    # import pickle
    # with open(os.path.join(dataset_root, 'meta.pickle'), 'wb') as f:
    #     pickle.dump(result, f)
        # raw = dataset[i]['input_raw']
        # srgb = dataset[i]['target_rgb']
        # target_raw = dataset[i]['target_raw']

        # if not os.path.exists(metadata):
        #     # print('here')
        #     print(raw, os.path.exists(metadata))

        # print(raw, os.path.exists(metadata))
    # dataset = SIDD_Dataset(dataset_root, patch_size=256)
    # for i in tqdm(range(len(dataset))):
    #     item = dataset[i]
    # metadata_root = os.path.join(dataset_root, 'metadata')
    # file_list = get_file_list(metadata_root, [], 'METADATA')
    # cur_phone = ''
    # for file_path in file_list:
    #     metadata = rawio.loadmat(file_path)['metadata']
    #     field_list = metadata.dtype.names
    #     cur_phone = get_cfa_pattern_mat(metadata, field_list, cur_phone=cur_phone)

# new_path = item['raw_path'].replace('/home/groupshare/SIDD_xxhu/data',
        #                                     '/home/groupshare/SIDD_xxhu/data_npz')[:-4]+'.npz'
        # if os.path.exists(new_path):
        #     continue
        # dir_name = os.path.dirname(new_path)
        # if not os.path.exists(dir_name):
        #     os.makedirs(dir_name)
        # np.savez(new_path, raw_img=item['raw_img'], rgb_img=item['rgb_img'],
        #          height=item['height'], width=item['width'], filename=item['filename'],
        #          flip_value=item['flip_value'])
        # print('1')
    # raw_list = get_file_list(test_raw_path, [], 'RAW')
    # rgb_list = get_file_list(test_raw_path, [], 'RGB')
    # srgb_list = [i.replace('RAW', 'SRGB')[:-4]+'.PNG' for i in raw_list]
    # c = set(rgb_list) - set(srgb_list)
    # print('1')
    # data = h5py.File(test_raw_path, 'r')
    # test = data['x'][:, :]
    # cv2.imwrite('./test.png', test)

    # test_meta_path = '/home/groupshare/SIDD_xxhu/metadata/0001_METADATA_RAW/0001_METADATA_RAW_001.MAT'
    # test_meta_path = '/home/groupshare/SIDD_xxhu/metadata/'
    # t = os.listdir(test_meta_path)
    # data = h5py.File(test_meta_path, 'r')
    # test = data['metadata']
    # file_lists = get_file_list(test_meta_path, [])
    # for file_path in file_lists:
    #     print(get_cropped_area(file_path))
    # test = rawio.loadmat(test_meta_path)['metadata']
    # aa = test['ActiveArea']
    # sub = test['SubIFDs']

