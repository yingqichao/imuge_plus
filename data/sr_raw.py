import os

import imageio
import rawpy
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

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

class SrRaw(Dataset):
    def __init__(self, data_root, patch_size=512):
        self.data_root = data_root
        self.load()
        self.patch_size = patch_size

    def load(self):
        raw_files_1 = glob.glob(self.data_root+'/*/'+'%05d.ARW' % 6)
        raw_files_2 = glob.glob(self.data_root + '/*/' + '%05d.ARW' % 7)
        self.raw_files = raw_files_1 + raw_files_2
        self.rgb_files = [i.replace('.ARW', '.JPG') for i in self.raw_files]

    def __getitem__(self, item):
        raw_path = self.raw_files[item]
        raw = rawpy.imread(raw_path)
        raw_image = raw.raw_image_visible
        rgb_path = self.rgb_files[item]
        h, w = raw_image.shape
        # fake_rgb = np.zeros((h, w, 3))
        fake_rgb = imageio.imread(rgb_path)

        ori_str = bytes.decode(raw.color_desc)
        raw_pattern = raw.raw_pattern.flatten()
        bayer_pattern = ''.join([ori_str[i] for i in raw_pattern])
        bayer_pattern = {
            'RGGB': 0,
            'GBRG': 1,
            'BGGR': 2,
            'GRBG': 3
        }[bayer_pattern]

        wb = raw.camera_whitebalance
        wb = [i / max(wb) for i in wb]
        wb = wb[:3]

        raw_image, srgb_image = center_crop(self.patch_size, raw_image, fake_rgb, False)
        black_level = raw.black_level_per_channel[0]
        white_level = raw.camera_white_level_per_channel[0]
        raw_image = (raw_image - black_level) / (white_level - black_level)
        raw_image = np.clip(raw_image, 0, 1)
        raw_image = np.expand_dims(raw_image, axis=2)

        srgb_image = srgb_image / 255
        raw_image = torch.Tensor(raw_image).permute(2, 0, 1)
        srgb_image = torch.Tensor(srgb_image).permute(2, 0, 1).float()

        return {
            'input_raw': raw_image,
            'target_rgb': srgb_image,
            'camera_whitebalance': np.array(wb),
            'bayer_pattern': bayer_pattern,
            'file_name': os.path.basename(raw_path),
            'camera_name': 'test_srraw'
        }

    def __len__(self):
        return len(self.raw_files)

if __name__ == '__main__':
    data_root = '/groupshare/sr_raw/train0/'
    dataset = SrRaw(data_root)
    for i in range(len(dataset)):
        item = dataset[i]
        print(item['bayer_pattern'])
    # raw_files = glob.glob(data_root+'*/*.ARW')
    # for raw_path in raw_files:
    #     raw = rawpy.imread(raw_path)
    #     print(raw.sizes.flip)