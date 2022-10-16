import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

# ['AsShotNeutral', 'ColorMatrix1', 'ColorMatrix2', 'angle', 'baseISO', 'bayertype',
# 'blacklevel', 'debayerMatrices', 'gamma', 'pattern', 'type', 'whitelevel']
# angle != 0的删掉

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

class DND(Dataset):
    def __init__(self, base_dir, meta_file, patch_size=512):
        import pickle
        self.base_dir = base_dir
        with open(meta_file, 'rb') as f:
            data = pickle.load(f)
        self.len_data = len(data.keys())
        self.raw_list = list(data.keys())
        self.meta_list = [data[key] for key in self.raw_list]
        self.patch_size = patch_size


    def __getitem__(self, item):
        index = int(self.raw_list[item])
        raw_path = os.path.join(self.base_dir, '%04d.mat' % (index+1))
        img = h5py.File(raw_path, 'r')
        raw_image = np.float32(np.array(img['Inoisy']).T)
        meta = self.meta_list[item]
        h, w = raw_image.shape
        fake_rgb = np.zeros((h, w, 3))

        bayer = meta['bayer_pattern']
        rgb_mapping = {1:'R', 2:'G', 3:'B'}
        bayer_pattern = ''.join([rgb_mapping[i] for i in bayer[0]+bayer[1]])
        bayer_pattern = {
            'RGGB': 0,
            'GBRG': 1,
            'BGGR': 2,
            'GRBG': 3
        }[bayer_pattern]

        wb = meta['wb']
        wb_inv = [1 / i[0] for i in wb]
        wb_inv = [wb_inv[j] / max(wb_inv) for j in range(len(wb_inv))]

        raw_image, srgb_image = center_crop(self.patch_size, raw_image, fake_rgb, False)
        raw_image = np.expand_dims(raw_image, axis=2)

        # black_level = meta['black_level']
        # white_level = meta['white_level']
        # raw_image = (raw_image - black_level) / (white_level - black_level)
        # raw_image = np.clip(raw_image, 0, 1)

        raw_image = torch.Tensor(raw_image).permute(2, 0, 1)
        srgb_image = torch.Tensor(srgb_image).permute(2, 0, 1).float()
        return {
            'input_raw': raw_image,
            'target_rgb': srgb_image,
            'camera_whitebalance': np.array(wb_inv),
            'bayer_pattern': bayer_pattern,
            'file_name': index,
            'camera_name': 'test_dnd'
        }


    def __len__(self):
        return self.len_data

def get_metadata(info_data, index):
    camera_info = info_data[info_data['camera'][0][index]]
    bayer_pattern = np.asarray(camera_info['pattern']).astype(np.int32).tolist()
    angle = np.asarray(camera_info['angle']).item()
    wb = np.asarray(camera_info['AsShotNeutral']).tolist()
    black_level = np.asarray(camera_info['blacklevel']).item()
    white_level = np.asarray(camera_info['whitelevel']).item()
    return {
        'bayer_pattern': bayer_pattern,
        'angle': angle,
        'wb': wb,
        'black_level': black_level,
        'white_level': white_level
    }

if __name__ == '__main__':
    import h5py, pickle
    dnd_root = '/groupshare/dnd_raw/'
    info_path = os.path.join(dnd_root, 'info.mat')
    dataset = DND('/groupshare/dnd_raw/', './dnd.pickle')
    for i in range(len(dataset)):
        test = dataset[i]
    exit(0)
    info_data = h5py.File(info_path, 'r')['info']
    result_data = {}
    for i in range(50):
        filename = os.path.join(dnd_root, '%04d.mat' % (i + 1))
        # img = h5py.File(filename, 'r')
        # noisy = np.float32(np.array(img['Inoisy']).T)
        meta = get_metadata(info_data, i)
        if meta['angle'] != 0:
            continue
        result_data[str(i)] = meta
    print(len(result_data))
    with open('./dnd.pickle', 'wb') as f:
        pickle.dump(result_data, f)