import torch
from torch.utils.data import Dataset, DataLoader
import rawpy
import glob
import os
import numpy as np
import colour_demosaicing
from PIL import Image
import random
import imageio
import cv2
# from imageio import imread
from tqdm import tqdm


# DNG：原本的raw文件
# RAW: 去马赛克之后的raw文件
# RGB：ISP管道输出的RGB图像
# stage: train 或 test
class FiveKDataset(Dataset):
    def __init__(self, dataset_root, camera_name, stage, patch_size=256, data_mode='RAW', uncond_p=0.2,
                 file_nums=10, rgb_scale=False, npz_uint16=True):
        self.dataset_root = dataset_root
        self.camera_name = camera_name
        self.data_mode = data_mode
        self.uncond_p = uncond_p
        self.rgb_scale = rgb_scale
        self.npz_uint16 = npz_uint16
        dataset_file = os.path.join(dataset_root, camera_name+'_'+stage+'.txt')
        self.raw_files, self.rgb_files = self.load(dataset_file)
        if stage != 'train':
            self.raw_files = self.raw_files[:file_nums]
            self.rgb_files = self.rgb_files[:file_nums]

        assert len(self.raw_files) == len(self.rgb_files)
        self.patch_size = patch_size
        self.gamma = True

    def visualize_raw(self, raw):
        # 两个相机都是RGGB
        # im = np.expand_dims(raw, axis=2)
        H, W = raw.shape[0], raw.shape[1]
        v_im = np.zeros([H, W, 3], dtype=np.uint16)
        v_im[0:H:2, 0:W:2, 0] = raw[0:H:2, 0:W:2]
        v_im[0:H:2, 1:W:2, 1] = raw[0:H:2, 1:W:2]
        v_im[1:H:2, 0:W:2, 1] = raw[1:H:2, 0:W:2]
        v_im[1:H:2, 1:W:2, 2] = raw[1:H:2, 1:W:2]
        return v_im

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


    def __getitem__(self, index):
        raw_path = self.raw_files[index]
        rgb_path = self.rgb_files[index]
        # print(raw_path)
        target_rgb_img = imageio.imread(rgb_path)
        if self.data_mode == 'RAW':
            raw = np.load(raw_path)
            raw_img = raw['raw']
            if self.npz_uint16:
                raw_img = self.pack_raw(raw_img)
            wb = raw['wb']
            wb = wb / wb.max()
            input_raw_img = raw_img * wb[:-1]
        else:
            raw = rawpy.imread(raw_path)
            input_raw_img = raw.raw_image_visible
            flip_val = raw.sizes.flip
            input_raw_img = flip(input_raw_img, flip_val)
            if self.camera_name == 'Canon_EOS_5D':
                input_raw_img = np.maximum(input_raw_img - 127.0, 0)
            input_raw_img = self.visualize_raw(input_raw_img)

        input_raw_img, target_rgb_img = aug(self.patch_size, input_raw_img, target_rgb_img, self.npz_uint16)
        if self.gamma:
            norm_value = np.power(4095, 1/2.2) if self.camera_name =='Canon_EOS_5D' else np.power(16383, 1/2.2)
            input_raw_img = np.power(input_raw_img, 1/2.2)
        else:
            norm_value = 4095 if self.camera_name =='Canon_EOS_5D' else 16383
        target_rgb_img = self.norm_img(target_rgb_img, 255, self.rgb_scale)
        input_raw_img = self.norm_img(input_raw_img, max_value=norm_value)

        input_raw_img = torch.Tensor(input_raw_img).permute(2, 0, 1)
        target_rgb_img = torch.Tensor(target_rgb_img).permute(2, 0, 1)

        if random.random() < self.uncond_p:
            # null label
            input_raw_img = torch.ones_like(input_raw_img)

        sample = {'input_raw': input_raw_img, 'target_rgb': target_rgb_img,
                  'file_name': raw_path.split("/")[-1].split(".")[0]}
        return sample

    def __len__(self):
        return len(self.raw_files)

    # 这个函数用来通过文件载入数据集
    def load(self, file_name):
        input_raws = []
        target_rgbs = []

        with open(file_name, "r") as f:
            valid_camera_list = [line.strip() for line in f.readlines()]

        for i, name in enumerate(valid_camera_list):
            full_name = os.path.join(self.dataset_root, self.camera_name)
            if self.data_mode == 'RAW':
                raw_folder_name = 'RAW_UINT16' if self.npz_uint16 else 'RAW'
                input_raws.append(os.path.join(full_name, raw_folder_name, name+'.npz'))
            else:
                input_raws.append(os.path.join(full_name, 'DNG', name+'.dng'))
            target_rgbs.append(os.path.join(full_name, 'RGB', name+'.jpg'))

        return input_raws, target_rgbs

    def norm_img(self, img, max_value, scale_minus1=False):
        if scale_minus1:
            half_value = max_value / 2
            img = img / half_value -1
            # scaled to [-1, 1]
        else:
            img = img / float(max_value)
        return img



# def crop_raw(image_root):
#     pass

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

camera_crop_coords = {
    'Canon_EOS_5D': {
        'x_coord': [1168, 1168 + 512, 1168 + 512 * 2, 1168 + 512 * 3],
        'y_coord': [432, 432 + 512, 432 + 512 * 2, 432 + 512 * 3]
    },
    'NIKON_D700': {
        'x_coord': [1120, 1120 + 512, 1120 + 512 * 2, 1120 + 512 * 3],
        'y_coord': [400, 400 + 512, 400 + 512 * 2, 400 + 512 * 3]
    }
}
# 输入ndarray
def crop_raw_img(raw_image, flip_val, camera_name):
    raw_image = flip(raw_image, flip_val)
    shape = raw_image.shape
    h, w = shape
    x_coord = camera_crop_coords[camera_name]['x_coord']
    y_coord = camera_crop_coords[camera_name]['y_coord']
    coords = []
    results = []
    for i in range(len(x_coord)):
        for j in range(len(y_coord)):
            coords.append([x_coord[i], y_coord[j]])
    for k in range(len(coords)):
        x, y = coords[k]
        if h > w:
            de_raw = raw_image[x:x+512, y:y+512]
        else:
            de_raw = raw_image[y:y+512, x:x+512]
        de_raw = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(de_raw, 'RGGB')
        results.append(de_raw)
    return results

def crop_rgb_img(rgb_image, camera_name):
    shape = rgb_image.shape
    h, w = shape[:2]
    x_coord = camera_crop_coords[camera_name]['x_coord']
    y_coord = camera_crop_coords[camera_name]['y_coord']
    coords = []
    results = []
    for i in range(len(x_coord)):
        for j in range(len(y_coord)):
            coords.append([x_coord[i], y_coord[j]])
    for k in range(len(coords)):
        x, y = coords[k]
        if h > w:
            # box = (x, y, x+512, y+512)
            de_raw = rgb_image[x:x + 512, y:y + 512]
        else:
            # box = (y, x, y+512, x+512)
            de_raw = rgb_image[y:y + 512, x:x + 512]
        results.append(de_raw)
    return results

def merge_images(out_image, gt_image):
    return np.hstack([out_image, gt_image])

# 此处之前踩过大坑，一般来说图像读取的维度都是H W C 但是Image.open读取的结果是W H C 服气
def crop_image():
    dataset_root = '/home/groupshare/invISP'
    camera_name = 'Canon_EOS_5D'
    raw_path = glob.glob(dataset_root + '/' + camera_name + '/DNG' + '/*.dng')
    rgb_root = dataset_root + '/' + camera_name + '/RGB/'
    deraw_root = dataset_root + '/' + camera_name + '/RAW/'
    # dng_paths = glob.glob(dataset_root+'/'+camera_name+'/DNG'+'/*.dng')
    output_root = '/home/groupshare/invISP_crop/'
    output_path = output_root + '/' + camera_name + '/'
    # files = os.listdir(output_path+'RGB')
    # list = []
    for file_path in tqdm(raw_path):
        file_name = file_path.split('/')[-1].split('.')[0]
        rgb_path = rgb_root + file_name + '.jpg'
        # deraw_path = deraw_root + file_name + '.npz'
        raw = rawpy.imread(file_path)
        flip_val = raw.sizes.flip
        cwb = raw.camera_whitebalance
        raw_img = raw.raw_image_visible
        if camera_name == 'Canon_EOS_5D':
            raw_img = np.maximum(raw_img - 127.0, 0)
        cropped_raw_images = crop_raw_img(raw_img, flip_val, camera_name)
        # raw_shape = raw_img.shape
        # rgb_image = imageio.imread(rgb_path)
        rgb_image = cv2.imread(rgb_path)
        cropped_images = crop_rgb_img(rgb_image, camera_name)
        for l in range(len(cropped_images)):
            cropped_rgb_image = cropped_images[l]
            cropped_raw_image = cropped_raw_images[l]
            # m_images = merge_images(cropped_raw_image, cropped_rgb_image)
            cropped_raw_file_path = output_path + '/RAW/' + file_name + '_' + str(l) + '.npz'
            np.savez(cropped_raw_file_path, raw=cropped_raw_image, cwb=cwb)
            cropped_rgb_file_path = output_path + '/RGB/' + file_name + '_' + str(l) + '.jpg'
            cv2.imwrite(cropped_rgb_file_path, cropped_rgb_image)




# 这里用来分割数据集
# 85:15的比例划分得到训练集:测试集
def split_dataset(dataset_root, camera_name):
    dataset_path = os.path.join(dataset_root, camera_name)
    rgb_files = sorted(os.listdir(os.path.join(dataset_path, "RGB")))
    val_list = []
    train_file = os.path.join(dataset_root, camera_name+'_train.txt')
    test_file = os.path.join(dataset_root, camera_name+'_test.txt')
    for rgb_file in rgb_files:
        val_list.append(rgb_file.split('.')[0])
    random.shuffle(val_list)
    split_len = int(len(val_list) * 0.85)
    train_list = val_list[:split_len]
    test_list = val_list[split_len:]
    with open(train_file, "w") as f_train:
        for train_item in train_list:
            f_train.write(train_item+'\n')
    with open(test_file, "w") as f_test:
        for test_item in test_list:
            f_test.write(test_item+'\n')


def center_crop(patch_size, input_raw, target_rgb, flow=True):
    h, w, _ = input_raw.shape
    x1 = int(round((w - patch_size) / 2.))
    y1 = int(round((h - patch_size) / 2.))
    patch_input_raw = input_raw[y1:y1 + patch_size, x1:x1 + patch_size, :]
    if flow:
        patch_target_rgb = target_rgb[y1:y1+patch_size, x1:x1+patch_size, :]
    else:
        patch_target_rgb = target_rgb[y1 * 2: y1 * 2 + patch_size * 2, x1 * 2: x1 * 2 + patch_size * 2, :]
    return patch_input_raw, patch_target_rgb

def random_crop(patch_size, input_raw, target_rgb, flow=True):
    h, w, _ = input_raw.shape
    rnd_h = random.randint(0, max(0, h - patch_size))
    rnd_w = random.randint(0, max(0, w - patch_size))

    patch_input_raw = input_raw[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
    if flow:
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
        idx = random.randint(0, 2)
        input_raw = np.flip(input_raw, axis=idx).copy()
        target_rgb = np.flip(target_rgb, axis=idx).copy()

    return input_raw, target_rgb

def aug(patch_size, input_raw, target_rgb, npz_uint16=True):
    # 把随机crop换成center_crop
    input_raw, target_rgb = center_crop(patch_size, input_raw, target_rgb, not npz_uint16)

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
    dataset = FiveKDataset(data_dir, camera_name, stage, patch_size, data_mode, uncond_p, file_nums, rgb_scale, npz_uint16)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=deterministic, num_workers=num_workers, drop_last=True,
                            pin_memory=False)
    while True:
        yield from dataloader

# 这个函数用来生成用uint16格式存储的npz文件
def data_process_npz(dataset_root, camera_name):
    camera_raw_path = os.path.join(dataset_root, camera_name, 'DNG')
    dng_paths = sorted(glob.glob(camera_raw_path+'/*.dng'))
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


def test_image_downsample(dataset_root, camera_name):
    # data_process_npz(dataset_root, camera_name)
    # exit(0)
    stage = 'test'
    # dataset = FiveKDataset(dataset_root, camera_name, stage, 256, data_mode='RAW')
    # for i in range(len(dataset)):
    #     item = dataset[i]
    dataset = FiveKDataset(dataset_root, camera_name, stage, 32, 'RAW', 0, 10, True, True)
    raw_files = dataset.raw_files
    # camera_path = os.path.join(dataset_root, camera_name)
    for raw_file in raw_files:
        data = np.load(raw_file)
        raw = data['raw']
        raw = dataset.pack_raw(raw)


        # raw_path = os.path.join(camera_path, 'RAW_UINT16', raw_file)





# 这里用来处理数据集 如裁剪等
# 数据集可用 data_mode=RAW表明是对插值后的图像
if __name__ == '__main__':
    # crop_image()
    # test = np.zeros([3, 3])
    # test[0, :] = [0.8642, -0.3058, -0.0243]
    # test[1, :] = [-0.3999,  1.1033,  0.3422]
    # test[2, :] = [-0.0453,  0.1099,  0.7814]
    # c = np.linalg.inv(test)
    # print('h')
    # test = np.power(-91, 1/2.2)

    dataset_root = '/ssd/invISP/'
    camera_name = 'NIKON_D700'
    test_image_downsample(dataset_root, camera_name)
    # exit(0)
    # data_process_npz(dataset_root, camera_name)
    # exit(0)
    stage = 'test'
    # dataset = FiveKDataset(dataset_root, camera_name, stage, 256, data_mode='RAW')
    # for i in range(len(dataset)):
    #     item = dataset[i]
    data = load_data(
        data_dir = dataset_root,
        camera_name = camera_name,
        stage = stage,
        batch_size = 1,
        patch_size = 128,
        data_mode = 'RAW',
        uncond_p = 0.1,
        deterministic = False,
        num_workers = 1,
        rgb_scale=True
    )
    for i in range(10):
        item = next(data)
        print(item['input_raw'].shape)
        print(item['target_rgb'].shape)


