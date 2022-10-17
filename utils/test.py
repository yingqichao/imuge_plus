import glob
import os
# print(os.getcwd())
from dng_util.DNG import DNG
from tqdm import tqdm
import pickle
import rawpy
import numpy as np
import shutil
import imageio
from data.pipeline_utils import get_metadata
import multiprocessing
from functools import partial


def extract_info(dataset_root):
    dng_files = glob.glob(dataset_root + '*/photos/*.dng')
    data = {}
    cam = {}
    for d_path in tqdm(dng_files):
        file_name = os.path.basename(d_path)
        dng = DNG(d_path)
        dng.openDNG()
        dng.readHeader()
        dng.readIFDs()
        tags = dng.ifd.tags
        if 272 in tags:
            t = tags[272].value
        elif 50708 in tags:
            t = tags[50708].value
        else:
            t = b'error'
        camera_name = bytes(t).decode()
        if 'FinePix F700' in camera_name:
            print('here')
        data[file_name.split('.')[-2]] = camera_name
        if camera_name not in cam:
            cam[camera_name] = 1
        else:
            cam[camera_name] += 1
    print(cam)
    with open('./dng_util/test.pickle', 'wb') as f:
        pickle.dump(data, f)


def generate_camera_txt(camera_dict, bayer_dict):
    output_path = '/ssd/FiveK_Dataset/'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    with open(bayer_dict, 'rb') as f:
        bayer_patterns = pickle.load(f)
    with open(camera_dict, 'rb') as f:
        camera_names = pickle.load(f)
    out_results = {}
    for file_name in camera_names.keys():
        if file_name in bayer_patterns and bayer_patterns[file_name] in ['RGGB', 'BGGR', 'GRBG', 'GBRG']:
            camer_name = camera_names[file_name]
            camer_name = camer_name.replace(' ', '_')
            # FinePix F700
            if camer_name not in out_results:
                out_results[camer_name] = [file_name]
            else:
                out_results[camer_name].append(file_name)
    for camer_name in out_results.keys():
        c_list = out_results[camer_name]
        print(os.path.join(output_path, f'{camer_name}.txt'))
        test = camer_name.strip(b'\x00'.decode())
        with open(os.path.join(output_path, f'{test}.txt'), 'w') as f:
            for c in c_list:
                f.write(c + '\n')


def generate_rgb(dataset_root, output_folder):
    from PIL import Image
    out_folder = os.path.join(output_folder, 'RGB')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    dng_files = glob.glob(dataset_root + '*/photos/*.dng')
    for d_path in tqdm(dng_files):
        file_name = os.path.basename(d_path).replace('.dng', '.png')
        out_path = os.path.join(output_folder, file_name)
        raw = rawpy.imread(d_path)
        im = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        Image.fromarray(im).save(out_path, subsampling=1)


def move_dngs(dataset_root, out_root, camera_dict):
    dng_files = glob.glob(dataset_root + '*/photos/*.dng')
    text_paths = os.listdir(out_root)
    fnames = []
    for text_path in text_paths:
        if not text_path.endswith('.txt'):
            continue
        with open(os.path.join(out_root, text_path), 'r') as f:
            fnames += [line.strip() for line in f.readlines()]
    with open(camera_dict, 'rb') as f:
        camera_names = pickle.load(f)
    for d_path in tqdm(dng_files):
        file_name = os.path.basename(d_path).split('.')[-2]
        if file_name not in fnames:
            continue
        camer_name = camera_names[file_name]
        camer_name = camer_name.replace(' ', '_')
        test = camer_name.strip(b'\x00'.decode())
        moved_path = os.path.join(out_root, test)
        if not os.path.exists(moved_path):
            os.makedirs(moved_path, exist_ok=True)
        shutil.move(d_path, os.path.join(moved_path, file_name + '.dng'))


def move_to_DNG(out_root):
    subs = os.listdir(out_root)
    for sub in subs:
        if sub.endswith('.txt'):
            continue
        folder = os.path.join(out_root, sub)
        dngs = os.listdir(folder)
        for dng in dngs:
            origin = os.path.join(folder, dng)
            new = os.path.join(folder, 'DNG')
            if not os.path.exists(new):
                os.makedirs(new)
            shutil.move(origin, os.path.join(new, dng))


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


# 这个函数用来跳着采样，来降低数据集里图像的大小
def data_process_skip(dataset_root, rgb_folder, camera_name):
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

    camera_raw_path = os.path.join(dataset_root, camera_name, 'DNG')
    npz_paths = sorted(glob.glob(camera_raw_path + '/*.dng'))
    output_raw_path = os.path.join(dataset_root, camera_name, 'RAW_UINT16')
    if not os.path.exists(output_raw_path):
        os.makedirs(output_raw_path, exist_ok=True)
    output_rgb_path = os.path.join(dataset_root, camera_name, 'RGB_PNG')
    if not os.path.exists(output_rgb_path):
        os.makedirs(output_rgb_path, exist_ok=True)
    # crop_util = test_crop()
    meta = {}
    for path in tqdm(npz_paths):
        file_name = os.path.basename(path)
        rgb_name = file_name.replace('.dng', '.png')
        rgb_path = os.path.join(rgb_folder, rgb_name)

        raw = rawpy.imread(path)
        flip_val = raw.sizes.flip
        cwb = raw.camera_whitebalance
        black_level = raw.black_level_per_channel
        color_matrix = raw.color_matrix
        rgb_xyz_matrix = raw.rgb_xyz_matrix
        metadata = get_metadata(path)
        metadata['black_level'] = black_level
        save_metadata = {
            'metadata': metadata,
            'color_matrix': color_matrix,
            'rgb_xyz_matrix': rgb_xyz_matrix,
            'camera_whitebalance': cwb,
            'flip_val': flip_val
        }
        meta[file_name.split('.')[-2]] = save_metadata

        raw_img = raw.raw_image_visible
        raw_img = flip(raw_img, flip_val)
        rgb = imageio.imread(rgb_path)
        rgb_h, rgb_w, _ = rgb.shape
        raw_h, raw_w = raw_img.shape
        assert raw_h == rgb_h and raw_w == rgb_w
        if raw_h > 2500 and raw_w > 2500:
            return
        rgb_h = rgb_h - (rgb_h % 8)
        rgb_w = rgb_w - (rgb_w % 8)
        rgb_resize = rgb[0:rgb_h:2, 0:rgb_w:2, :]
        raw_img = raw_img[0:rgb_h, 0:rgb_w]
        imageio.imwrite(os.path.join(output_rgb_path, rgb_name), rgb_resize)
        p_raw = pack_raw(raw_img)
        raw_resize = p_raw[0:rgb_h:2, 0:rgb_w:2, :]
        new_H, new_W = raw_resize.shape[0], raw_resize.shape[1]
        v_im = np.zeros([new_H * 2, new_W * 2], dtype=np.uint16)
        v_im[0:new_H * 2:2, 0:new_W * 2:2] = raw_resize[0:new_H, 0:new_W, 0]
        v_im[0:new_H * 2:2, 1:new_W * 2:2] = raw_resize[0:new_H, 0:new_W, 1]
        v_im[1:new_H * 2:2, 0:new_W * 2:2] = raw_resize[0:new_H, 0:new_W, 2]
        v_im[1:new_H * 2:2, 1:new_W * 2:2] = raw_resize[0:new_H, 0:new_W, 3]
        np.savez(os.path.join(output_raw_path, file_name.replace('.dng', '.npz')), raw=v_im, wb=cwb)
    import pickle
    with open(os.path.join(dataset_root, camera_name, 'metadata.pickle'), 'wb') as f:
        pickle.dump(meta, f)


def split_camera(camera_name, dataset_root, ratio=0.85):
    import random
    with open(os.path.join(dataset_root, camera_name + '.txt'), 'r') as f:
        file_list = [i.strip() for i in f.readlines()]
    random.shuffle(file_list)
    split_num = int(len(file_list) * ratio)
    train_list = file_list[:split_num]
    test_list = file_list[split_num:]
    with open(os.path.join(dataset_root, camera_name + '_train.txt'), 'w') as f:
        for c in train_list:
            f.write(c + '\n')
    with open(os.path.join(dataset_root, camera_name + '_test.txt'), 'w') as f:
        for c in test_list:
            f.write(c + '\n')


if __name__ == '__main__':
    dataset_root = '/ssd/FiveK_Dataset/'
    with open('./dng_util/camera.txt', 'r') as f:
        camera_name = [i.strip() for i in f.readlines()]
    from data.fivek_dataset import FiveKDataset_total
    from torch.utils.data import DataLoader
    import time
    from data.pipeline import pipeline_tensor2image
    from data.pipeline import rawpy_tensor2image
    from PIL import Image

    train_set = FiveKDataset_total([dataset_root]*len(camera_name), camera_name, stage='test',
                                   patch_size=512, file_nums=200)
    test_folder = './dng_util/test/'
    os.makedirs(test_folder, exist_ok=True)
    dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4, drop_last=True, pin_memory=False)
    start = time.time()

    for i, value in tqdm(enumerate(dataloader)):
        file_name = value['file_name']
        camera_name = value['camera_name']
        # print(file_name[0], value['bayer_pattern'][0])

        input_raw = value['input_raw'][0]
        # todo: 测试my own pipeline
        # metadata = train_set.metadata_list[file_name[0]]
        # flip_val = metadata['flip_val']
        # metadata = metadata['metadata']
        # metadata['flip_val'] = flip_val
        # metadata['camera_name'] = camera_name[0]
        #
        # input_raw = input_raw.permute(1, 2, 0).squeeze(2)
        # numpy_rgb = pipeline_tensor2image(raw_image=input_raw, metadata=metadata, input_stage='normal')
        # numpy_rgb = (numpy_rgb * 255).astype(np.uint8)

        # todo：测试rawpy
        # numpy_rgb = rawpy_tensor2image(raw_image=input_raw, template=file_name[0], camera_name=camera_name[0], patch_size=512)
        target_rgb = value['target_rgb'][0].permute(1, 2, 0).cpu().numpy() * 255
        target_rgb = target_rgb.astype(np.uint8)
        # print(target_rgb[0])
        # print('--split--')
        # print(numpy_rgb[0])
        # numpy_rgb = np.concatenate([numpy_rgb, target_rgb], axis=1)
        Image.fromarray(target_rgb).save(os.path.join('/groupshare/ISP_results/test_results/original_image', '%05d.png' % i), subsampling=1)

    end = time.time()
    print((end - start))
