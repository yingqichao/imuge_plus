import glob
import os

import cv2
import imageio
import numpy as np
import rawpy
from PIL import Image

from .pipeline import run_pipeline_v2


def my_own_pipeline(images_dir, output_dir=None):
    ####################################################################################################
    # todo: Finetuning ISP pipeline and training identity function on RAW2RAW
    # todo: kept frozen are the networks: invISP, mantranet (+2 more)
    # todo: training: RAW2RAW network (which is denoted as KD-JPEG)
    ####################################################################################################
    params = {
        'input_stage': 'raw',  # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'output_stage': 'gamma',  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
        'demosaic_type': 'EA',
        'save_dtype': np.uint8
    }

    # processing a directory
    # images_dir = './data/'
    if output_dir is None:
        output_dir = images_dir
    image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
    for image_path in image_paths:

        # render
        output_image = run_pipeline_v2(image_path, params)

        # save
        out_image_name = os.path.basename(image_path)
        output_image_name = out_image_name.replace('.dng', '_{}.'.format(params['output_stage']) + params['save_as'])
        output_image_path = os.path.join(output_dir, output_image_name)
        max_val = 2 ** 16 if params['save_dtype'] == np.uint16 else 255
        output_image = (output_image[..., ::-1] * max_val).astype(params['save_dtype'])
        if params['save_as'] == 'jpg':
            cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            cv2.imwrite(output_image_path, output_image)

def tensor2raw(image_path,):
    params = {
        'input_stage': 'raw',  # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'output_stage': 'gamma',  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
        'demosaic_type': 'EA',
        'save_dtype': np.uint8
    }

    # render
    output_image = run_pipeline_v2(image_path, params)

    # save
    out_image_name = os.path.basename(image_path)
    output_image_name = out_image_name.replace('.dng', '_{}.'.format(params['output_stage']) + params['save_as'])
    output_image_path = os.path.join(output_dir, output_image_name)
    max_val = 2 ** 16 if params['save_dtype'] == np.uint16 else 255
    output_image = (output_image[..., ::-1] * max_val).astype(params['save_dtype'])
    if params['save_as'] == 'jpg':
        cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(output_image_path, output_image)

def use_rawpy(images_dir, output_dir=None):
    params = {
        'save_as': 'png',
        'jpeg_qf': 100 # if save_as == jpg
    }
    if output_dir is None:
        output_dir = images_dir
    image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
    for image_path in image_paths:
        raw = rawpy.imread(image_path)
        file_name = os.path.basename(image_path)
        save_as = params['save_as']
        output_image_name = file_name.replace('.dng', f'_rawpy.{save_as}')
        output_image_path = os.path.join(output_dir, output_image_name)
        im = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        if save_as == 'jpg':
            Image.fromarray(im).save(output_image_path, quality=params['jpeg_qf'], subsampling=1)
        else:
            Image.fromarray(im).save(output_image_path, subsampling=1)


def save_raw_png(images_dir, output_dir=None):
    if output_dir is None:
        output_dir = images_dir
    dng_files = glob.glob(os.path.join(images_dir, '*.dng'))
    for raw_file in dng_files:
        raw_name = os.path.basename(raw_file)
        raw = rawpy.imread(raw_file)
        raw = raw.raw_image_visible
        png_image = raw.astype(np.uint16)
        new_name = raw_name.replace('.dng', '.png')
        new_path = os.path.join(output_dir, new_name)
        imageio.imwrite(new_path, png_image)


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

def save_raw_npz(images_dir, output_dir=None):
    if output_dir is None:
        output_dir = images_dir
    dng_files = glob.glob(os.path.join(images_dir, '*.dng'))
    for raw_file in dng_files:
        raw_name = os.path.basename(raw_file)
        raw = rawpy.imread(raw_file)
        flip_val = raw.sizes.flip
        # cwb = raw.camera_whitebalance
        raw_image = raw.raw_image_visible
        raw_image = np.maximum(raw_image - 127.0, 0)
        raw_image = flip(raw_image, flip_val).astype(np.uint16)
        # de_raw = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw_image, 'RGGB')
        # de_raw = flip(de_raw, flip_val)
        # png_image = raw.astype(np.uint16)
        new_name = raw_name.replace('.dng', '.npz')
        new_path = os.path.join(output_dir, new_name)
        np.savez(new_path, raw=raw_image)
        # imageio.imwrite(new_path, png_image)

# def test_read(images_dir):
#     file_names = os.listdir(images_dir)
#     for file_name in file_names:
#         file_path = os.path.join(images_dir, file_name)
#         # test = np.load(file_path)
#         test = imageio.imread(file_path)
#         # test = test.raw_image_visible
#         print(test[0][0])

def pack_raw(raw):
    # 两个相机都是RGGB
    H, W = raw.shape[0], raw.shape[1]
    raw = np.expand_dims(raw, axis=2)
    R = raw[0:H:2, 0:W:2, :]
    Gr = raw[0:H:2, 1:W:2, :]
    Gb = raw[1:H:2, 0:W:2, :]
    B = raw[1:H:2, 1:W:2, :]
    # G_avg = (Gr + Gb) / 2
    out = np.concatenate((R, Gr, Gb, B), axis=2)
    return out, H, W

def down(images_dir, out_dir):
    files = os.listdir(images_dir)
    scale_factor = 10
    for file in files:
        path = os.path.join(images_dir, file)
        image = cv2.imread(path)
        h, w = image.shape[:2]
        dst_image = cv2.resize(image, (w//scale_factor, h//scale_factor), None, 0, 0, cv2.INTER_LINEAR)
        out_path = os.path.join(out_dir, file)
        cv2.imwrite(out_path, dst_image)
#
#
# def test_down(images_dir):
#     files = os.listdir(images_dir)
#     norm_value = 4095
#     scale_factor = 10
#     for file in files:
#         path = os.path.join(images_dir, file)
#         data = np.load(path)['raw']
#         data = data / norm_value * 255
#         data_packed, h, w = pack_raw(data)
#         down_lists = []
#         for j in range(4):
#             dst = cv2.resize(data_packed[:, :, j], (w//scale_factor, h//scale_factor), None, 0, 0, cv2.INTER_LINEAR)
#             dst = np.expand_dims(dst, axis=2)
#             down_lists.append(dst)
#         test_lists = [down_lists[0]]
#         test_lists.append((down_lists[1]+down_lists[2])/2)
#         test_lists.append(down_lists[-1])
#         out = np.concatenate(test_lists, axis=2)
#         out_path = os.path.join('./test', file.replace('.npz', '.png'))
#         out = out[:, :, ::-1]
#         cv2.imwrite(out_path, out)


if __name__ == '__main__':
    # test = np.load('./data_raw_npz/a0004-jmac_MG_1384.npz')
    images_dir = 'D://DNG//'
    output_dir = 'D://DNG//'
    # down(images_dir, output_dir)
    use_rawpy(images_dir, output_dir)
    # save_raw_npz(images_dir, output_dir)
    # test_down(output_dir)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # use_rawpy(images_dir)
    # save_raw_npz(images_dir, output_dir)
    # start_time = time.time()
    # test_read(images_dir)
    # end_time = time.time()
    # print((end_time - start_time))
