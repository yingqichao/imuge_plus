import os
import pickle
import random

import numpy as np
import imageio
from functools import partial

def read_pickle():
    pickle_path = '/groupshare/COCOdataset/coco.pickle'
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def random_locate(h, w, image_size):
    rnd_h = random.randint(0, max(0, image_size - h))
    rnd_w = random.randint(0, max(0, image_size - w))
    return rnd_h, rnd_w


def sample_anno(pickle_data, rgb_file, selected_num):
    selected_num = int(selected_num)
    anno_list = np.random.choice(pickle_data, size=selected_num)
    height, width, _ = rgb_file.shape
    assert height == width
    mask = np.zeros((height, width))
    for anno in anno_list:
        anno_mask = anno['mask']
        anno_rgb = anno['rgb']
        anno_h, anno_w = anno_mask.shape
        anno_mask = np.expand_dims(anno_mask, axis=2)
        start_h, start_w = random_locate(anno_h, anno_w, height)
        sti_rgb = rgb_file[start_h:start_h+anno_h, start_w:start_w+anno_w, :]
        sti_rgb = np.where(anno_mask == 1, anno_rgb, sti_rgb)
        rgb_file[start_h:start_h + anno_h, start_w:start_w + anno_w, :] = sti_rgb
        anno_mask = anno_mask.squeeze(axis=2)
        sti_mask = mask[start_h:start_h + anno_h, start_w:start_w + anno_w]
        sti_mask = np.where(anno_mask == 1, anno_mask, sti_mask)
        mask[start_h:start_h + anno_h, start_w:start_w + anno_w] = sti_mask
    return rgb_file, mask


def auto_generate(coco_pickle, images_folder, forged_folder, mask_folder):
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(forged_folder, exist_ok=True)
    with open(coco_pickle, 'rb') as f:
        data = pickle.load(f)
    generate_forgery = partial(sample_anno, data)
    image_names = os.listdir(images_folder)
    rgb_size = len(image_names)
    select_array = np.random.randint(1, 4, size=rgb_size)
    for i in range(rgb_size):
        fname = image_names[i]
        cur_image = imageio.imread(os.path.join(images_folder, fname))
        sel = select_array[i]
        r, m = generate_forgery(cur_image, sel)
        imageio.imwrite(os.path.join(forged_folder, fname), r)
        imageio.imwrite(os.path.join(mask_folder, fname), m * 255)


if __name__ == '__main__':
    coco_pickle = '/groupshare/COCOdataset/coco.pickle'
    images_folder = './CAMERA_1/'
    forged_folder = './forged/'
    mask_folder = './mask/'
    auto_generate(coco_pickle, images_folder, forged_folder, mask_folder)


