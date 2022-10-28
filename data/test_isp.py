import os
import random

import imageio
import rawpy
import data.imaging as imaging
import numpy as np


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

"""
默认完成的步骤：黑电平校正 白平衡 去马赛克 (去马赛克伪影去除) 颜色校正 伽马校正 色调映射
"""
def postprocess(raw_image, metadata, input_stage='raw', output_stage='tone_mapping'):
    current_stage = 'raw'
    data = raw_image
    if input_stage == current_stage:
        black_level = metadata['black_level']
        white_level = metadata['white_level']
        data = imaging.black_level_correction(data, black_level, white_level, normalized=True, clip_range=None)
        input_stage = 'normalize'

    current_stage = 'normalize'
    if output_stage == current_stage:
        return data

    if input_stage == current_stage:
        channel_gain = metadata['channel_gain']
        data = imaging.channel_gain_white_balance(data, channel_gain)
        input_stage = 'whitebalance'

    current_stage = 'whitebalance'
    if output_stage == current_stage:
        return data

    if input_stage == current_stage:
        bayer_pattern = metadata['bayer_pattern']
        data = imaging.demosaic(data, bayer_pattern, clip_range=[0, 1]).directionally_weighted_gradient_based_interpolation()
        input_stage = 'demosaic'

    current_stage = 'demosaic'
    if output_stage == current_stage:
        return data

    if input_stage == current_stage:
        bayer_pattern = metadata['bayer_pattern']
        data = imaging.demosaic(data, bayer_pattern, clip_range=[0, 1]).post_process_local_color_ratio(0.80*65535)
        edge_detection_kernel_size = 5
        edge_threshold = 0.05
        data, _ = imaging.demosaic(data, bayer_pattern, clip_range=[0, 1]).post_process_median_filter(
            edge_detection_kernel_size, edge_threshold)
        input_stage = 'demosaic_artifact_reduction'

    current_stage = 'demosaic_artifact_reduction'
    if output_stage == current_stage:
        return data

    if input_stage == current_stage:
        color_matrix = metadata['color_matrix']
        data = imaging.color_correction(data, color_matrix).apply_cmatrix()
        input_stage = 'color_correction'

    current_stage = 'color_correction'
    if output_stage == current_stage:
        return data

    if input_stage == current_stage:
        data = imaging.nonlinearity(data, 'brightening').luma_adjustment(95.)
        data = imaging.nonlinearity(data, 'gamma').by_value(1/2.2, [0, 1])
        input_stage = 'gamma'

    current_stage = 'gamma'
    if output_stage == current_stage:
        return data

    if input_stage == current_stage:
        data = imaging.tone_mapping(data).nonlinear_masking(1.0)
        input_stage = 'tone_mapping'

    current_stage = 'tone_mapping'
    if output_stage == current_stage:
        return data
    return ''


# bayer_list = ['RGGB', 'GBRG', 'BGGR', 'GRBG']
bayer_list = ['rggb', 'gbrg', 'bggr', 'grbg']

def get_bayer_pattern(raw_pattern, ori_str, flip_val):
    ori_str = bytes.decode(ori_str)
    raw_pattern = raw_pattern.flatten()
    bayer_pattern = ''.join([ori_str[i] for i in raw_pattern])

    bayer_patterns = {
        'RGGB': 0,
        'GBRG': 1,
        'BGGR': 2,
        'GRBG': 3
    }
    origin_bayer = bayer_patterns[bayer_pattern]
    if flip_val == 5:
        bayer_pattern = (origin_bayer + 1) % 4
    elif flip_val == 3:
        bayer_pattern = (origin_bayer + 2) % 4
    elif flip_val == 6:
        bayer_pattern = (origin_bayer + 3) % 4
    else:
        bayer_pattern = origin_bayer
    return bayer_list[bayer_pattern]

def get_metadata(raw):
    black_level = raw.black_level_per_channel
    # white_level = [raw.white_level] * 4
    white_level = raw.camera_white_level_per_channel
    color_matrix = raw.rgb_xyz_matrix[:3]
    bitdepth = raw.sizes.top_margin
    camera_whitebalance = raw.camera_whitebalance
    camera_whitebalance = [i/max(camera_whitebalance) for i in camera_whitebalance]
    raw_pattern, color_desc = raw.raw_pattern, raw.color_desc
    flip_val = raw.sizes.flip
    bayer_pattern = get_bayer_pattern(raw_pattern, color_desc, flip_val)
    default_mapping = {
        'r': 0, 'g': 1, 'b': 2
    }
    channel_gain = [camera_whitebalance[default_mapping[i]] for i in bayer_pattern]
    return {
        'black_level': black_level,
        'white_level': white_level,
        'color_matrix': color_matrix,
        'bitdepth': bitdepth,
        'channel_gain': channel_gain,
        'bayer_pattern': bayer_pattern,
        'flip_val': flip_val
    }


if __name__ == '__main__':
    import pickle
    with open('/groupshare/raise/metadata.pickle', 'rb') as f:
        data = pickle.load(f)
    exit(0)
    np.seterr(divide='ignore', invalid='ignore')
    # test_raw_path = 'images/a0031-WP_CRW_0736.dng'
    # test_raw_path = './rfffb852et.NEF'
    data_root = '/groupshare/raise/RAW/'
    dng_files = os.listdir(data_root)
    pickle_data = {}
    import pickle
    from tqdm import tqdm
    # random.shuffle(dng_files)
    for file_name in tqdm(dng_files):
        test_raw_path = os.path.join(data_root, file_name)
        raw = rawpy.imread(test_raw_path)
        # rgb = raw.postprocess(no_auto_bright=True, use_auto_wb=False)
        # imageio.imwrite('./test2.png', rgb)
        metadata = get_metadata(raw)
        pickle_data[file_name[:-4]] = metadata
        # raw_image = raw.raw_image_visible.copy()
        # raw_image = flip(raw_image, metadata['flip_val'])
        # raw_image = raw_image
        # data = postprocess(raw_image, metadata)
        # import cv2
        #
        # out = (data[..., ::-1] * 255).astype(np.uint8)
        # cv2.imwrite('./test.png', out)
        # exit(0)
    with open('/groupshare/raise/metadata.pickle', 'wb') as f:
        pickle.dump(pickle_data, f)