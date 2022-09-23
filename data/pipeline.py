import os

import numpy as np
from .pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, \
    apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, fix_orientation, \
    lens_shading_correction
import rawpy

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


"""
raw_image: raw2raw network output, torch.tensor
metadata: load from pickle, the key is "metadata"
return: rgb image, numpy variable
"""
# todo: normalize returned rgb image with np format
def pipeline_tensor2image(*, raw_image, metadata, input_stage='normal', output_stage='gamma'):
    params = {
        'input_stage': input_stage,  # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'output_stage': output_stage,  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
        'demosaic_type': 'EA',
        'save_dtype': np.uint8
    }
    # first: transfer the torch into numpy variable
    raw_image = raw_image.cpu().numpy()
    final_rgb = run_pipeline_v2(raw_image, params, metadata, False)
    return final_rgb


"""
raw_image: raw2raw network output, torch.tensor [C H W] C = 1
template: the result of rawpy load template.dng
return: rgb image, numpy variable
"""
def rawpy_tensor2image(*, raw_image, template, camera_name, patch_size):
    if type(template) == str:
        # 传入的template参数s
        default_root = '/ssd/invISP/'
        dng_path = os.path.join(default_root, camera_name, 'DNG', template+'.dng')
        template = rawpy.imread(dng_path)
    flip_val = template.sizes.flip
    print(patch_size)
    raw_image = raw_image.permute(1, 2, 0)
    raw_image = raw_image.detach().cpu().numpy()
    raw_image = np.squeeze(raw_image, axis=2)
    raw_image = np.ascontiguousarray(unflip(raw_image, flip_val))

    if camera_name == 'Canon_EOS_5D':
        max_value = 4095
    else:
        max_value = 16383
    tmp_raw = raw_image * float(max_value)
    # tmp_raw = np.squeeze(origin_raw, axis=2)
    if camera_name == 'Canon_EOS_5D':
        tmp_raw = tmp_raw + 127.0
    template.raw_image_visible[:patch_size, :patch_size] = tmp_raw.astype(np.uint16)
    im = template.postprocess(use_camera_wb=True, no_auto_bright=True)
    im = unflip(im, flip_val)
    return flip(im[:patch_size, :patch_size, :], flip_val)

def run_pipeline_v2(image_or_path='/ssd/invISP/Canon_EOS_5D/DNG/a0004-jmac_MG_1384.dng', params=None, metadata=None, fix_orient=True):
    params_ = params.copy()
    if type(image_or_path) == str:
        image_path = image_or_path
        # raw image data
        ####################################################################################################
        # todo: raw_image should be a tensor
        ####################################################################################################
        raw_image = get_visible_raw_image(image_path)
        # metadata
        ####################################################################################################
        # todo: metadata will be loaded from pickle
        ####################################################################################################
        metadata = get_metadata(image_path)
    else:
        raw_image = image_or_path.copy()
        # must provide metadata
        if metadata is None:
            raise ValueError("Must provide metadata when providing image data in first argument.")

    current_stage = 'raw'
    current_image = raw_image

    if params_['input_stage'] == current_stage:
        # linearization
        linearization_table = metadata['linearization_table']
        if linearization_table is not None:
            print('Linearization table found. Not handled.')
            # TODO

        current_image = normalize(current_image, metadata['black_level'], metadata['white_level'])
        params_['input_stage'] = 'normal'

    current_stage = 'normal'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        gain_map_opcode = None
        if 'opcode_lists' in metadata:
            if 51009 in metadata['opcode_lists']:
                opcode_list_2 = metadata['opcode_lists'][51009]
                gain_map_opcode = opcode_list_2[9]
        if gain_map_opcode is not None:
            current_image = lens_shading_correction(current_image, gain_map_opcode=gain_map_opcode,
                                                    bayer_pattern=metadata['cfa_pattern'])
        params_['input_stage'] = 'lens_shading_correction'

    current_stage = 'lens_shading_correction'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = white_balance(current_image, metadata['as_shot_neutral'], metadata['cfa_pattern'])
        params_['input_stage'] = 'white_balance'

    current_stage = 'white_balance'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = demosaic(current_image, metadata['cfa_pattern'], output_channel_order='RGB',
                                 alg_type=params_['demosaic_type'])
        params_['input_stage'] = 'demosaic'

    current_stage = 'demosaic'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = apply_color_space_transform(current_image, metadata['color_matrix_1'],
                                                    metadata['color_matrix_2'])
        params_['input_stage'] = 'xyz'

    current_stage = 'xyz'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = transform_xyz_to_srgb(current_image)
        params_['input_stage'] = 'srgb'

    current_stage = 'srgb'

    if fix_orient:
        # fix image orientation, if needed (after srgb stage, ok?)
        current_image = fix_orientation(current_image, metadata['orientation'])

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = apply_gamma(current_image)
        params_['input_stage'] = 'gamma'

    current_stage = 'gamma'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = apply_tone_map(current_image)
        params_['input_stage'] = 'tone'

    current_stage = 'tone'

    if params_['output_stage'] == current_stage:
        return current_image

    # invalid input/output stage!
    raise ValueError('Invalid input/output stage: input_stage = {}, output_stage = {}'.format(params_['input_stage'],
                                                                                              params_['output_stage']))


def run_pipeline(image_path, params):
    # raw image data
    raw_image = get_visible_raw_image(image_path)

    # metadata
    metadata = get_metadata(image_path)

    # linearization
    linearization_table = metadata['linearization_table']
    if linearization_table is not None:
        print('Linearization table found. Not handled.')
        # TODO

    normalized_image = normalize(raw_image, metadata['black_level'], metadata['white_level'])

    if params['output_stage'] == 'normal':
        return normalized_image

    white_balanced_image = white_balance(normalized_image, metadata['as_shot_neutral'], metadata['cfa_pattern'])

    if params['output_stage'] == 'white_balance':
        return white_balanced_image

    demosaiced_image = demosaic(white_balanced_image, metadata['cfa_pattern'], output_channel_order='BGR',
                                alg_type=params['demosaic_type'])

    # fix image orientation, if needed
    demosaiced_image = fix_orientation(demosaiced_image, metadata['orientation'])

    if params['output_stage'] == 'demosaic':
        return demosaiced_image

    xyz_image = apply_color_space_transform(demosaiced_image, metadata['color_matrix_1'], metadata['color_matrix_2'])

    if params['output_stage'] == 'xyz':
        return xyz_image

    srgb_image = transform_xyz_to_srgb(xyz_image)

    if params['output_stage'] == 'srgb':
        return srgb_image

    gamma_corrected_image = apply_gamma(srgb_image)

    if params['output_stage'] == 'gamma':
        return gamma_corrected_image

    tone_mapped_image = apply_tone_map(gamma_corrected_image)
    if params['output_stage'] == 'tone':
        return tone_mapped_image

    output_image = None
    return output_image

if __name__ == '__main__':
    run_pipeline_v2(image_or_path='/ssd/invISP/Canon_EOS_5D/DNG/a0004-jmac_MG_1384.dng')