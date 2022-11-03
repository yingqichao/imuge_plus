import math
import os

import cv2
import torch
import torch.nn as nn
# from cycleisp_models.cycleisp import Raw2Rgb
# from MVSS.models.mvssnet import get_mvss
# from MVSS.models.resfcn import ResFCN
# from data.pipeline import pipeline_tensor2image
# import matlab.engine
import torch.nn.functional as Functional
import torchvision.transforms.functional as F
from torch.nn.parallel import DistributedDataParallel
from data.pipeline import isp_tensor2image
# from data.pipeline import rawpy_tensor2image
# print("Starting MATLAB engine...")
# engine = matlab.engine.start_matlab()
# print("MATLAB engine loaded successful.")
# logger = logging.getLogger('base')
# json_path = '/qichaoying/Documents/COCOdataset/annotations/incnances_val2017.json'
# load coco data
# coco = COCO(annotation_file=json_path)
#
# # get all image index info
# ids = list(sorted(coco.imgs.keys()))
# print("number of images: {}".format(len(ids)))
#
# # get all coco class labels
# coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
# import lpips
from MantraNet.mantranet import pre_trained_model
# from .networks import SPADE_UNet
from lama_models.HWMNet import HWMNet
from lama_models.my_own_elastic_dtcwt import my_own_elastic
from models.base_model import BaseModel
# from .invertible_net import Inveritible_Decolorization_PAMI
# from models.networks import UNetDiscriminator
# from loss import PerceptualLoss, StyleLoss
# from .networks import SPADE_UNet
# from lama_models.HWMNet import HWMNet
# import contextual_loss as cl
# import contextual_loss.functional as F
from models.invertible_net import Inveritible_Decolorization_PAMI
from models.networks import UNetDiscriminator
from noise_layers import *
from utils import stitch_images
from models.ISP.Modified_invISP import Modified_invISP


class Performance_Test(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """

            this file is mode 3

        """
        super(Performance_Test, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        ### OSN performance (val)
        self.network_list = self.default_ISP_networks + self.default_RAW_to_RAW_networks
        # self.network_list += ['localizer']
        self.save_network_list = []
        self.training_network_list = []

        ### ISP networks
        self.define_ISP_network_training()
        self.load_model_wrapper(folder_name='ISP_folder', model_name='load_ISP_models',
                                network_lists=self.default_ISP_networks, strict=False)
        ### RAW2RAW network
        self.define_RAW2RAW_network()
        self.load_model_wrapper(folder_name='protection_folder', model_name='load_RAW_models',
                                network_lists=self.default_RAW_to_RAW_networks)

        ### detector
        which_model = self.opt['using_which_model_for_test']
        if 'localizer' in which_model:
            self.network_list += ['localizer']
            self.define_localizer()
        else:
            self.network_list += ['discriminator_mask']
            self.discriminator_mask = self.define_CATNET()  # self.define_my_own_elastic_as_detector()
            self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
                                    network_lists=['discriminator_mask'])


        ## inpainting model
        self.define_inpainting_edgeconnect()
        self.define_inpainting_ZITS()
        self.define_inpainting_lama()

    def define_localizer(self):

        which_model = self.opt['using_which_model_for_test']
        if 'OSN' in which_model:
            self.localizer = self.define_OSN_as_detector()
        elif 'CAT' in which_model:
            self.localizer = self.define_CATNET()
        elif 'MVSS' in which_model:
            self.localizer = self.define_MVSS_as_detector()
        elif 'Mantra' in which_model:
            self.localizer = self.define_MantraNet_as_detector()
        elif 'Resfcn' in which_model:
            self.localizer = self.define_resfcn_as_detector()
        elif 'MPF' in which_model:
            print("using my_own_elastic as localizer.")
            self.localizer = self.define_my_own_elastic_as_detector()

        ## loading finetuned models or MPF
        if 'finetuned' in which_model or 'MPF' in which_model:
            ## 注意！！这里可能涉及到给模型改名
            self.load_model_wrapper(folder_name='localizer_folder', model_name='load_localizer_models',
                                    network_lists=['localizer'])

    @torch.no_grad()
    def get_performance_of_OSN(self, step):
        gt_rgb = self.label_val

        input_raw_one_dim = self.real_H_val
        file_name = self.file_name_val
        camera_name = self.camera_name_val

        ### tensor sized (B,3)
        camera_white_balance = self.camera_white_balance_val
        ### tensor sized (B,1) ranging from [0,3]
        bayer_pattern = self.bayer_pattern_val
        input_raw = self.visualize_raw(input_raw_one_dim, bayer_pattern=bayer_pattern,
                                       white_balance=camera_white_balance)
        batch_size, num_channels, height_width, _ = gt_rgb.shape

        logs = {}
        logs['lr'] = 0

        test_model = self.localizer if "localizer" in self.opt['which_model_for_detector'] else self.discriminator_mask
        # if "MPF" in self.opt['which_model_for_detector']:
        #     test_model = self.discriminator_mask
        # elif "MVSS" in self.opt['which_model_for_detector']:
        #     test_model = self.discriminator
        # elif "OSN" in self.opt['which_model_for_detector']:
        #     test_model = self.localizer
        # else:
        #     raise NotImplementedError("Detector名字不对，请检查！")
        test_model.eval()

        ### if the model requires image protection?
        ### generate gt_rgb using ISP instead of directly loading
        gt_rgb = self.render_image_using_ISP(input_raw=input_raw, input_raw_one_dim=input_raw_one_dim, gt_rgb=gt_rgb,
                                             file_name=file_name, camera_name=camera_name)

        ### generate non-tampered protected (or not) image
        if "localizer" not in self.opt['using_which_model_for_test'] or 'finetuned' in self.opt[
            'using_which_model_for_test']:
            ## RAW protection ##
            if not self.opt["test_baseline"]:
                self.KD_JPEG.eval()
                modified_raw_one_dim = self.RAW_protection_by_my_own_elastic(input_raw_one_dim=input_raw_one_dim)

                ### model selection
                modified_raw = self.visualize_raw(modified_raw_one_dim, bayer_pattern=bayer_pattern,
                                                  white_balance=camera_white_balance)
                RAW_L1 = self.l1_loss(input=modified_raw, target=input_raw)
                RAW_PSNR = self.psnr(self.postprocess(modified_raw), self.postprocess(
                    input_raw)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
                logs['RAW_PSNR'] = RAW_PSNR
                logs['RAW_L1'] = RAW_L1.item()
                # RAW_L1_REV = self.l1_loss(input=raw_reversed, target=input_raw_one_dim)
                input_raw = self.clamp_with_grad(modified_raw)
                input_raw_one_dim = self.clamp_with_grad(modified_raw_one_dim)

            ### ISP rendering: when not using test_baseline
            if not self.opt["test_baseline"]:
                non_tampered_image = self.render_image_using_ISP(input_raw=input_raw,
                                                                 input_raw_one_dim=input_raw_one_dim, gt_rgb=gt_rgb,
                                                                 file_name=file_name, camera_name=camera_name)

            ## RGB protection ##
            else:
                self.KD_JPEG.eval()
                # print("rgb protection")
                non_tampered_image = self.baseline_generate_protected_rgb(gt_rgb=gt_rgb)

        ## test on OSN, skip image protection ###
        else:
            # print('test osn/cat/mvss/resfcn')
            non_tampered_image = gt_rgb

            # print("remember to remove this gaussian blur, we use it to beat CAT!!!")
            # non_tampered_image = self.median_blur(non_tampered_image)

        RGB_PSNR = self.psnr(self.postprocess(non_tampered_image), self.postprocess(
            gt_rgb)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
        logs['RGB_PSNR'] = RGB_PSNR

        # for i in range(batch_size):
        #     step = step % 130
        #     file_name = "%05d.png" % ((step * batch_size + i) % 758)

        ### cropping
        locs, _, non_tampered_image = self.cropping_mask_generation(
            forward_image=non_tampered_image, min_rate=self.opt['cropping_lower_bound'], max_rate=1.0)
        _, _, gt_rgb = self.cropping_mask_generation(forward_image=gt_rgb, locs=locs)

        ### get tampering source and mask
        if self.opt['inference_load_real_world_tamper']:
            all_tamper_source = None
            all_mask_GT = None
            for i in range(batch_size):
                # step = step % 23
                file_name = "%05d.png" % ((step * batch_size + i) % 757 + 1)
                folder_name = '/groupshare/ISP_results/test_results/forged/'  # f'/groupshare/ISP_results/xxhu_test/{self.task_name}/FORGERY_{idx_isp}/'
                mask_file_name = file_name  # f"{str(step).zfill(5)}_0_{str(self.rank)}.png"
                mask_folder_name = '/groupshare/ISP_results/test_results/mask/'  # f'/groupshare/ISP_results/xxhu_test/{self.task_name}/MASK/'
                # print(f"reading {folder_name+file_name}")
                img_GT = cv2.imread(folder_name + file_name, cv2.IMREAD_COLOR)
                mask_GT = cv2.imread(mask_folder_name + mask_file_name, cv2.IMREAD_GRAYSCALE)

                img_GT = img_GT.astype(np.float32) / 255.
                if img_GT.ndim == 2:
                    img_GT = np.expand_dims(img_GT, axis=2)
                # some images have 4 channels
                if img_GT.shape[2] > 3:
                    img_GT = img_GT[:, :, :3]
                mask_GT = mask_GT.astype(np.float32) / 255.

                orig_height, orig_width, _ = img_GT.shape
                H, W, _ = img_GT.shape

                mask_GT = torch.from_numpy(np.ascontiguousarray(mask_GT)).float().unsqueeze(0).unsqueeze(0).cuda()

                # BGR to RGB, HWC to CHW, numpy to tensor
                if img_GT.shape[2] == 3:
                    img_GT = img_GT[:, :, [2, 1, 0]]

                tamper_source = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float().unsqueeze(
                    0).cuda()

                if all_mask_GT is None:
                    all_mask_GT = mask_GT
                    all_tamper_source = tamper_source
                else:
                    all_mask_GT = torch.cat((all_mask_GT, mask_GT), 0)
                    all_tamper_source = torch.cat((all_tamper_source, tamper_source), 0)

            tamper_source = all_tamper_source
            mask_GT = all_mask_GT
            #### tampered image ####
            test_input = non_tampered_image * (1 - mask_GT) + tamper_source * mask_GT

        else:
            #### auto generated  ####
            if self.previous_protected is not None:
                self.previous_protected = self.label[0:1]
            self.previous_images = self.label[0:1]

            masks, masks_GT, percent_range = self.mask_generation(modified_input=gt_rgb,
                                                                  percent_range=None)

            self.previous_protected = gt_rgb

            ans = self.index_helper_for_testing(attack_indices_amounts=[
                    self.amount_of_inpainting, self.amount_of_augmentation
                ],
                indices_you_want=[
                    self.opt['edgeconnect_as_inpainting'],
                    self.opt['simulated_contrast'],
                ]
            )


            #####  conduct tampering  ######
            test_input, masks, mask_GT = self.tampering_RAW(
                masks=masks, masks_GT=masks_GT,
                modified_input=non_tampered_image, percent_range=percent_range,
                index=self.opt['inference_tamper_index'],
                gt_rgb=gt_rgb
            )

        ### attacks generate them all ###
        attack_lists = [
            (None, None, None), (0, None, None), (1, None, None), (2, None, None),
            (3, 18, None), (3, 14, None), (3, 10, None), (4, None, None),
            (None, None, 0), (None, None, 1), (None, None, 2), (None, None, 3),
        ]
        # attack_lists = [
        #     (0, 20, 1),  (1, 20, 3), (3, 18, 0),
        #     (None,None,None),(None,None,0),(None,None,1),
        #     (0, 20, None), (0, 20, 2),
        #     (1, 20, None), (1, 20, 0),
        #     (2, 20, None), (2, 20, 1), (2, 20, 2),
        #     (3, 10, None), (3, 10, 3), (3, 10, 0),
        #     (3, 14, None), (3, 14, 2), (3, 14, 1),
        #     (3, 18, None), (3, 18, 3),
        #     (4, 20, None), (4, 20, 1), (4, 20, 2),
        # ]

        do_attack, quality_idx, do_augment = attack_lists[self.opt['inference_benign_attack_begin_idx']]
        logs_pred, pred_resfcn, _ = self.get_predicted_mask(target_model=test_model,
                                                            modified_input=test_input,
                                                            masks_GT=mask_GT, do_attack=do_attack,
                                                            quality_idx=quality_idx,
                                                            do_augment=do_augment,
                                                            step=step,
                                                            filename_append="",
                                                            save_image=self.opt['inference_save_image']
                                                            )

        logs.update(logs_pred)

        return logs, (pred_resfcn), True
