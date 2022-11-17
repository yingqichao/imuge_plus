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


# 只用来检测在受保护图上做的篡改
class Localization_Mode_One(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """

            this file is mode 1
            this model does not implement abstract method

        """
        super(Localization_Mode_One, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        """

            currently the same as mode 0. feel free to change.

        """

        # self.network_list = self.default_ISP_networks + self.default_RAW_to_RAW_networks + self.default_detection_networks_for_training
        # self.save_network_list = []
        # self.training_network_list = []
        #
        # ### ISP networks
        # self.define_ISP_network_training()
        # self.load_model_wrapper(folder_name='ISP_folder', model_name='load_ISP_models',
        #                         network_lists=self.default_ISP_networks)
        # ### RAW2RAW network
        # self.define_RAW2RAW_network()
        # self.load_model_wrapper(folder_name='protection_folder', model_name='load_RAW_models',
        #                         network_lists=self.default_RAW_to_RAW_networks)

        ### detector
        which_model = self.opt['which_model_for_detector']
        if 'localizer' in which_model:
            self.network_list += ['localizer']
            self.localizer = self.define_detector(opt_name='using_which_model_for_test')
            ## loading finetuned models or MPF
            which_model_detailed = self.opt['using_which_model_for_test']
            if 'finetuned' in which_model_detailed or 'MPF' in which_model_detailed:
                ## 注意！！这里可能涉及到给模型改名
                self.load_model_wrapper(folder_name='localizer_folder', model_name='load_localizer_models',
                                        network_lists=['localizer'])
        else:
            self.network_list += ['discriminator_mask']
            self.discriminator_mask = self.define_detector(opt_name='using_which_model_for_test')
            self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
                                    network_lists=['discriminator_mask'])

        # ### detector networks: 这个可能需要改一下，要看到底加载哪个模型，现在是MPF
        # print("using my_own_elastic as discriminator_mask.")
        # self.discriminator_mask = self.define_my_own_elastic_as_detector()
        # self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
        #                         network_lists=['discriminator_mask'])

    def predict(self, step):
        modified_input = self.real_H_val
        mask_GT = self.canny_image_val

        test_model = self.localizer if 'localizer' in self.opt['which_model_for_detector'] else self.discriminator_mask
        test_model.eval()

        logs = {}
        logs['lr'] = 0

        attack_list = [
            (None, None, None), (0, None, None), (1, None, None), (2, None, None),
            (3, 18, None), (3, 14, None), (3, 10, None), (4, None, None),
            (None, None, 0), (None, None, 1), (None, None, 2), (None, None, 3),
        ]
        do_attack, quality_idx, do_augment = attack_list[self.opt['inference_benign_attack_begin_idx']]
        logs_pred, pred_resfcn, _ = self.get_predicted_mask(target_model=test_model,
                                                            modified_input=modified_input,
                                                            masks_GT=mask_GT, do_attack=do_attack,
                                                            quality_idx=quality_idx,
                                                            do_augment=do_augment,
                                                            step=step,
                                                            filename_append="",
                                                            save_image=self.opt['inference_save_image']
                                                            )

        logs.update(logs_pred)

        return logs, (pred_resfcn), True

    @torch.no_grad()
    def get_predicted_mask(self, target_model=None, modified_input=None, masks_GT=None, save_image=True,
                           do_attack=None, do_augment=None, quality_idx=None, step=None, filename_append=""):

        if target_model is None:
            target_model = self.discriminator_mask
        target_model.eval()
        # if modified_input is None:
        #     modified_input = self.real_H
        # if masks_GT is None:
        #     masks_GT = self.canny_image
        logs = {}
        logs['lr'] = 0
        batch_size, num_channels, height_width, _ = modified_input.shape

        ###################################   ISP attack   #############################################################
        if do_augment is not None:
            modified_adjusted = self.data_augmentation_on_rendered_rgb(modified_input, index=do_augment)
        else:
            modified_adjusted = modified_input

        attacked_forward = modified_adjusted

        #####################################   Benign attacks   #######################################################
        ### including JPEG compression Gaussian Blurring, Median blurring and resizing
        if do_attack is not None:
            if quality_idx is None:
                quality_idx = self.get_quality_idx_by_iteration(index=self.global_step)

            attacked_image, _, _ = self.benign_attacks(attacked_forward=attacked_forward,
                                                       quality_idx=quality_idx, index=do_attack)
        else:
            attacked_image = attacked_forward

        # ERROR = attacked_image-attacked_forward
        error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(
            attacked_forward)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
        logs['ERROR'] = error_l1

        #########################   Image Manipulation Detection Network (Downstream task)   ###########################
        # if "localizer" in self.opt['using_which_model_for_test']:
        pred_resfcn, CE_resfcn = self.detector_predict(model=target_model,
                                                       attacked_image=attacked_image.detach().contiguous(),
                                                       opt_name='using_which_model_for_test',
                                                       masks_GT=masks_GT)

        logs['CE'] = CE_resfcn.item()

        F1, RECALL, AUC, IoU = self.F1score(pred_resfcn, masks_GT, thresh=0.5, get_auc=True)
        # F1_1, RECALL_1 = self.F1score(refined_resfcn_bn, masks_GT, thresh=0.5)
        logs['F1'] = F1
        # logs['F1_1'] = F1_1
        logs['RECALL'] = RECALL
        logs['AUC'] = AUC
        logs['IoU'] = IoU
        # logs['RECALL_1'] = RECALL_1

        pred_resfcn_bn = torch.where(pred_resfcn > 0.5, 1.0, 0.0)

        if save_image:
            dataset_predict_root = self.opt['datasets']['val']['dataroot_Predict']
            name = f"{dataset_predict_root}/{self.opt['using_which_model_for_test']}/004/"
            print('saving sample ' + name)
            for image_no in range(batch_size):
                # self.print_this_image(modified_input[image_no],
                #                       f"{name}/{str(step).zfill(5)}_{filename_append}tampered_{self.model_path}_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")

                self.print_this_image(pred_resfcn[image_no],
                                      f"{name}/{str(step).zfill(5)}_{image_no}_{filename_append}pred_ce_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                self.print_this_image(pred_resfcn_bn[image_no],
                                      f"{name}/{str(step).zfill(5)}_{image_no}_{filename_append}pred_cebn_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                # self.print_this_image(refined_resfcn[image_no],
                #                       f"{name}/{str(step).zfill(5)}_{filename_append}pred_L1_{self.model_path}_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                # self.print_this_image(refined_resfcn_bn[image_no],
                #                       f"{name}/{str(step).zfill(5)}_{filename_append}pred_L1bn_{self.model_path}_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                self.print_this_image(attacked_image[image_no],
                                      f"{name}/{str(step).zfill(5)}_{image_no}_{filename_append}tamper_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                self.print_this_image(masks_GT[image_no],
                                      f"{name}/{str(step).zfill(5)}_{image_no}_gt.png")
                self.print_this_image(modified_input[image_no],
                                      f"{name}/{str(step).zfill(5)}_{image_no}_ori.png")
                print("tampering localization saved at:{}".format(f"{name}/{str(step).zfill(5)}_{image_no}"))

        return logs, (pred_resfcn), False