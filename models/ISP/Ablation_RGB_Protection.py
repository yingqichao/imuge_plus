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


class Ablation_RGB_Protection(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """

            this file is mode 3

        """
        super(Ablation_RGB_Protection, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        ### regular training for ablation (RGB protection)
        self.network_list = ["KD_JPEG", "discriminator_mask"]
        self.save_network_list = ["KD_JPEG", "discriminator_mask"]
        self.training_network_list = ["discriminator_mask"]

        ### RGB protection
        print("using my_own_elastic as RGB protection.")
        self.define_RAW2RAW_network()
        self.load_model_wrapper(folder_name='protection_folder', model_name='load_RAW_models',
                                network_lists=self.default_RAW_to_RAW_networks)
        ### detector
        self.discriminator_mask = self.define_OSN_as_detector()
        self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
                                network_lists=['discriminator_mask'])

        ## inpainting model
        self.define_inpainting_edgeconnect()
        self.define_inpainting_ZITS()
        self.define_inpainting_lama()

    def optimize_parameters_ablation_on_RAW(self, step=None):
        #### SYMBOL FOR NOTIFYING THE OUTER VAL LOADER #######
        did_val = False
        if step is not None:
            self.global_step = step

        logs, debug_logs = {}, []
        # self.real_H = self.clamp_with_grad(self.real_H)
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        collected_protected_image = None
        if not (self.previous_images is None or self.previous_previous_images is None):

            self.KD_JPEG.train()
            self.discriminator_mask.train()

            gt_rgb = self.real_H

            batch_size, num_channels, height_width, _ = gt_rgb.shape

            with torch.enable_grad():
                ##################    Generation of protected RAW   ###############################################

                ### RGB PROTECTION ###
                modified_input = self.baseline_generate_protected_rgb(gt_rgb=gt_rgb)

                RAW_L1 = self.l1_loss(input=modified_input, target=gt_rgb)
                RAW_SSIM = - self.ssim_loss(modified_input, gt_rgb)
                # ISP_percept, ISP_style = self.perceptual_loss(modified_input, gt_rgb,
                #                                                   with_gram=True)

                modified_input = self.clamp_with_grad(modified_input)

                RAW_PSNR = self.psnr(self.postprocess(modified_input), self.postprocess(gt_rgb)).item()
                logs['RAW_PSNR'] = RAW_PSNR
                logs['RAW_L1'] = RAW_L1.item()
                # logs['Percept'] = ISP_percept.item()


                collected_protected_image = modified_input.detach() if collected_protected_image is None else \
                    torch.cat([collected_protected_image, modified_input.detach()], dim=0)

                ############################    attack layer  ######################################################
                attacked_image, attacked_adjusted, attacked_forward, masks, masks_GT = self.standard_attack_layer(
                    modified_input=modified_input, gt_rgb=gt_rgb, logs=logs
                )

                # ERROR = attacked_image-attacked_forward
                error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(attacked_forward)).item() #self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
                logs['ERROR'] = error_l1

                ###################    Image Manipulation Detection Network (Downstream task)   ####################

                ### UPDATE discriminator_mask AND LATER AFFECT THE MOMENTUM LOCALIZER
                if "discriminator_mask" in self.training_network_list:
                    CE_resfcn, l1_resfcn, pred_resfcn = self.detecting_forgery(
                        attacked_image=attacked_image.detach().contiguous(),
                        masks_GT=masks_GT, logs=logs)

                    CE_loss = CE_resfcn + l1_resfcn
                    logs['CE'] = CE_resfcn.item()
                    logs['CE_ema'] = CE_resfcn.item()

                    CE_loss.backward()

                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
                    self.optimizer_discriminator_mask.step()
                    self.optimizer_discriminator_mask.zero_grad()

                if "KD_JPEG" in self.training_network_list:
                    CE_resfcn, l1_resfcn, pred_resfcn = self.detecting_forgery(
                        attacked_image=attacked_image,
                        masks_GT=masks_GT, logs=logs)

                    CE_loss = CE_resfcn
                    logs['CE_ema'] = CE_resfcn.item()


                    loss = 0
                    loss_l1 = self.opt['L1_hyper_param'] * RAW_L1
                    loss += loss_l1
                    loss_ssim = self.opt['ssim_hyper_param'] * RAW_SSIM
                    loss += loss_ssim
                    hyper_param = self.exponential_weight_for_backward(value=RAW_PSNR, exp=2)
                    loss += hyper_param * CE_loss  # (CE_MVSS+CE_mantra+CE_resfcn)/3
                    logs['loss'] = loss.item()
                    logs['ISP_SSIM_NOW'] = -loss_ssim.item()

                    ### Grad Accumulation
                    loss.backward()
                    # self.scaler_kd_jpeg.scale(loss).backward()

                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.KD_JPEG.parameters(), 1)

                    self.optimizer_KD_JPEG.step()
                    self.optimizer_KD_JPEG.zero_grad()
                    self.optimizer_discriminator_mask.zero_grad()
                else:
                    logs['loss'] = 0


            ##### printing the images  ######
            anomalies = False  # CE_recall.item()>0.5
            if anomalies or self.global_step % 200 == 3 or self.global_step <= 10:
                images = stitch_images(

                    self.postprocess(modified_input),
                    self.postprocess(gt_rgb),
                    self.postprocess(10 * torch.abs(modified_input - gt_rgb)),
                    self.postprocess(attacked_forward),
                    self.postprocess(attacked_adjusted),
                    self.postprocess(attacked_image),
                    self.postprocess(10 * torch.abs(attacked_forward - attacked_image)),
                    ### tampering detection
                    self.postprocess(masks_GT),
                    self.postprocess(pred_resfcn),
                    img_per_row=1
                )

                name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                           f"_{str(self.rank)}.png"
                print('\nsaving sample ' + name)
                images.save(name)


        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % 1000 == 999 or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)
        if self.real_H is not None:
            ### update the tampering source
            if self.previous_images is not None:
                ### previous_previous_images is for tampering-based data augmentation
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H
            ### update the tampering source with pattern
            # if self.previous_protected is not None:
            #     self.previous_previous_protected = self.previous_protected.clone().detach()
            self.previous_protected = collected_protected_image

        self.global_step = self.global_step + 1

        # print(logs)
        # print(debug_logs)
        return logs, debug_logs, did_val

    def detecting_forgery(self, *, attacked_image, masks_GT, logs):
        pred_resfcn = self.predict_with_NO_sigmoid(model=self.discriminator_mask,attacked_image=attacked_image)
        CE_resfcn = self.bce_loss(pred_resfcn, masks_GT)
        l1_resfcn = 0

        return CE_resfcn, l1_resfcn, pred_resfcn
