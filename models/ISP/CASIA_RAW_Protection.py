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


class CASIA_RAW_Protection(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """
            this file is mode 9

        """
        super(CASIA_RAW_Protection, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        ### mode=8: InvISP to bi-directionally convert RGB and RAW,
        self.network_list = ["generator", 'KD_JPEG','discriminator_mask']
        self.save_network_list = ['KD_JPEG','discriminator_mask']
        self.training_network_list = ['KD_JPEG','discriminator_mask']

        ### ISP networks
        # self.netG = self.define_UNet_as_ISP()
        # self.load_model_wrapper(folder_name='original_ISP_folder', model_name='load_origin_ISP_models',
        #                         network_lists=['netG'], strict=True)

        self.generator = self.define_invISP()
        # self.qf_predict_network = self.define_UNet_as_ISP()
        self.load_model_wrapper(folder_name='ISP_folder', model_name='load_ISP_models',
                                network_lists=["generator"], strict=True)

        ### RAW2RAW network
        self.define_RAW2RAW_network(n_channels=3)
        self.load_model_wrapper(folder_name='protection_folder', model_name='load_RAW_models',
                                network_lists=['KD_JPEG'])

        ### detector
        print(f"using {self.opt['finetune_detector_name']} as discriminator_mask.")
        self.discriminator_mask = self.define_detector(opt_name='finetune_detector_name')
        self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
                                network_lists=['discriminator_mask'])

    def RAW_protection_on_CASIA(self, step=None):
        ####################  Image Manipulation Detection Network (Downstream task)  ##################
        ### mantranet: localizer mvssnet: netG resfcn: discriminator

        #### SYMBOL FOR NOTIFYING THE OUTER VAL LOADER #######
        did_val = False
        if step is not None:
            self.global_step = step

        logs, debug_logs = {}, []
        # self.real_H = self.clamp_with_grad(self.real_H)
        lr = self.get_current_learning_rate()
        logs['lr'] = lr


        # self.real_H, self.canny_image

        with torch.enable_grad():

            self.generator.eval()
            # self.qf_predict_network.train()
            self.KD_JPEG.train()
            self.discriminator_mask.train()
            gt_rgb = self.real_H
            masks_GT = self.canny_image

            ####### InvISP revert image into RAW ########################

            raw_invisp, _ = self.generator(self.real_H, rev=True)
            raw_invisp = self.clamp_with_grad(raw_invisp)

            modified_input = self.real_H + self.KD_JPEG(self.real_H)

            RAW_L1 = self.l1_loss(input=modified_input, target=gt_rgb)
            RAW_SSIM = - self.ssim_loss(modified_input, gt_rgb)
            # ISP_percept, ISP_style = self.perceptual_loss(modified_input, gt_rgb,
            #                                                   with_gram=True)

            modified_input = self.clamp_with_grad(modified_input)

            RAW_PSNR = self.psnr(self.postprocess(modified_input), self.postprocess(gt_rgb)).item()
            logs['RAW_PSNR'] = RAW_PSNR
            logs['RAW_L1'] = RAW_L1.item()

            ############################    attack layer  ######################################################
            attacked_image = modified_input*(1-masks_GT)+gt_rgb*masks_GT

            index_for_postprocessing = self.global_step
            quality_idx = self.get_quality_idx_by_iteration(index=index_for_postprocessing)
            ## settings for attack
            kernel = random.choice([3, 5, 7])  # 3,5,7
            resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
                            int(self.random_float(0.5, 2) * self.width_height))

            skip_robust = np.random.rand() > self.opt['skip_attack_probability']
            if not skip_robust and self.opt['consider_robost']:

                attacked_image, attacked_real_jpeg_simulate, _ = self.benign_attacks(attacked_forward=attacked_image,
                                                                                     quality_idx=quality_idx,
                                                                                     index=index_for_postprocessing,
                                                                                     kernel_size=kernel,
                                                                                     resize_ratio=resize_ratio
                                                                                     )
            else:
                attacked_image = attacked_image



            ###################    Image Manipulation Detection Network (Downstream task)   ####################

            ### UPDATE discriminator_mask AND LATER AFFECT THE MOMENTUM LOCALIZER
            if "discriminator_mask" in self.training_network_list:
                pred_resfcn, CE_resfcn = self.detector_predict(model=self.discriminator_mask,
                                                               attacked_image=attacked_image.detach().contiguous(),
                                                               opt_name='finetune_detector_name',
                                                               masks_GT=masks_GT)

                logs['CE'] = CE_resfcn.item()
                logs['CE_ema'] = CE_resfcn.item()

                CE_resfcn.backward()

                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
                self.optimizer_discriminator_mask.step()
                self.optimizer_discriminator_mask.zero_grad()

            if "KD_JPEG" in self.training_network_list:
                pred_resfcn, CE_resfcn = self.detector_predict(model=self.discriminator_mask,
                                                               attacked_image=attacked_image,
                                                               opt_name='finetune_detector_name',
                                                               masks_GT=masks_GT)

                logs['CE_ema'] = CE_resfcn.item()

                loss = 0
                loss_l1 = self.opt['L1_hyper_param'] * RAW_L1
                loss += loss_l1
                loss_ssim = self.opt['ssim_hyper_param'] * RAW_SSIM
                loss += loss_ssim
                hyper_param = self.exponential_weight_for_backward(value=RAW_PSNR, exp=2)
                loss += hyper_param * CE_resfcn
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


        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(gt_rgb),
                self.postprocess(raw_invisp),
                self.postprocess(modified_input),
                self.postprocess(10 * torch.abs(modified_input - gt_rgb)),
                self.postprocess(pred_resfcn),
                self.postprocess(masks_GT),
                # self.postprocess(rendered_rgb_invISP),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/isp_images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (self.opt['model_save_period']-1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)

        self.global_step = self.global_step + 1

        # print(logs)
        # print(debug_logs)
        return logs, debug_logs, did_val


    def ISP_mixing_during_training(self, *, modified_raw, modified_raw_one_dim, input_raw_one_dim, stored_lists, file_name, gt_rgb, camera_name=None):
        stored_image_generator, stored_image_qf_predict, stored_image_netG = stored_lists
        # if self.global_step % 3 == 0:
        isp_model_0, isp_model_1 = self.generator, self.qf_predict_network
        stored_list_0, stored_list_1 = stored_image_generator, stored_image_qf_predict
        # elif self.global_step % 3 == 1:
        #     isp_model_0, isp_model_1 = self.netG, self.generator
        #     stored_list_0, stored_list_1 = stored_image_netG, stored_image_generator
        # else: #if self.global_step % 3 == 2:
        #     isp_model_0, isp_model_1 = "pipeline", self.qf_predict_network
        #     stored_list_0, stored_list_1 = None, stored_image_qf_predict
        # elif self.global_step % 6 == 3:
        #     isp_model_0, isp_model_1 = self.netG, self.qf_predict_network
        #     stored_list_0, stored_list_1 = stored_image_netG, stored_image_qf_predict
        # elif self.global_step % 6 == 4:
        #     isp_model_0, isp_model_1 = "pipeline", self.netG
        #     stored_list_0, stored_list_1 = None, stored_image_netG
        # else: #if self.global_step % 6 == 5:
        #     isp_model_0, isp_model_1 = self.netG, self.generator
        #     stored_list_0, stored_list_1 = stored_image_netG, stored_image_generator


        loss_terms = 0
        ### first
        if isinstance(isp_model_0, str):
            modified_input_0 = self.pipeline_ISP_gathering(modified_raw_one_dim=modified_raw_one_dim,
                                                           file_name=file_name, gt_rgb=gt_rgb, camera_name=camera_name)
            tamper_source_0 = self.pipeline_ISP_gathering(modified_raw_one_dim=input_raw_one_dim,
                                                           file_name=file_name, gt_rgb=gt_rgb, camera_name=camera_name)
            ISP_L1_0, ISP_SSIM_0 = 0,0
        else:
            tamper_source_0 = stored_list_0
            modified_input_0, ISP_L1_0, ISP_SSIM_0 = self.differentiable_ISP_gathering(
                model=isp_model_0,modified_raw=modified_raw,tamper_source=tamper_source_0)
            loss_terms += 1

        ### second
        tamper_source_1 = stored_list_1
        modified_input_1, ISP_L1_1, ISP_SSIM_1 = self.differentiable_ISP_gathering(
            model=isp_model_1, modified_raw=modified_raw, tamper_source=tamper_source_1)
        loss_terms += 1

        ISP_L1, ISP_SSIM = (ISP_L1_0+ISP_L1_1)/loss_terms,  (ISP_SSIM_0+ISP_SSIM_1)/loss_terms

        ##################   doing mixup on the images   ###############################################
        ### note: our goal is that the rendered rgb by the protected RAW should be close to that rendered by unprotected RAW
        ### thus, we are not let the ISP network approaching the ground-truth RGB.
        # skip_the_second = np.random.rand() > 0.8
        alpha_0 = np.random.rand()
        alpha_1 = 1 - alpha_0

        modified_input = alpha_0 * modified_input_0
        modified_input += alpha_1 * modified_input_1
        tamper_source = alpha_0 * tamper_source_0
        tamper_source += alpha_1 * tamper_source_1
        tamper_source = tamper_source.detach()

        # ISP_L1_sum = self.l1_loss(input=modified_input, target=tamper_source)
        # ISP_SSIM_sum = - self.ssim_loss(modified_input, tamper_source)

        ### collect the protected images ###
        modified_input = self.clamp_with_grad(modified_input)
        tamper_source = self.clamp_with_grad(tamper_source)

        ## return format: modified_input, tamper_source, semi_images, semi_sources, semi_losses
        return modified_input, tamper_source, (modified_input_0, modified_input_1), (tamper_source_0, tamper_source_1), (ISP_L1, ISP_SSIM)
