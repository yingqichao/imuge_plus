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


class Invert_RGB_to_RAW(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """
            this file is mode 8

        """
        super(Invert_RGB_to_RAW, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        ### mode=8: InvISP to bi-directionally convert RGB and RAW,
        self.network_list = ["generator", "qf_predict_network"]
        self.save_network_list, self.training_network_list = self.network_list, self.network_list

        ### ISP networks
        self.generator = self.define_invISP()
        self.qf_predict_network = self.define_UNet_as_ISP()
        self.load_model_wrapper(folder_name='ISP_folder', model_name='load_ISP_models',
                                network_lists=self.network_list, strict=False)

    def invert_RGB_to_RAW(self, step=None):
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


        sum_batch_size = self.real_H.shape[0]

        ### camera_white_balance SIZE (B,3)
        camera_white_balance = self.camera_white_balance
        file_name = self.file_name
        ### bayer_pattern sized (B,1) ranging from [0,3]
        bayer_pattern = self.bayer_pattern

        input_raw_one_dim = self.real_H
        gt_rgb = self.label
        input_raw = self.visualize_raw(input_raw_one_dim, bayer_pattern=bayer_pattern,
                                       white_balance=camera_white_balance, eval=not self.opt['train_isp_networks'])

        batch_size, num_channels, height_width, _ = input_raw.shape
        # input_raw = self.clamp_with_grad(input_raw)

        with torch.enable_grad():

            self.generator.train()
            self.qf_predict_network.train()

            ####### UNetDiscriminator ##############
            modified_input_qf_predict = self.qf_predict_network(gt_rgb)
            CYCLE_loss = self.l1_loss(input=modified_input_qf_predict, target=input_raw)
            modified_input_qf_predict_detach = self.clamp_with_grad(modified_input_qf_predict)
            CYCLE_PSNR = self.psnr(self.postprocess(modified_input_qf_predict_detach), self.postprocess(input_raw)).item()
            logs['UNETREV_PSNR'] = CYCLE_PSNR
            logs['UNETREV_L1'] = CYCLE_loss.item()
            CYCLE_loss.backward()
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()

            ####### InvISP ########################
            modified_input_generator = self.generator(input_raw)
            ISP_forward = self.l1_loss(input=modified_input_generator, target=gt_rgb)
            modified_input_generator = self.clamp_with_grad(modified_input_generator)
            rev_RAW, _ = self.generator(modified_input_generator, rev=True)
            ISP_backward = self.l1_loss(input=rev_RAW, target=input_raw)
            ISP_loss = ISP_backward + ISP_forward
            modified_input_generator_detach = modified_input_generator.detach()
            ISP_PSNR = self.psnr(self.postprocess(modified_input_generator_detach), self.postprocess(gt_rgb)).item()
            logs['ISP_PSNR'] = ISP_PSNR
            logs['ISP_L1'] = ISP_forward.item()
            ISP_REV_PSNR = self.psnr(self.postprocess(rev_RAW), self.postprocess(input_raw)).item()
            logs['ISPREV_PSNR'] = ISP_REV_PSNR
            logs['ISPREV_L1'] = ISP_backward.item()

            ISP_loss.backward()
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.generator.parameters(), 1)
            self.optimizer_generator.step()
            self.optimizer_generator.zero_grad()

        ### emptying cache to save memory ###
        # torch.cuda.empty_cache()

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(gt_rgb),
                self.postprocess(modified_input_generator_detach),
                self.postprocess(input_raw),
                self.postprocess(rev_RAW),
                self.postprocess(modified_input_qf_predict_detach),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/isp_images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            print(f'Bayer: {bayer_pattern}. Saving sample {name}')
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
