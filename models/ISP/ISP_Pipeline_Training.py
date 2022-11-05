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


class ISP_Pipeline_Training(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """
            this file is mode 2

        """
        super(ISP_Pipeline_Training, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        ### mode=2: regular training (our network design), including ISP, RAW2RAW and localization (train)
        self.network_list = self.default_ISP_networks + self.default_RAW_to_RAW_networks + ['discriminator_mask']
        self.save_network_list, self.training_network_list = [], []
        if self.opt["train_isp_networks"]:
            self.save_network_list += self.default_ISP_networks
            self.training_network_list += self.default_ISP_networks
        if self.opt["train_RAW2RAW"]:
            self.save_network_list += self.default_RAW_to_RAW_networks
            self.training_network_list += self.default_RAW_to_RAW_networks
        if self.opt["train_detector"]:
            self.save_network_list += ['discriminator_mask']
            self.training_network_list += ['discriminator_mask']

        ### ISP networks
        self.define_ISP_network_training()
        self.load_model_wrapper(folder_name='ISP_folder', model_name='load_ISP_models',
                                network_lists=self.default_ISP_networks, strict=False)
        ### RAW2RAW network
        self.define_RAW2RAW_network()
        self.load_model_wrapper(folder_name='protection_folder',model_name='load_RAW_models',
                                network_lists=self.default_RAW_to_RAW_networks)
        ### detector networks: using hybrid model
        print("using CATNET as discriminator_mask.")

        self.discriminator_mask = self.define_CATNET() #self.define_my_own_elastic_as_detector()
        self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
                                network_lists=['discriminator_mask'])

        ## inpainting model
        self.define_inpainting_edgeconnect()
        self.define_inpainting_ZITS()
        self.define_inpainting_lama()

    def optimize_parameters_main(self, step=None):
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

        #### THESE VARIABLES STORE THE RENDERED RGB FROM CLEAN RAW ######
        stored_image_netG = None
        stored_image_generator = None
        stored_image_qf_predict = None
        collected_protected_image = None
        inpainted_image = None

        if not (self.previous_images is None or self.previous_previous_images is None):
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

            if self.opt['include_isp_inference']:
                with torch.enable_grad() if self.opt['train_isp_networks'] else torch.no_grad():
                    ### HINT FOR WHICH IS WHICH
                    ### generator: INV ISP
                    ### netG: HWMNET (BEFORE MODIFICATION)
                    ### qf_predict_network: UNETDISCRIMINATOR

                    if self.opt['train_isp_networks']:
                        self.generator.train()
                        self.netG.train()
                        self.qf_predict_network.train()
                    else:
                        self.generator.eval()
                        self.netG.eval()
                        self.qf_predict_network.eval()

                    #######################    Image ISP training    ###############################################
                    ### we first train several nn-based ISP networks BEFORE TRAINING THE PIPELINE

                    ####### UNetDiscriminator ##############
                    modified_input_qf_predict, CYCLE_loss = self.ISP_image_generation_general(network=self.qf_predict_network,
                                                                                        input_raw=input_raw.detach().contiguous(),
                                                                                        target=gt_rgb)

                    modified_input_qf_predict_detach = self.clamp_with_grad(modified_input_qf_predict.detach())
                    CYCLE_PSNR = self.psnr(self.postprocess(modified_input_qf_predict_detach),  self.postprocess(gt_rgb)).item()
                    logs['CYCLE_PSNR'] = CYCLE_PSNR
                    logs['CYCLE_L1'] = CYCLE_loss.item()
                    # del modified_input_qf_predict
                    # torch.cuda.empty_cache()
                    stored_image_qf_predict = modified_input_qf_predict_detach if stored_image_qf_predict is None else \
                        torch.cat([stored_image_qf_predict, modified_input_qf_predict_detach], dim=0)

                    # self.optimizer_generator.zero_grad()
                    if self.opt['train_isp_networks']:
                        CYCLE_loss.backward()

                        if self.train_opt['gradient_clipping']:
                            nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
                        self.optimizer_qf.step()
                        self.optimizer_qf.zero_grad()


                    # #### HWMNET ####
                    # modified_input_netG, THIRD_loss = self.ISP_image_generation_general(network=self.netG,
                    #                                                                        input_raw=input_raw.detach().contiguous(),
                    #                                                                        target=gt_rgb)
                    #
                    # modified_input_netG_detach = self.clamp_with_grad(modified_input_netG.detach())
                    # PIPE_PSNR = self.psnr(self.postprocess(modified_input_netG_detach),self.postprocess(gt_rgb)).item()
                    # logs['PIPE_PSNR'] = PIPE_PSNR
                    # logs['PIPE_L1'] = THIRD_loss.item()
                    # ## STORE THE RESULT FOR LATER USE
                    # stored_image_netG = modified_input_netG_detach if stored_image_netG is None else \
                    #     torch.cat([stored_image_netG, modified_input_netG_detach], dim=0)
                    #
                    # if self.opt['train_isp_networks']:
                    #     # self.optimizer_generator.zero_grad()
                    #     THIRD_loss.backward()
                    #
                    #     if self.train_opt['gradient_clipping']:
                    #         nn.utils.clip_grad_norm_(self.netG.parameters(), 1)
                    #     self.optimizer_G.step()
                    #     self.optimizer_G.zero_grad()

                    #### InvISP #####
                    modified_input_generator, ISP_loss = self.ISP_image_generation_general(network=self.generator,
                                                                                           input_raw=input_raw.detach().contiguous(),
                                                                                           target=gt_rgb)

                    modified_input_generator_detach = modified_input_generator.detach()
                    ISP_PSNR = self.psnr(self.postprocess(modified_input_generator_detach), self.postprocess(gt_rgb)).item()
                    logs['ISP_PSNR'] = ISP_PSNR
                    logs['ISP_L1'] = ISP_loss.item()
                    stored_image_generator = modified_input_generator_detach if stored_image_generator is None else \
                        torch.cat([stored_image_generator, modified_input_generator_detach], dim=0)

                    if self.opt['train_isp_networks']:
                        ### Grad Accumulation (which we have abandoned)
                        # self.optimizer_generator.zero_grad()
                        ISP_loss.backward()

                        if self.train_opt['gradient_clipping']:
                            nn.utils.clip_grad_norm_(self.generator.parameters(), 1)
                        self.optimizer_generator.step()
                        self.optimizer_generator.zero_grad()

                ### emptying cache to save memory ###
                # torch.cuda.empty_cache()


                if self.opt['train_isp_networks'] and (self.global_step % 200 == 3 or self.global_step <= 10):
                    images = stitch_images(
                        self.postprocess(input_raw),
                        self.postprocess(modified_input_generator_detach),
                        self.postprocess(modified_input_qf_predict_detach),
                        # self.postprocess(modified_input_netG_detach),
                        self.postprocess(gt_rgb),
                        img_per_row=1
                    )

                    name = f"{self.out_space_storage}/isp_images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                           f"_{str(self.rank)}.png"
                    print(f'Bayer: {bayer_pattern}. Saving sample {name}')
                    images.save(name)

            if self.opt['train_full_pipeline']:
                ### HINT FOR WHICH IS WHICH
                ### KD_JPEG: RAW2RAW, WHICH IS A MODIFIED HWMNET WITH STYLE CONDITION
                ### discriminator_mask: HWMNET WITH SUBTASK
                ### discriminator: MOVING AVERAGE OF discriminator_mask
                self.KD_JPEG.train() if "KD_JPEG" in self.training_network_list else self.KD_JPEG.eval()
                self.generator.eval()
                self.netG.eval()
                self.qf_predict_network.eval()
                self.discriminator_mask.train() if "discriminator_mask" in self.training_network_list else self.discriminator_mask.eval()
                # self.discriminator.train() if "discriminator" in self.training_network_list else self.discriminator.eval()
                # self.localizer.train() if "localizer" in self.training_network_list else self.localizer.eval()


                with torch.enable_grad():

                    ### RAW PROTECTION ###
                    if self.task_name == "my_own_elastic":
                        modified_raw_one_dim = self.RAW_protection_by_my_own_elastic(input_raw_one_dim=input_raw_one_dim)

                    else:
                        modified_raw_one_dim = self.KD_JPEG(input_raw_one_dim)

                    modified_raw = self.visualize_raw(modified_raw_one_dim, bayer_pattern=bayer_pattern, white_balance=camera_white_balance)
                    RAW_L1 = self.l1_loss(input=modified_raw, target=input_raw)

                    modified_raw = self.clamp_with_grad(modified_raw)

                    RAW_PSNR = self.psnr(self.postprocess(modified_raw), self.postprocess(input_raw)).item()
                    logs['RAW_PSNR'] = RAW_PSNR
                    logs['RAW_L1'] = RAW_L1.item()

                    ########################    RAW2RGB pipelines   ################################################
                    ### note: our goal is that the rendered rgb by the protected RAW should be close to that rendered by unprotected RAW
                    ### thus, we are not let the ISP network approaching the ground-truth RGB.

                    ### model selection，shuffle the gts to enable color control

                    modified_input, tamper_source, semi_images, semi_sources, semi_losses = self.ISP_mixing_during_training(
                        modified_raw=modified_raw,
                        stored_lists=(stored_image_generator, stored_image_qf_predict, stored_image_netG),
                        modified_raw_one_dim=modified_raw_one_dim,
                        input_raw_one_dim=input_raw_one_dim,
                        file_name=file_name, gt_rgb=gt_rgb,
                        camera_name=self.camera_name
                    )
                    modified_input_0, modified_input_1 = semi_images
                    tamper_source_0, tamper_source_1 = semi_sources
                    ISP_L1, ISP_SSIM = semi_losses


                    PSNR_DIFF = self.psnr(self.postprocess(modified_input), self.postprocess(tamper_source)).item()
                    ISP_PSNR = self.psnr(self.postprocess(modified_input), self.postprocess(gt_rgb)).item()
                    logs['PSNR_DIFF'] = PSNR_DIFF
                    logs['ISP_PSNR_NOW'] = ISP_PSNR

                    collected_protected_image = tamper_source

                    #######################   attack layer   #######################################################
                    attacked_image, attacked_adjusted, attacked_forward, masks, masks_GT = self.standard_attack_layer(
                        modified_input=modified_input, gt_rgb=gt_rgb, logs=logs
                    )

                    # ERROR = attacked_image-attacked_forward
                    error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(attacked_adjusted)).item() #self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
                    logs['ERROR'] = error_l1

                    #####################   Image Manipulation Detection Network (Downstream task)   ###############
                    ### mantranet: localizer mvssnet: netG resfcn: discriminator
                    # _, pred_mvss = self.netG(attacked_image)
                    # CE_MVSS = self.bce_with_logit_loss(pred_mvss, masks_GT)
                    # logs['CE_MVSS'] = CE_MVSS.item()
                    # pred_mantra = self.localizer(attacked_image)
                    # CE_mantra = self.bce_with_logit_loss(pred_mantra, masks_GT)
                    # logs['CE_mantra'] = CE_mantra.item()
                    ### why contiguous? https://discuss.pytorch.org/t/runtimeerror-set-sizes-and-strides-is-not-allowed-on-a-tensor-created-from-data-or-detach/116910/10


                    ### UPDATE discriminator_mask AND LATER AFFECT THE MOMENTUM LOCALIZER
                    if "discriminator_mask" in self.training_network_list:

                        CE_resfcn, l1_resfcn, pred_resfcn = self.detecting_forgery(
                                               attacked_image=attacked_image.detach().contiguous(),
                                               masks_GT=masks_GT, logs=logs)

                        CE_loss = CE_resfcn + l1_resfcn
                        logs['CE'] = CE_resfcn.item()
                        logs['CE_ema'] = CE_resfcn.item()
                        # logs['Mean'] = l1_mean.item()
                        # logs['Std'] = l1_std.item()
                        CE_loss.backward()
                        if self.train_opt['gradient_clipping']:
                            nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
                        self.optimizer_discriminator_mask.step()
                        self.optimizer_discriminator_mask.zero_grad()


                    ### USING THE MOMENTUM LOCALIZER TO TRAIN THE PIPELINE
                    if "KD_JPEG" in self.training_network_list:

                        CE_resfcn, l1_resfcn, pred_resfcn = self.detecting_forgery(
                            attacked_image=attacked_image,
                            masks_GT=masks_GT, logs=logs)

                        CE_loss = CE_resfcn
                        logs['CE_ema'] = CE_resfcn.item()

                        loss = 0
                        loss_l1 = self.opt['L1_hyper_param'] * ISP_L1
                        loss += loss_l1
                        hyper_param_raw = self.opt['RAW_L1_hyper_param'] # if (ISP_PSNR < self.opt['psnr_thresh']) else self.opt['RAW_L1_hyper_param']/5
                        loss += hyper_param_raw * RAW_L1
                        loss_ssim = self.opt['ssim_hyper_param'] * ISP_SSIM
                        loss += loss_ssim
                        # hyper_param_percept = self.opt['perceptual_hyper_param'] # if (ISP_PSNR < self.opt['psnr_thresh']) else self.opt['perceptual_hyper_param'] / 4
                        # loss_percept = hyper_param_percept * (ISP_percept_0+ISP_percept_1)/2
                        # loss += loss_percept
                        # loss_style = self.opt['style_hyper_param'] * (ISP_style_0 +ISP_style_1) / 2
                        # loss += loss_style
                        # hyper_param = self.opt['CE_hyper_param'] if (ISP_PSNR>=self.opt['psnr_thresh']) else self.opt['CE_hyper_param']/10
                        hyper_param = self.exponential_weight_for_backward(value=ISP_PSNR, exp=2)
                        loss += hyper_param * CE_loss  # (CE_MVSS+CE_mantra+CE_resfcn)/3

                        logs['ISP_SSIM_NOW'] = -loss_ssim.item()
                        # logs['Percept'] = loss_percept.item()
                        # logs['Style'] = loss_style.item()
                        logs['Gray'] = loss_l1.item()
                        logs['loss'] = loss.item()


                        ##### Grad Accumulation (not used any more)
                        loss.backward()

                        if self.train_opt['gradient_clipping']:
                            nn.utils.clip_grad_norm_(self.KD_JPEG.parameters(), 1)

                        self.optimizer_KD_JPEG.step()

                        self.optimizer_KD_JPEG.zero_grad()
                        self.optimizer_G.zero_grad()
                        self.optimizer_discriminator_mask.zero_grad()
                        self.optimizer_generator.zero_grad()
                        self.optimizer_qf.zero_grad()
                        # self.optimizer_localizer.zero_grad()
                        # self.optimizer_discriminator.zero_grad()

                ### update and track history losses
                self.update_history_losses(index=self.global_step, PSNR=PSNR_DIFF,
                                           loss=loss.item(),
                                           loss_CE=CE_loss.item(), PSNR_attack=error_l1)

                #########################    printing the images   #################################################
                anomalies = False  # CE_recall.item()>0.5
                if anomalies or self.global_step % 200 == 3 or self.global_step <= 10:
                    images = stitch_images(
                        self.postprocess(input_raw),
                        ### RAW2RAW
                        self.postprocess(modified_raw),
                        self.postprocess(10 * torch.abs(modified_raw - input_raw)),
                        ### rendered images and protected images
                        self.postprocess(modified_input_0),
                        self.postprocess(tamper_source_0),
                        self.postprocess(10 * torch.abs(modified_input_0 - tamper_source_0)),
                        self.postprocess(modified_input_1),
                        self.postprocess(tamper_source_1),
                        self.postprocess(10 * torch.abs(modified_input_1 - tamper_source_1)),
                        self.postprocess(modified_input),

                        self.postprocess(gt_rgb),

                        ### tampering and benign attack
                        self.postprocess(attacked_forward),
                        self.postprocess(attacked_adjusted),
                        self.postprocess(attacked_image),
                        self.postprocess(10 * torch.abs(attacked_forward - attacked_adjusted)),
                        self.postprocess(10 * torch.abs(attacked_adjusted - attacked_image)),
                        ### tampering detection
                        # self.postprocess(attacked_cannied),
                        self.postprocess(masks_GT),

                        self.postprocess(pred_resfcn),
                        # self.postprocess(torch.sigmoid(post_resfcn)),

                        img_per_row=1
                    )

                    name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                               f"_{str(self.rank)}.png"
                    print('\nsaving sample ' + name)
                    images.save(name)

                ### doing ema average
                # if self.begin_using_momentum:
                #     print("Moving average...")
                #     self._momentum_update_key_encoder()

                ### inference single image for testing
                # if self.global_step % 199 == 3:
                #     did_val = True
                #     self.inference_single_image()#input_raw_one_dim=input_raw_one_dim, input_raw=input_raw, gt_rgb=gt_rgb,
                #                                 # camera_white_balance=camera_white_balance, file_name=file_name,
                #                                 # camera_name=camera_name, bayer_pattern=bayer_pattern)


        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (self.opt['model_save_period']-1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)
        if self.real_H is not None:
            ### update the tampering source
            if self.previous_images is not None:
                ### previous_previous_images is for tampering-based data augmentation
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.label
            ### update the tampering source with pattern
            # if self.previous_protected is not None:
            #     self.previous_previous_protected = self.previous_protected.clone().detach()
            self.previous_protected = collected_protected_image

        self.global_step = self.global_step + 1

        # print(logs)
        # print(debug_logs)
        return logs, debug_logs, did_val

    def detecting_forgery(self, *, attacked_image, masks_GT, logs):
        _, pred_resfcn = self.CAT_predict(model=self.discriminator_mask,attacked_image=attacked_image)
        CE_resfcn = self.bce_loss(pred_resfcn, masks_GT)
        l1_resfcn = 0

        # if self.global_step % self.amount_of_detectors in self.opt['detector_using_MPF_indices']: #"MPF" in self.opt['which_model_for_detector']:
        #     ### get canny of attacked image
        #     attacked_cannied, _ = self.get_canny(attacked_image, masks_GT)
        #     predicted_masks = self.discriminator_mask(attacked_image, attacked_cannied)
        #     pred_resfcn, post_resfcn = predicted_masks
        #     CE_resfcn = self.bce_loss(torch.sigmoid(pred_resfcn), masks_GT)
        #     l1_resfcn = self.l2_loss(torch.sigmoid(post_resfcn), masks_GT)
        #     logs['CEL1'] = l1_resfcn.item()
        #     logs['l1_ema'] = l1_resfcn.item()
        #     pred_resfcn = torch.sigmoid(pred_resfcn)
        # elif self.global_step % self.amount_of_detectors in self.opt['detector_using_MVSS_indices']: #"MVSS" in self.opt['which_model_for_detector']:
        #     predicted_masks = self.discriminator(attacked_image)
        #     _, pred_resfcn = predicted_masks
        #     CE_resfcn = self.bce_loss(torch.sigmoid(pred_resfcn), masks_GT)
        #     l1_resfcn = 0
        #     pred_resfcn = torch.sigmoid(pred_resfcn)
        # elif self.global_step % self.amount_of_detectors in self.opt['detector_using_OSN_indices']: # "OSN" in self.opt['which_model_for_detector']:
        #     pred_resfcn = self.localizer(attacked_image)
        #     CE_resfcn = self.bce_loss(pred_resfcn, masks_GT)
        #     l1_resfcn = 0
        # else:
        #     raise NotImplementedError("Detector名字不对，请检查！")

        return CE_resfcn, l1_resfcn, pred_resfcn


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
