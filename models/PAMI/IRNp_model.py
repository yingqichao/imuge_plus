import copy
import logging
import os

import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.nn.parallel import DistributedDataParallel

from models.networks import UNetDiscriminator, JPEGGenerator
from noise_layers import *
from utils import stitch_images
from models.base_model import BaseModel
from models.conditional_jpeg_generator import QF_predictor
from models.invertible_net import Inveritible_Decolorization_PAMI, ResBlock

logger = logging.getLogger('base')
# json_path = '/home/qichaoying/Documents/COCOdataset/annotations/incnances_val2017.json'
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

class IRNpModel(BaseModel):
    def __init__(self, opt,args):

        super(IRNpModel, self).__init__(opt,args)
        ########### CONSTANTS ###############

        self.real_H, self.real_H_path, self.previous_images, self.previous_previous_images = None, None, None, None
        self.previous_canny = None
        self.task_name = self.opt['datasets']['train']['name'] #self.train_opt['task_name']
        self.gpu_id = self.opt["gpu_ids"][0]
        ############## Nets ################################
        self.network_list = []
        if self.args.mode in {0,1,3}: # training of Imuge+
            self.network_list = ['netG', 'localizer','discriminator_mask']

            self.localizer = UNetDiscriminator() #Localizer().cuda()
            # stat(self.localizer.to(torch.device('cpu')), (3, 512, 512))
            self.localizer = self.localizer.cuda()
            self.localizer = DistributedDataParallel(self.localizer, device_ids=[torch.cuda.current_device()],
                                                     )  # find_unused_parameters=True


            self.netG = Inveritible_Decolorization_PAMI(dims_in=[[4, 64, 64]], block_num=[2, 2, 2],
                                                        subnet_constructor=ResBlock)
            # stat(self.netG.to(torch.device('cpu')),(4,512,512))
            self.netG = self.netG.cuda()
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],
                                                )  # find_unused_parameters=True
            # self.generator = Discriminator(in_channels=3, use_sigmoid=True).cuda()
            # self.generator = DistributedDataParallel(self.generator, device_ids=[torch.cuda.current_device()])
            self.discriminator_mask = UNetDiscriminator(in_channels=4, out_channels=1, residual_blocks=2) #Localizer().cuda()
            self.discriminator_mask = self.discriminator_mask.cuda()
            self.discriminator_mask = DistributedDataParallel(self.discriminator_mask,
                                                              device_ids=[torch.cuda.current_device()],
                                                              ) # find_unused_parameters=True

            self.scaler_localizer = torch.cuda.amp.GradScaler()
            self.scaler_G = torch.cuda.amp.GradScaler()
            self.scaler_discriminator_mask = torch.cuda.amp.GradScaler()
            self.scaler_generator = torch.cuda.amp.GradScaler()

        elif args.mode==2:
            self.network_list = ['KD_JPEG','generator','qf_predict']
            self.KD_JPEG_net = JPEGGenerator()
            self.KD_JPEG_net = self.KD_JPEG_net.cuda()
            self.KD_JPEG_net = DistributedDataParallel(self.KD_JPEG_net, device_ids=[torch.cuda.current_device()],
                                                       find_unused_parameters=True)
            self.generator = JPEGGenerator()
            self.generator = self.generator.cuda()
            self.generator = DistributedDataParallel(self.generator, device_ids=[torch.cuda.current_device()],
                                                     find_unused_parameters=True)

            self.qf_predict_network = QF_predictor()
            self.qf_predict_network = self.qf_predict_network.cuda()
            self.qf_predict_network = DistributedDataParallel(self.qf_predict_network, device_ids=[torch.cuda.current_device()],
                                                       find_unused_parameters=True)

            self.scaler_KD_JPEG = torch.cuda.amp.GradScaler()
            self.scaler_generator = torch.cuda.amp.GradScaler()
            self.scaler_qf_predict = torch.cuda.amp.GradScaler()

        else:
            raise NotImplementedError("从1012开始，需要指定一下读取哪些模型")

        ####################################################################################################
        # todo: Optimizers and Schedulers
        ####################################################################################################
        self.define_optimizers()

        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=118287))


        ######## init constants
        self.forward_image_buff = None

        ########## Load pre-trained ##################
        if self.args.mode in [0,1,3]:
            # good_models: '/model/Rerun_4/29999'
            self.out_space_storage = '/data/20220106_IMUGE'
            self.model_storage = '/model/Rerun_3/'
            self.model_path = str(16999) # 42999 # 8999 # newest: 14999
        else:
            self.out_space_storage = '/data/20220106_IMUGE'
            self.model_storage = '/jpeg_model/Rerun_3/'
            self.model_path = str(16999)  # 29999

        ########## WELCOME BACK!!!
        ########## THE FOLLOWING MODELS SHOULD BE KEEP INTACT!!!
        # self.model_path = '/model/welcome_back/COCO/18010'
        # self.model_path = '/model/backup/COCO/24003'
        # self.model_path = '/model/CelebA/15010'
        # self.model_path = '/model/ILSVRC/2010'
        load_models = True
        load_state = False
        if load_models:
            self.pretrain = self.out_space_storage + self.model_storage + self.model_path
            self.reload(self.pretrain, self.network_list)
            ## load states
            state_path = self.out_space_storage + self.model_storage + '{}.state'.format(self.model_path)
            if load_state:
                print('Loading training state')
                if os.path.exists(state_path):
                    self.resume_training(state_path, self.network_list)
                else:
                    print('Did not find state [{:s}] ...'.format(state_path))


    def optimize_parameters_router(self, mode, step=None):
        if mode == 0.0:
            return self.optimize_parameters(step=step)
        elif mode==1.0:
            return self.evaluate()
        elif mode==2.0:
            return self.KD_JPEG_Generator_training(step=step)
        elif mode==3.0:
            return self.generate_immunized_images_only(step=step)

    def feed_data_router(self, batch, mode):
        self.feed_data(batch, mode='train')

    def feed_data_val_router(self, batch, mode):
        self.feed_data(batch, mode='val')


    def feed_data(self, batch, mode="train"):
        img, label, canny_image = batch
        if mode=="train":
            self.real_H = img.cuda()
            self.canny_image = canny_image.cuda()
        else:
            self.real_H_val = img.cuda()
            self.canny_image_val = canny_image.cuda()

    def optimize_parameters(self, step=0, latest_values=None, train=True, eval_dir=None):
        self.netG.train()
        self.localizer.train()
        self.discriminator_mask.train()
        # self.discriminator.train()
        # self.generator.train()
        self.optimizer_discriminator_mask.zero_grad()
        self.optimizer_G.zero_grad()
        self.optimizer_localizer.zero_grad()
        # self.optimizer_generator.zero_grad()
        # self.optimizer_discriminator_mask.zero_grad()

        logs, debug_logs = {}, []

        self.real_H = self.clamp_with_grad(self.real_H)
        batch_size, num_channels, height_width, _ = self.real_H.shape
        psnr_thresh = self.opt['psnr_thresh']
        save_interval = 1000
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        modified_input = self.real_H.clone().detach()
        modified_input = self.clamp_with_grad(modified_input)
        modified_canny = self.canny_image.clone().detach()
        modified_canny = self.clamp_with_grad(modified_canny)
        # check_status = self.l1_loss(modified_input, self.real_H) + self.l1_loss(modified_canny,self.canny_image)
        # if check_status > 0:
        #     print(f"Strange input detected! {check_status} skip")
        #     return logs, debug_logs, False
        if not (self.previous_images is None or self.previous_previous_images is None):

            with torch.enable_grad():
                ## settings for attack
                kernel = random.choice([3, 5, 7])  # 3,5,7
                resize_ratio = (int(self.random_float(0.7, 1.5) * self.width_height),
                                int(self.random_float(0.7, 1.5) * self.width_height))

                if self.global_step%10 in self.opt['crop_indices']:
                        # not (self.global_step%self.amount_of_benign_attack in (self.opt['simulated_gblur_indices'] + self.opt['simulated_mblur_indices'])) and \
                        # not (self.global_step%self.amount_of_benign_attack in self.opt['simulated_strong_JPEG_indices'] and quality_idx<16):
                    logs["cropped"] = True
                    percent_range = [0, 0.25]
                    index_for_postprocessing = 7
                else:
                    logs["cropped"] = False
                    percent_range = None
                    index_for_postprocessing = self.global_step

                quality_idx = self.get_quality_idx_by_iteration(index=index_for_postprocessing)

                ## define the mask and augment
                rate_mask, masks, masks_GT = 0, None, None
                while rate_mask<0.05 or rate_mask>=0.33 or (logs["cropped"] and rate_mask>=0.2): # prevent too small or too big
                    masks, masks_GT, percent_range = self.mask_generation(modified_input=modified_input,
                                                           percent_range=percent_range, index=self.global_step)

                    ### tamper-based data augmentation
                    skip_aug = np.random.rand() > self.opt['skip_aug_probability']
                    if not skip_aug:
                        modified_input, modified_canny = self.tamper_based_augmentation(
                            modified_input=modified_input, modified_canny=modified_canny, masks=masks,
                            masks_GT=masks_GT, index=self.global_step)
                    GT_input = modified_input.clone().detach()
                    GT_canny = modified_canny.clone().detach()

                    if logs["cropped"]:
                        ### determine the crop location and mask
                        locs, cropped, masks = self.cropping_mask_generation(
                            forward_image=masks, min_rate=self.opt['cropping_lower_bound'], max_rate=1.0)
                        masks = torch.where(masks > 0.5, 1.0, 0.0)
                        masks_GT = masks[:, :1]

                    rate_mask = torch.mean(masks_GT)


                ### forward image generation
                forward_stuff = self.netG(x=torch.cat((modified_input, modified_canny), dim=1))
                forward_image, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]

                forward_null = self.clamp_with_grad(forward_null)
                l_null = self.l1_loss(forward_null, torch.zeros_like(modified_canny).cuda())
                l_percept_fw_ssim = - self.ssim_loss(forward_image, modified_input)
                l_forward = self.l1_loss(forward_image, modified_input)
                forward_image = self.clamp_with_grad(forward_image)
                psnr_forward = self.psnr(self.postprocess(modified_input), self.postprocess(forward_image)).item()

                ### cropping, skip the two blurs and heavy JPEG compression
                if logs["cropped"]:
                    # locs, cropped, forward_processed = self.cropping_mask_generation(
                    #     forward_image=forward_image,  min_rate=self.opt['cropping_lower_bound'], max_rate=1.0)
                    # h_start, h_end, w_start, w_end = locs
                    _, _, forward_processed = self.cropping_mask_generation(forward_image=forward_image, locs=locs)
                    ### adjust the ground truth
                    # _, _, masks = self.cropping_mask_generation(forward_image=masks, locs=locs)
                    _, _, GT_input = self.cropping_mask_generation(forward_image=GT_input, locs=locs)
                    _, _, GT_canny = self.cropping_mask_generation(forward_image=GT_canny, locs=locs)
                    # masks = torch.where(masks>0.5,1.0,0.0)
                    # masks_GT = masks[:,:1]
                else:
                    forward_processed = forward_image


                ### tampering the image
                attacked_forward, masks, masks_GT = self.tampering_PAMI(
                    forward_image=forward_processed, masks=masks, masks_GT=masks_GT, modified_canny=modified_canny,
                    percent_range=percent_range, index=self.global_step)

                ### image post-processings
                skip_robust = logs["cropped"] or np.random.rand() > self.opt['skip_attack_probability']
                if not skip_robust:
                    ### real-world attack: simulated + (real-simulated).detach
                    attacked_image, attacked_real_jpeg_simulate, adjusted_settings = self.benign_attacks(attacked_forward=attacked_forward,
                                                                                     quality_idx=quality_idx,
                                                                                     index=index_for_postprocessing,
                                                                                     kernel_size=kernel,
                                                                                     resize_ratio=resize_ratio
                                                                                      )
                    kernel, quality_idx, resize_ratio = adjusted_settings
                    logs['kernel'] = kernel
                    logs['quality_idx'] = quality_idx
                    # UPDATE THE GROUND-TRUTH TO EASE TRAINING
                    # GT_modified_input = modified_input
                    GT_modified_input = self.benign_attacks_without_simulation(forward_image=GT_input,
                                                                               quality_idx=quality_idx,
                                                                               index=index_for_postprocessing,
                                                                               kernel_size=kernel,
                                                                               resize_ratio=resize_ratio
                                                                               ).detach()

                    error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(attacked_forward)).item()
                    logs['ERROR'] = error_l1

                    real_and_simulate = self.psnr(self.postprocess(attacked_image), self.postprocess(attacked_real_jpeg_simulate)).item()
                    logs['SIMU'] = real_and_simulate

                else:
                    attacked_image = attacked_forward
                    # attacked_real_jpeg_simulate = attacked_forward
                    # UPDATE THE GROUND-TRUTH TO EASE TRAINING
                    GT_modified_input = GT_input
                    error_l1, real_and_simulate = 40, 40
                    logs['ERROR'], logs['SIMU'] = error_l1, real_and_simulate

                ## train localizer
                gen_attacked_train = self.localizer(attacked_image.detach())
                CE_train = self.bce_with_logit_loss(gen_attacked_train, masks_GT)
                logs['CE'] = CE_train.item()

                CE_train.backward()
                # self.scaler_localizer.scale(CE_train).backward()
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.localizer.parameters(), 1)
                self.optimizer_localizer.step()
                self.optimizer_localizer.zero_grad()


                ## IMAGE RECOVERY
                ## RECOVERY
                tampered_attacked_image = attacked_image * (1 - masks)
                # tampered_attacked_image = self.clamp_with_grad(tampered_attacked_image)
                canny_input = torch.zeros_like(modified_canny).cuda()
                reversed_stuff, _ = self.netG(torch.cat((tampered_attacked_image, canny_input), dim=1), rev=True)
                reversed_image, reversed_canny = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]

                l_backward = self.l1_loss(reversed_image, GT_modified_input)
                l_backward_local = self.l1_loss(reversed_image * masks, GT_modified_input * masks)

                l_back_canny = self.l1_loss(reversed_canny, GT_canny)
                l_back_canny_local = self.l1_loss(reversed_canny * masks, GT_canny * masks)
                l_percept_bk_ssim = - self.ssim_loss(reversed_image, GT_modified_input)
                reversed_image = self.clamp_with_grad(reversed_image)
                reversed_canny = self.clamp_with_grad(reversed_canny)
                psnr_backward = self.psnr(self.postprocess(GT_modified_input), self.postprocess(reversed_image)).item()

                # # ## training gan
                # if self.global_step % self.opt["GAN_update_period"] == self.opt["GAN_update_period"]-1:
                #     dis_input_real = torch.cat((modified_input, modified_canny), dim=1)
                #     dis_input_fake = torch.cat((reversed_image, reversed_canny), dim=1)
                #     dis_input = torch.cat([dis_input_real,dis_input_fake],dim=0)
                #     dis_batch = self.discriminator_mask(dis_input.detach())
                #     # dis_fake = self.discriminator_mask(dis_input_fake)
                #     dis_loss = self.bce_with_logit_loss(dis_batch, torch.cat([torch.ones_like(masks_GT), 1-masks_GT],dim=0))
                #     # dis_fake_loss = self.bce_with_logit_loss(dis_fake, 1 - masks_GT)
                #     # dis_loss = (dis_real_loss + dis_fake_loss) / 2
                #     logs['DIS'] = dis_loss.item()*self.opt["GAN_update_period"]
                # # modified_simulated = modified_input + self.discriminator_mask(modified_input)
                # # modified_simulated = self.clamp_with_grad(modified_simulated)
                # # dis_loss = self.l1_loss(modified_simulated, GT_modified_input)
                # # dis_psnr = self.psnr(self.postprocess(GT_modified_input), self.postprocess(modified_simulated)).item()
                # # logs['DIS'] = dis_psnr
                #     dis_loss.backward()
                #     # self.scaler_G.scale(loss).backward()
                #     if self.train_opt['gradient_clipping']:
                #         nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
                #     self.optimizer_discriminator_mask.step()
                #     self.optimizer_discriminator_mask.zero_grad()


                # ### LOCALIZATION LOSS
                gen_attacked_train = self.localizer(attacked_image)
                CE = self.bce_with_logit_loss(gen_attacked_train, masks_GT)
                masks_real = torch.where(torch.sigmoid(gen_attacked_train) > 0.5, 1.0, 0.0)
                masks_real = self.Erode_Dilate(masks_real).repeat(1, 3, 1, 1)
                logs['CE_ema'] = CE.item()

                ###################
                ### LOSSES
                loss = 0
                l_forward_sum = 0
                mask_rate = torch.mean(masks)
                logs['mask_rate'] =  mask_rate.item()
                #################
                # FORWARD LOSS

                l_forward_sum += 1.0 * l_forward
                # l_forward_sum += 0.01 * l_percept_fw_ssim
                logs['lF'] = l_forward.item()
                # weight_fw = 8 if psnr_forward<psnr_thresh else 6
                # logs.append(('CK', self.global_step%5))
                loss += self.opt['Loss_forward'] * l_forward_sum
                loss += self.opt['Null_loss'] * l_null
                logs['null'] = l_null.item()
                #################
                ### BACKWARD LOSS
                # residual
                # l_backward_ideal = self.l1_loss(reversed_image_from_residual, modified_input)
                # l_back_canny_ideal = self.l1_loss(reversed_canny_from_residual, modified_canny)
                # l_backward_local_ideal = (self.l1_loss(reversed_image_from_residual * masks, modified_input * masks))
                # loss += 0.1 * l_backward_ideal
                # loss += 0.1 * l_back_canny_ideal
                # logs.append(('RR', l_backward_local_ideal.item()))
                # ideal
                # l_backward_ideal = self.l1_loss(reversed_image_ideal, modified_input)
                # l_back_canny_ideal = self.l1_loss(reversed_canny_ideal, modified_canny)
                # l_backward_local_ideal = (self.l1_loss(reversed_image_ideal * masks, modified_input * masks))
                l_backward_sum = 0
                l_backward_sum += 0.5*l_backward
                l_backward_sum += 1.5*l_backward_local
                l_backward_sum += 1.0*l_back_canny
                # l_backward_sum += 0.01 * l_percept_bk_ssim

                # ### dis_loss
                # # if self.global_step % self.opt["GAN_update_period"] == self.opt["GAN_update_period"] - 1:
                # dis_input_fake = torch.cat((reversed_image, reversed_canny), dim=1)
                # dis_batch = self.discriminator_mask(dis_input_fake)
                # dis_loss = self.bce_with_logit_loss(dis_batch, torch.ones_like(masks_GT))
                # logs['DIS_A'] = dis_loss.item()*self.opt["GAN_update_period"]
                #
                # l_backward_sum += self.opt['GAN_weight'] * dis_loss

                # l_backward_sum += 1.0 * l_back_canny_local
                # l_backward_sum += 0.01*l_percept_bk_ssim
                logs['lB'] = l_backward.item()
                logs['local'] = l_backward_local.item()
                logs['canny'] = l_back_canny.item()

                # weight_bk = self.opt['Loss_back_psnr_less'] if psnr_forward < psnr_thresh else self.opt['Loss_back_psnr_higher']
                weight_bk = self.exponential_weight_for_backward(value=psnr_forward)
                loss += weight_bk * l_backward_sum
                #################
                ## PERCEPTUAL LOSS: loss is like [0.34], but lpips is lile [0.34,0.34,0.34]
                # l_forward_percept = self.perceptual_loss(forward_image, modified_input).squeeze()
                # l_backward_percept = self.perceptual_loss(reversed_image, modified_expand).squeeze()
                # loss += (0.01 if psnr_forward < psnr_thresh else 0.005) * l_forward_percept
                # loss += 0.01 * l_backward_percept
                # logs.append(('F_p', l_forward_percept.item()))
                # logs.append(('B_p', l_backward_percept.item()))
                ### CROSS-ENTROPY
                weight_CE = self.opt['CE_psnr_less'] if psnr_forward < psnr_thresh else self.opt['CE_psnr_less']
                # weight_GAN = 0.002
                loss += weight_CE * CE

                logs['loss'] = loss.item()



                # logs.append(('ID', self.gpu_id))
                # logs.append(('W', weight_CE))
                # logs.append(('WF', weight_fw))
                #################
                ### FREQUENY LOSS
                ## fft does not support halftensor
                # # mask_h, mask_l = decide_circle(r=21,N=batch_size, L=height_width)
                # # mask_h, mask_l = mask_h.cuda(), mask_l.cuda()
                # forward_fft_loss = fft_L1_loss_color(fake_image=forward_image, real_image=modified_input)
                # backward_fft_loss = fft_L1_loss_color(fake_image=reversed_image, real_image=modified_expand)
                # # forward_fft_loss += fft_L1_loss_mask(fake_image=forward_image, real_image=modified_input,mask=mask_l)
                # # backward_fft_loss += fft_L1_loss_mask(fake_image=reversed_image, real_image=modified_expand,mask=mask_l)
                # loss += weight_fw * forward_fft_loss
                # loss += 1 * backward_fft_loss
                # logs.append(('F_FFT', forward_fft_loss.item()))
                # logs.append(('B_FFT', backward_fft_loss.item()))
                ##################
                ## GAN
                # loss += weight_GAN * REV_GAN

                ## LOG FILE
                SSFW = (-l_percept_fw_ssim).item()
                SSBK = (-l_percept_bk_ssim).item()
                # logs.append(('lBi', l_backward_local_ideal.item()))
                # logs.append(('lBedge', l_back_canny.item()))
                logs['PF'] = psnr_forward
                logs['PB'] = psnr_backward
                # logs.append(('NL', l_null.item()))
                logs['SF'] = SSFW
                logs['SB'] = SSBK
                # logs.append(('FW', psnr_forward))
                # logs.append(('BK', psnr_backward))
                # logs.append(('LOCAL', l_backward_local.item()))

                ##### handling errorous example
                loss_without_canny = (loss-l_null).item()
                if loss_without_canny > 0.1 or error_l1 < 25 or real_and_simulate < 30 or CE>0.2:
                    print(
                        f"Error in tamper {self.global_step % self.amount_of_tampering} attack {self.global_step % self.amount_of_benign_attack}")
                    print(logs)
                    images = stitch_images(
                        self.postprocess(modified_input),
                        self.postprocess(modified_canny),
                        self.postprocess(forward_image),
                        self.postprocess(10 * torch.abs(modified_input - forward_image)),
                        self.postprocess(attacked_forward),
                        self.postprocess(attacked_image),
                        self.postprocess(10 * torch.abs(attacked_forward - attacked_image)),
                        self.postprocess(masks_GT),
                        self.postprocess(torch.sigmoid(gen_attacked_train)),
                        self.postprocess(10 * torch.abs(masks_GT - torch.sigmoid(gen_attacked_train))),
                        self.postprocess(masks_real),
                        self.postprocess(tampered_attacked_image),
                        self.postprocess(reversed_image),
                        self.postprocess(GT_modified_input),
                        self.postprocess(10 * torch.abs(GT_modified_input - reversed_image)),
                        self.postprocess(reversed_canny),
                        self.postprocess(GT_canny),
                        self.postprocess(10 * torch.abs(reversed_canny - GT_canny)),
                        img_per_row=1
                    )

                    name = self.out_space_storage + '/errors/' + str(self.global_step).zfill(5) + "_ " + str(
                        3) + "_ " + str(self.rank) \
                           + ".png"
                    print('\nsaving sample ' + name)
                    images.save(name)

                    if loss_without_canny>1:
                        print(f"log before final explosion: {logs}")
                        raise StopIteration(f"Oh! the gradient has exploded.{loss_without_canny}")

                loss.backward()
                # self.scaler_G.scale(loss).backward()
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.netG.parameters(), 1)
                self.optimizer_G.step()
                # self.scaler_G.step(self.optimizer_G)
                # self.scaler_G.update()


            self.localizer.module.update_clock()

            ### update and track history losses
            self.update_history_losses(index=self.global_step,PSNR=psnr_backward,loss=l_backward_local.item(),
                                       loss_CE=CE.item(),PSNR_attack=error_l1)



            ################# observation zone
            # with torch.no_grad():
            #     REVERSE, _ = self.netG(torch.cat((attacked_real_jpeg * (1 - masks),
            #                    torch.zeros_like(modified_canny).cuda()), dim=1), rev=True)
            #     REVERSE = self.clamp_with_grad(REVERSE)
            #     REVERSE = REVERSE[:, :3, :, :]
            #     l_REV = (self.l1_loss(REVERSE * masks_expand, modified_input * masks_expand))
            #     logs.append(('observe', l_REV.item()))
            anomalies = False #CE_recall.item()>0.5
            if anomalies or self.global_step % 200 == 3 or self.global_step<=10:
                images = stitch_images(
                    self.postprocess(modified_input),
                    self.postprocess(modified_canny),
                    self.postprocess(forward_image),
                    self.postprocess(10 * torch.abs(modified_input - forward_image)),
                    self.postprocess(attacked_forward),
                    self.postprocess(attacked_image),
                    self.postprocess(10 * torch.abs(attacked_forward - attacked_image)),
                    self.postprocess(masks_GT),
                    self.postprocess(torch.sigmoid(gen_attacked_train)),
                    self.postprocess(10 * torch.abs(masks_GT - torch.sigmoid(gen_attacked_train))),
                    self.postprocess(masks_real),
                    self.postprocess(tampered_attacked_image),
                    self.postprocess(reversed_image),
                    self.postprocess(GT_modified_input),
                    self.postprocess(10 * torch.abs(GT_modified_input - reversed_image)),
                    self.postprocess(reversed_canny),
                    self.postprocess(GT_canny),
                    self.postprocess(10 * torch.abs(reversed_canny - GT_canny)),
                    img_per_row=1
                )

                name = self.out_space_storage + '/images/'+self.task_name+'_'+str(3)+'/'\
                       +str(self.global_step).zfill(5) + "_ "+str(3) + "_ "+str(self.rank) \
                       +("" if not anomalies else "_anomaly")+ ".png"
                print('\nsaving sample ' + name)
                images.save(name)

        ######## Finally ####################
        if self.global_step % 1000== 999 or self.global_step==9:
            if self.rank==0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.network_list)
        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
                self.previous_previous_canny = self.previous_canny.clone().detach()
            self.previous_images = self.real_H
            self.previous_canny = self.canny_image
        self.global_step = self.global_step + 1
        return logs, debug_logs, False

    # 进行开运算方式:腐蚀+膨胀
    def Erode_Dilate(self, raw_mask):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        dilate_mask = torch.rand_like(raw_mask)
        for idx_atkimg in range(batch_size):
            grid = raw_mask[idx_atkimg][0].clone().detach()
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            ndarr = cv2.resize(copy.deepcopy(ndarr), (height_width, height_width),
                                          interpolation=cv2.INTER_LINEAR)

            # 获取卷积核
            kernel_erode = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(16, 16))
            kernel_dilate = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(32, 32))
            # 首先进行腐蚀操作

            erode_ndarr = cv2.erode(src=ndarr, kernel=kernel_erode, iterations=1)
            # 再进行膨胀操作
            dilate_ndarr = cv2.dilate(src=erode_ndarr, kernel=kernel_dilate, iterations=1)

            dilate_ndarr = dilate_ndarr.astype(np.float32) / 255.

            dilate_ndarr = torch.from_numpy(
                np.ascontiguousarray(dilate_ndarr)).float()

            dilate_ndarr = dilate_ndarr.unsqueeze(0).unsqueeze(0).cuda()
            dilate_mask[idx_atkimg:idx_atkimg + 1] = dilate_ndarr
        return dilate_mask

    def tamper_based_augmentation(self,*, modified_input,modified_canny,masks,masks_GT, index=None):
        # tamper-based data augmentation
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        if index is None:
            index = self.global_step
        index = index % self.amount_of_tampering
        if index not in self.opt['simulated_copymove_indices'] and index not in self.opt['simulated_copysplicing_indices']:
            modified_input = (modified_input * (1 - masks) + self.previous_images * masks).clone().detach()
            modified_canny = (modified_canny * (1 - masks_GT) + self.previous_canny * masks_GT).clone().detach()

        return modified_input, modified_canny


    def tampering_PAMI(self, *,  forward_image, masks, masks_GT, modified_canny, percent_range,
                       index=None):
        ####### Tamper ###############
        # attacked_forward = torch.zeros_like(modified_input)
        # for img_idx in range(batch_size):
        if index is None:
            index = self.global_step
        index = index % self.amount_of_tampering

        if index in self.opt['simulated_splicing_indices']:  # self.using_splicing():
            ### todo: splicing
            attacked_forward = self.splicing(forward_image=forward_image, masks=masks)
            # attack_name = "splicing"

        elif index in self.opt['simulated_copymove_indices']:  # self.using_copy_move():
            ### todo: copy-move
            attacked_forward, masks, masks_GT = self.copymove(forward_image=forward_image, masks=masks, masks_GT=masks_GT,
                                                              percent_range=percent_range)
            # del self.tamper_shifted
            # del self.mask_shifted
            # torch.cuda.empty_cache()

        elif index in self.opt['simulated_inpainting_indices']:  # self.using_simulated_inpainting:
            ### todo: simulated inpainting
            attacked_forward = self.inpainting_for_PAMI(forward_image=forward_image, masks=masks, modified_canny=modified_canny)

        elif index in self.opt['simulated_copysplicing_indices']:  # self.using_simulated_inpainting:
            ### todo: copy-splicing
            attacked_forward, masks, masks_GT = self.copysplicing(forward_image=forward_image, masks=masks, percent_range=percent_range)

        else:
            raise NotImplementedError("Tamper的Index有错误，请检查！")


        attacked_forward = self.clamp_with_grad(attacked_forward)
        # attacked_forward = self.Quantization(attacked_forward)

        return attacked_forward, masks, masks_GT



    # def benign_attacks(self, attacked_forward, quality_idx, logs):
    #     batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
    #     attacked_real_jpeg = torch.rand_like(attacked_forward).cuda()
    #
    #     if self.global_step % 5 == 1:
    #         blurring_layer = self.gaussian_blur
    #     elif self.global_step % 5 == 2:
    #         blurring_layer = self.median_blur
    #     elif self.global_step % 5 == 0:
    #         blurring_layer = self.resize
    #     else:
    #         blurring_layer = self.identity
    #
    #     quality = int(quality_idx * 5)
    #
    #     jpeg_layer_after_blurring = self.jpeg_simulate[quality_idx - 10][0] if quality < 100 else self.identity
    #     attacked_real_jpeg_simulate = self.Quantization(self.clamp_with_grad(jpeg_layer_after_blurring(blurring_layer(attacked_forward))))
    #     if self.global_step % 5 == 4:
    #         attacked_image = attacked_real_jpeg_simulate
    #     else:  # if self.global_step%5==3:
    #         for idx_atkimg in range(batch_size):
    #             grid = attacked_forward[idx_atkimg]
    #             realworld_attack = self.real_world_attacking_on_ndarray(grid, quality)
    #             attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack
    #
    #         attacked_real_jpeg = attacked_real_jpeg.clone().detach()
    #         attacked_image = attacked_real_jpeg_simulate + (attacked_real_jpeg - attacked_real_jpeg_simulate).clone().detach()
    #
    #     # error_scratch = attacked_real_jpeg - attacked_forward
    #     # l_scratch = self.l1_loss(error_scratch, torch.zeros_like(error_scratch).cuda())
    #     # logs.append(('SCRATCH', l_scratch.item()))
    #     return attacked_image
    #
    # def benign_attacks_without_simulation(self, forward_image, quality_idx, logs):
    #     batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
    #     attacked_real_jpeg = torch.rand_like(forward_image).cuda()
    #
    #     quality = int(quality_idx * 5)
    #
    #     for idx_atkimg in range(batch_size):
    #         grid = forward_image[idx_atkimg]
    #         realworld_attack = self.real_world_attacking_on_ndarray(grid, quality)
    #         attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack
    #
    #     return attacked_real_jpeg
    #
    # def real_world_attacking_on_ndarray(self,grid, qf_after_blur, index=None):
    #     batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
    #     if index is None:
    #         index = self.global_step % 5
    #     if index == 0:
    #         grid = self.resize(grid.unsqueeze(0))[0]
    #     ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #     if index == 1:
    #         realworld_attack = cv2.GaussianBlur(ndarr, (5, 5), 0)
    #     elif index == 2:
    #         realworld_attack = cv2.medianBlur(ndarr, 5)
    #     else:
    #         realworld_attack = ndarr
    #     if qf_after_blur!=100:
    #         _, realworld_attack = cv2.imencode('.jpeg', realworld_attack, (int(cv2.IMWRITE_JPEG_QUALITY), qf_after_blur))
    #         realworld_attack = cv2.imdecode(realworld_attack, cv2.IMREAD_UNCHANGED)
    #     # realworld_attack = data.util.channel_convert(realworld_attack.shape[2], 'RGB', [realworld_attack])[0]
    #     # realworld_attack = cv2.resize(copy.deepcopy(realworld_attack), (height_width, height_width),
    #     #                               interpolation=cv2.INTER_LINEAR)
    #     realworld_attack = realworld_attack.astype(np.float32) / 255.
    #     realworld_attack = torch.from_numpy(
    #         np.ascontiguousarray(np.transpose(realworld_attack, (2, 0, 1)))).float()
    #     realworld_attack = realworld_attack.unsqueeze(0).cuda()
    #     return realworld_attack

    def localization_loss(self,*, model, attacked_image,masks_GT):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        ### LOCALIZATION
        gen_attacked_train = model(attacked_image)

        CE_recall = self.bce_with_logit_loss(gen_attacked_train, masks_GT)

        CE = 0
        CE += CE_recall

        return CE, gen_attacked_train

    def recovery_image_generation(self,attacked_image,masks,modified_canny):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        ## RECOVERY
        tampered_attacked_image = attacked_image * (1 - masks)
        tampered_attacked_image = self.clamp_with_grad(tampered_attacked_image)
        canny_input = torch.zeros_like(modified_canny).cuda()

        reversed_stuff, _ = self.netG(torch.cat((tampered_attacked_image, canny_input), dim=1), rev=True)
        reversed_stuff = self.clamp_with_grad(reversed_stuff)
        reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
        reversed_image = reversed_ch1
        reversed_canny = reversed_ch2


        return reversed_image, reversed_canny

    def GAN_loss(self,model, reversed_image, reversed_canny):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        gen_input_fake = torch.cat((reversed_image,reversed_canny),dim=1)
        # dis_input_real = modified_input.clone().detach()
        # dis_real = self.discriminator_mask(dis_input_real)  # in: (grayscale(1) + edge(1))
        gen_fake = model(gen_input_fake)
        REV_GAN = self.bce_with_logit_loss(gen_fake, torch.ones_like(gen_fake))  # / torch.mean(masks)
        # gen_style_loss = 0
        # for i in range(len(dis_real_feat)):
        #     gen_style_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        # gen_style_loss = gen_style_loss / 5
        # logs.append(('REV_GAN', gen_style_loss.item()))
        # REV_GAN += gen_style_loss
        return REV_GAN

    def GAN_training(self,model, modified_input,modified_canny,reversed_image,reversed_canny,masks_GT):
        dis_input_real = torch.cat((modified_input,modified_canny),dim=1)
        dis_input_fake = torch.cat((reversed_image,reversed_canny),dim=1)
        dis_real = model(dis_input_real)
        dis_fake = model(dis_input_fake)
        dis_real_loss = self.bce_with_logit_loss(dis_real, torch.ones_like(dis_real))
        dis_fake_loss = self.bce_with_logit_loss(dis_fake, 1-masks_GT)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2
        return dis_loss

    def KD_JPEG_Generator_training(self, step, latest_values=None, train=True, eval_dir=None):
        with torch.enable_grad():
            self.KD_JPEG_net.train()
            self.qf_predict_network.train()
            self.generator.train()
            self.optimizer_KD_JPEG.zero_grad()
            self.optimizer_generator.zero_grad()
            self.optimizer_qf_predict.zero_grad()
            self.real_H = self.clamp_with_grad(self.real_H)
            batch_size, num_channels, height_width, _ = self.real_H.shape
            assert batch_size%6==0, "error! note that training kd_jpeg must require that batch_size is dividable by six!"
            lr = self.get_current_learning_rate()
            logs, debug_logs = [], []
            logs.append(('lr', lr))
            with torch.cuda.amp.autocast():
                label_array = [0, 1, 2, 3, 4, 5] * int(batch_size//6)
                label = torch.tensor(label_array).long().cuda()
                bins = [10,30,50,70,90,100]
                qf_input_array = [[bins[i%6]/100.] for i in range(batch_size)] # [[0.1],[0.3],[0.5],[0.7],[0.9],[1.0]]
                qf_input = torch.tensor(qf_input_array).cuda()

                GT_real_jpeg = torch.rand_like(self.real_H)
                for i in range(batch_size):
                    jpeg_image = self.real_world_attacking_on_ndarray(self.real_H[i], qf_after_blur=bins[i%6], index=self.global_step % 5)
                    GT_real_jpeg[i:i+1] = jpeg_image
                    realH_aug_image = self.real_world_attacking_on_ndarray(self.real_H[i], qf_after_blur=100, index=self.global_step % 5)
                    self.real_H[i:i + 1] = realH_aug_image

                ## check
                if self.global_step==0:
                    print(f"label: {label}")
                    print(f"qf_input: {qf_input}")
                    print(f"qf_input_array: {qf_input_array}")

                ## TRAINING QF PREDICTOR
                QF_real = self.qf_predict_network(GT_real_jpeg)

                jpeg_simulated, simul_feats = self.KD_JPEG_net(self.real_H, qf_input)
                jpeg_simulated = self.clamp_with_grad(jpeg_simulated)
                # jpeg_reconstructed, recon_feats = self.generator(GT_real_jpeg, qf_input)
                # jpeg_reconstructed = self.clamp_with_grad(jpeg_reconstructed)

                QF_simul = self.qf_predict_network(jpeg_simulated)
                # QF_recon = self.qf_predict_network(jpeg_reconstructed)

                ###################
                ### LOSSES for student
                loss_stu = 0
                simul_l1 = self.l1_loss(jpeg_simulated, GT_real_jpeg)
                loss_stu += simul_l1
                simul_ce = self.CE_loss(QF_simul, label)
                loss_stu += 0.002 * simul_ce
                # gen_fm_loss = 0
                # for i in range(len(simul_feats)):
                #     gen_fm_loss += self.l1_loss(simul_feats[i], recon_feats[i].detach())
                # gen_fm_loss /= len(simul_feats)
                # loss_stu += 1* gen_fm_loss
                psnr_simul = self.psnr(self.postprocess(jpeg_simulated), self.postprocess(GT_real_jpeg)).item()
                logs.append(('loss_stu', loss_stu.item()))
                logs.append(('simul_ce', simul_ce.item()))
                logs.append(('simul_l1', simul_l1.item()))
                # logs.append(('gen_fm_loss', gen_fm_loss.item()))
                logs.append(('psnr_simul', psnr_simul))
                logs.append(('SIMUL', psnr_simul))

                # ###################
                # ### LOSSES for teacher
                # loss_tea = 0
                # recon_l1 = self.l1_loss(jpeg_reconstructed, GT_real_jpeg)
                # loss_tea += recon_l1
                # recon_ce = self.CE_loss(QF_recon, label)
                # loss_tea += 0.002 * recon_ce
                # psnr_recon = self.psnr(self.postprocess(jpeg_reconstructed), self.postprocess(GT_real_jpeg)).item()
                # logs.append(('loss_tea', loss_tea.item()))
                # logs.append(('recon_ce', recon_ce.item()))
                # logs.append(('recon_l1', recon_l1.item()))
                # logs.append(('psnr_recon', psnr_recon))
                # logs.append(('RECON', psnr_recon))

                ###################
                ### LOSSES for QF predictor
                loss_qf = 0
                loss_qf += self.CE_loss(QF_real, label)
                logs.append(('qf_ce', loss_qf.item()))

            self.optimizer_KD_JPEG.zero_grad()
            # loss.backward()
            self.scaler_KD_JPEG.scale(loss_stu).backward()
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.KD_JPEG_net.parameters(), 1)
            # self.optimizer_G.step()
            self.scaler_KD_JPEG.step(self.optimizer_KD_JPEG)
            self.scaler_KD_JPEG.update()

            # self.optimizer_generator.zero_grad()
            # # CE_train.backward()
            # self.scaler_generator.scale(loss_tea).backward()
            # if self.train_opt['gradient_clipping']:
            #     nn.utils.clip_grad_norm_(self.generator.parameters(), 1)
            # # self.optimizer_localizer.step()
            # self.scaler_generator.step(self.optimizer_generator)
            # self.scaler_generator.update()

            # self.optimizer_qf_predict.zero_grad()
            # # loss.backward()
            # self.scaler_qf_predict.scale(loss_qf).backward()
            # if self.train_opt['gradient_clipping']:
            #     nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            # # self.optimizer_G.step()
            # self.scaler_qf_predict.step(self.optimizer_qf_predict)
            # self.scaler_qf_predict.update()

            ################# observation zone
            # with torch.no_grad():
            # pass
            anomalies = False  # CE_recall.item()>0.5
            if anomalies or self.global_step % 200 == 3 or self.global_step <= 3:
                images = stitch_images(
                    self.postprocess(self.real_H),
                    self.postprocess(GT_real_jpeg),
                    self.postprocess(10 * torch.abs(self.real_H - GT_real_jpeg)),
                    # self.postprocess(jpeg_reconstructed),
                    # self.postprocess(10 * torch.abs(jpeg_reconstructed - GT_real_jpeg)),
                    self.postprocess(jpeg_simulated),
                    self.postprocess(10 * torch.abs(jpeg_simulated - GT_real_jpeg)),
                    img_per_row=1
                )

                name = self.out_space_storage + '/jpeg_images/' + self.task_name + '_' + str(3) + '/' \
                       + str(self.global_step).zfill(5) + "_ " + str(3) + "_ " + str(self.rank) \
                       + ("" if not anomalies else "_anomaly") + ".png"
                print('\nsaving sample ' + name)
                images.save(name)

        ######## Finally ####################

        if self.global_step % 1000 == 999 or self.global_step==9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='jpeg_model', network_list=self.network_list)
        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
                self.previous_previous_canny = self.previous_canny.clone().detach()
            self.previous_images = self.real_H
            self.previous_canny = self.canny_image
        self.global_step = self.global_step + 1
        return logs, debug_logs


    def generate_immunized_images_only(self,step):
        self.netG.eval()
        # print(self.canny_image.shape)
        forward_stuff = self.netG(x=torch.cat((self.real_H_val, self.canny_image_val), dim=1))
        self.immunize = forward_stuff[:, :3, :, :]
        self.immunize = self.clamp_with_grad(self.immunize)
        name = f'/data/20220106_IMUGE/immunized_images_buffer'
        for image_no in range(self.immunize.shape[0]):
            camera_ready = self.immunize[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         f"{name}/{str(self.global_step)}_new.png", nrow=1, padding=0,
                                         normalize=False)

            camera_ready = self.real_H_val[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         f"{name}/{str(self.global_step)}_ori.png", nrow=1, padding=0,
                                         normalize=False)
            print(f"Saved:{name}/{str(self.global_step)}_{image_no}.png")

        self.global_step = self.global_step + 1
        psnr_forward = self.psnr(self.postprocess(self.real_H_val), self.postprocess(self.immunize)).item()
        logs={}
        logs['lr'] = 0
        logs['file_generated'] = psnr_forward

        return logs, None, True

    def localize_and_recover_only(self, data_origin=None,data_immunize=None,data_tampered=None,data_tampersource=None,data_mask=None):
        self.netG.eval()
        self.localizer.eval()
        # print(self.canny_image.shape)
        forward_stuff = self.netG(x=torch.cat((self.real_H_val, self.canny_image_val), dim=1))
        self.immunize = forward_stuff[:, :3, :, :]
        self.immunize = self.clamp_with_grad(self.immunize)
        name = f'/data/20220106_IMUGE/immunized_images_buffer'
        for image_no in range(self.immunize.shape[0]):
            camera_ready = self.immunize[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         f"{name}/{str(self.global_step)}_new.png", nrow=1, padding=0,
                                         normalize=False)
            print(f"Saved:{name}/{str(self.global_step)}_{image_no}.png")

        self.global_step = self.global_step + 1
        psnr_forward = self.psnr(self.postprocess(self.real_H_val), self.postprocess(self.immunize)).item()
        logs = {}
        logs['lr'] = 0
        logs['file_generated'] = psnr_forward

        return logs, None, True

    # def evaluate(self,data_origin=None,data_immunize=None,data_tampered=None,data_tampersource=None,data_mask=None):
    #     self.netG.eval()
    #     self.localizer.eval()
    #
    #     eval_size = self.width_height
    #
    #     load_image_query = {
    #         "load_immunize": False,
    #         "load_tampered": False,
    #         "load_attack": False,
    #     }
    #
    #     with torch.no_grad():
    #         psnr_forward_sum, psnr_backward_sum = [0,0,0,0,0],  [0,0,0,0,0]
    #         ssim_forward_sum, ssim_backward_sum =  [0,0,0,0,0],  [0,0,0,0,0]
    #         F1_sum =  [0,0,0,0,0]
    #         valid_images = [0,0,0,0,0]
    #         logs, debug_logs = [], []
    #         origin_dict = self.get_paths_from_images(data_origin)
    #         immunize_dict = self.get_paths_from_images(data_immunize)
    #         tamper_dict = self.get_paths_from_images(data_tampered)
    #         tampersource_dict = self.get_paths_from_images(data_tampersource)
    #         mask_dict = self.get_paths_from_images(data_mask)
    #
    #         for img_path in tampersource_dict:
    #             ### p q r represent path, dirpath and filename
    #             ### original image
    #             if not (img_path in origin_dict and img_path in mask_dict):
    #                 ## required material not found, skip
    #                 continue
    #
    #             ori_path = origin_dict[img_path] if "1019" not in data_tampered else origin_dict[img_path.split("_")[0]+"_ori.png"]
    #             self.real_H, self.canny_image = self.load_image(ori_path, require_canny=True)
    #             print("Ori: {} {}".format(ori_path, self.real_H.shape))
    #
    #             ### mask image
    #             mask_path = mask_dict[img_path]
    #             self.mask = torch.where(self.load_image(mask_path, grayscale=True)>0.5,1.0,0.0)
    #             self.mask = self.mask.repeat(1,3,1,1)
    #
    #             ### tamper source
    #             if tampersource_dict is not None:
    #                 another_path = tampersource_dict[img_path]
    #                 self.another_image = self.load_image(another_path)
    #                 ### clean up signal ###
    #                 self.another_image = self.gaussian_blur.forward_with_specific_kernel(self.another_image, kernel=3)
    #
    #
    #             ### skip images with too large tamper masks
    #             masked_rate = torch.mean(self.mask)
    #             # redo_gen_mask = masked_rate>0.5
    #
    #             catogory = min(4,int(masked_rate*20))
    #             valid_images[catogory] += 1
    #             # is_copy_move = False
    #
    #             ### immunized image
    #             if load_image_query["load_immunize"]:
    #                 immu_path = immunize_dict[img_path]
    #                 self.immunize = self.load_image(immu_path)
    #             else:
    #                 ##### clean up signal ########
    #                 modified_input = self.gaussian_blur.forward_with_specific_kernel(self.real_H,kernel=3)
    #                 locs, cropped, modified_input = self.cropping_mask_generation(
    #                     forward_image=modified_input, min_rate=self.opt['cropping_lower_bound'], max_rate=1.0)
    #
    #                 # print(self.canny_image.shape)
    #                 # forward_stuff = self.netG(x=torch.cat((modified_input, self.canny_image), dim=1))
    #                 # self.immunize, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
    #                 # self.immunize = self.clamp_with_grad(self.immunize)
    #                 # self.immunize = self.immunize * self.mask + modified_input * (1-self.mask)
    #
    #                 self.immunize = modified_input
    #
    #                 # self.immunize = self.Quantization(self.immunize)
    #                 # forward_null = self.clamp_with_grad(forward_null)
    #
    #             ####### Tamper ###############
    #             if load_image_query["load_tampered"]: #self.attacked_image is None:
    #                 ### tampered image
    #                 attack_path = attack_path[img_path]
    #                 self.diffused_image = self.load_image(attack_path)
    #             else:
    #                 self.diffused_image = self.immunize * (1-self.mask) + self.another_image * self.mask
    #                 self.diffused_image = self.clamp_with_grad(self.diffused_image)
    #
    #             ####### CV2 (real-world image postprocessing) ###############
    #             index = self.opt['test']['index']
    #             quality_idx = self.opt['test']['quality_idx'] #self.get_quality_idx_by_iteration(index=index)
    #             kernel = random.choice([3, 5, 7])  # 3,5,7
    #             resize_ratio = (int(self.random_float(0.7, 1.5) * self.width_height),
    #                             int(self.random_float(0.7, 1.5) * self.width_height))
    #             self.attacked_image = self.real_world_attacking_on_ndarray(grid=self.diffused_image[0],
    #                                                                        qf_after_blur=quality_idx*5,
    #                                                                        index=index, kernel=kernel, resize_ratio=resize_ratio)
    #             psnr_diff = self.psnr(self.postprocess(self.attacked_image), self.postprocess(self.diffused_image)).item()
    #             print(f"psnr diff: {psnr_diff}")
    #
    #             self.reverse_GT = self.real_world_attacking_on_ndarray(grid=self.real_H[0],
    #                                                                        qf_after_blur=quality_idx*5,
    #                                                                        index=index, kernel=kernel, resize_ratio=resize_ratio)
    #
    #             ###### mask prediction #########
    #             self.predicted_mask = torch.sigmoid(self.localizer(self.attacked_image))
    #             self.predicted_mask_binary = torch.where(self.predicted_mask > 0.5, 1.0, 0.0)
    #             F1, TP = self.F1score(self.predicted_mask_binary, torch.where(self.mask[:,:1]>0.5,1.0,0.0), thresh=0.5)
    #             F1_sum[catogory] += F1
    #             self.predicted_mask_binary = self.Erode_Dilate(self.predicted_mask_binary)
    #             # masks_GT = self.predicted_mask_binary
    #             self.predicted_mask_binary = self.predicted_mask_binary.repeat(1, 3, 1, 1)
    #
    #             ####### image recovery #########
    #             self.rectified_image = self.attacked_image * (1 - self.predicted_mask_binary)
    #             self.rectified_image = self.clamp_with_grad(self.rectified_image)
    #             canny_input = torch.zeros_like(self.canny_image).cuda()
    #             reversed_stuff, reverse_feature = self.netG(
    #                 torch.cat((self.rectified_image, canny_input), dim=1), rev=True)
    #             reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
    #             reversed_ch1 = self.clamp_with_grad(reversed_ch1)
    #             reversed_ch2 = self.clamp_with_grad(reversed_ch2)
    #             self.reversed_image = reversed_ch1
    #             self.reversed_canny = reversed_ch2
    #
    #             ####### clean up #########
    #             psnr_forward = self.psnr(self.postprocess(self.real_H), self.postprocess(self.immunize)).item()
    #             psnr_backward = self.psnr(self.postprocess(self.reverse_GT),
    #                                       self.postprocess(self.reversed_image)).item()
    #             l_percept_fw_ssim = - self.ssim_loss(self.immunize, self.real_H)
    #             l_percept_bk_ssim = - self.ssim_loss(self.reversed_image, self.reverse_GT)
    #             ssim_forward = (-l_percept_fw_ssim).item()
    #             ssim_backward = (-l_percept_bk_ssim).item()
    #             psnr_forward_sum[catogory] += psnr_forward
    #             psnr_backward_sum[catogory] += psnr_backward
    #             ssim_forward_sum[catogory] += ssim_forward
    #             ssim_backward_sum[catogory] += ssim_backward
    #             print("PF {:2f} PB {:2f} ".format(psnr_forward,psnr_backward))
    #             print("SF {:3f} SB {:3f} ".format(ssim_forward, ssim_backward))
    #             print("PFSum {:2f} SFSum {:2f} ".format(np.sum(psnr_forward_sum)/np.sum(valid_images),
    #                                                     np.sum(ssim_forward_sum)/np.sum(valid_images)))
    #             print("PB {:3f} {:3f} {:3f} {:3f} {:3f}".format(
    #                 psnr_backward_sum[0]/(valid_images[0]+1e-3), psnr_backward_sum[1]/(valid_images[1]+1e-3),
    #                 psnr_backward_sum[2]/(valid_images[2]+1e-3),
    #                 psnr_backward_sum[3]/(valid_images[3]+1e-3), psnr_backward_sum[4]/(valid_images[4]+1e-3)))
    #             print("SB {:3f} {:3f} {:3f} {:3f} {:3f}".format(
    #                 ssim_backward_sum[0] / (valid_images[0]+1e-3), ssim_backward_sum[1] / (valid_images[1]+1e-3),
    #                 ssim_backward_sum[2] / (valid_images[2]+1e-3),
    #                 ssim_backward_sum[3] / (valid_images[3]+1e-3), ssim_backward_sum[4] / (valid_images[4]+1e-3)))
    #             print("F1 {:3f} {:3f} {:3f} {:3f} {:3f}".format(
    #                 F1_sum[0] / (valid_images[0]+1e-3), F1_sum[1] / (valid_images[1]+1e-3),
    #                 F1_sum[2] / (valid_images[2]+1e-3),
    #                 F1_sum[3] / (valid_images[3]+1e-3), F1_sum[4] / (valid_images[4]+1e-3)))
    #             print("Valid {:3f} {:3f} {:3f} {:3f} {:3f}".format(valid_images[0],valid_images[1],valid_images[2],
    #                                                                valid_images[3],valid_images[4]))
    #
    #             ######### Save independent images #############
    #             save_images = True
    #             if save_images:
    #                 # eval_kind = self.opt['eval_kind'] #'copy-move/results/RESIZE'
    #                 # eval_attack = self.opt['eval_attack']
    #                 # main_folder = os.path.join(self.out_space_storage,'results', self.opt['dataset_name'], eval_kind)
    #                 # sub_folder = os.path.join(main_folder,eval_attack)
    #                 # if not os.path.exists(main_folder): os.mkdir(main_folder)
    #                 # if not os.path.exists(sub_folder): os .mkdir(sub_folder)
    #                 # if not os.path.exists(sub_folder+ '/recovered_image'): os.mkdir(sub_folder+ '/recovered_image')
    #                 # if not os.path.exists(sub_folder + '/predicted_masks'): os.mkdir(sub_folder + '/predicted_masks')
    #                 sub_folder = self.opt['path']['data_storage']
    #
    #                 name = sub_folder + '/recovered_image/' + img_path
    #                 for image_no in range(self.reversed_image.shape[0]):
    #                     camera_ready = self.reversed_image[image_no].unsqueeze(0)
    #                     torchvision.utils.save_image((camera_ready * 255).round() / 255,
    #                                                  name, nrow=1, padding=0,
    #                                                  normalize=False)
    #                 print("Saved:{}".format(name))
    #
    #                 name = sub_folder + '/immunized_images/' + img_path
    #                 for image_no in range(self.immunize.shape[0]):
    #                     camera_ready = self.immunize[image_no].unsqueeze(0)
    #                     torchvision.utils.save_image((camera_ready * 255).round() / 255,
    #                                                  name, nrow=1, padding=0,
    #                                                  normalize=False)
    #                 print("Saved:{}".format(name))
    #
    #                 name = sub_folder + '/input/' + img_path
    #                 for image_no in range(self.real_H.shape[0]):
    #                     camera_ready = self.real_H[image_no].unsqueeze(0)
    #                     torchvision.utils.save_image((camera_ready * 255).round() / 255,
    #                                                  name, nrow=1, padding=0,
    #                                                  normalize=False)
    #                 print("Saved:{}".format(name))
    #
    #                 name = sub_folder + '/forged/' + img_path
    #                 for image_no in range(self.attacked_image.shape[0]):
    #                     camera_ready = self.attacked_image[image_no].unsqueeze(0)
    #                     torchvision.utils.save_image((camera_ready * 255).round() / 255,
    #                                                  name, nrow=1, padding=0,
    #                                                  normalize=False)
    #                 print("Saved:{}".format(name))
    #
    #                 name = sub_folder + '/mask_GT/' + img_path
    #                 for image_no in range(self.mask.shape[0]):
    #                     camera_ready = self.mask[image_no].unsqueeze(0)
    #                     torchvision.utils.save_image((camera_ready * 255).round() / 255,
    #                                                  name, nrow=1, padding=0,
    #                                                  normalize=False)
    #                 print("Saved:{}".format(name))
    #
    #                 name = sub_folder + '/predicted_masks/' + img_path
    #
    #                 for image_no in range(self.predicted_mask.shape[0]):
    #                     camera_ready = self.predicted_mask[image_no].unsqueeze(0)
    #                     torchvision.utils.save_image((camera_ready * 255).round() / 255,
    #                                                  name, nrow=1, padding=0,
    #                                                  normalize=False)
    #                 print("Saved:{}".format(name))

    def evaluate(self,data_origin=None,data_immunize=None,data_tampered=None,data_tampersource=None,data_mask=None):
        self.netG.eval()
        self.localizer.eval()

        eval_size = self.width_height

        load_image_query = {
            "load_immunize": False,
            "load_tampered": False,
            "load_attack": False,
        }

        with torch.no_grad():
            psnr_forward_sum, psnr_backward_sum = [0,0,0,0,0],  [0,0,0,0,0]
            ssim_forward_sum, ssim_backward_sum =  [0,0,0,0,0],  [0,0,0,0,0]
            F1_sum =  [0,0,0,0,0]
            valid_images = [0,0,0,0,0]
            logs, debug_logs = [], []
            origin_dict = self.get_paths_from_images(data_origin)
            immunize_dict = self.get_paths_from_images(data_immunize)
            tamper_dict = self.get_paths_from_images(data_tampered)
            tampersource_dict = self.get_paths_from_images(data_tampersource)
            mask_dict = self.get_paths_from_images(data_mask)

            for img_path in tampersource_dict:
                ### p q r represent path, dirpath and filename
                ### original image
                if not (img_path in origin_dict and img_path in mask_dict):
                    ## required material not found, skip
                    continue

                ori_path = origin_dict[img_path] if "10" not in data_tampered else origin_dict[img_path.split("_")[0]+"_ori.png"]
                self.real_H, self.canny_image = self.load_image(ori_path, require_canny=True)
                print("Ori: {} {}".format(ori_path, self.real_H.shape))

                ### mask image
                mask_path = mask_dict[img_path]
                self.mask = torch.where(self.load_image(mask_path, grayscale=True)>0.5,1.0,0.0)
                self.mask = self.mask.repeat(1,3,1,1)

                ### tamper source
                if tampersource_dict is not None:
                    another_path = tampersource_dict[img_path]
                    self.another_image = self.load_image(another_path)

                ### skip images with too large tamper masks
                masked_rate = torch.mean(self.mask)
                redo_gen_mask = masked_rate>0.5

                catogory = min(4,int(masked_rate*20))
                valid_images[catogory] += 1
                is_copy_move = False

                ### immunized image
                if load_image_query["load_immunize"]:
                    immu_path = immunize_dict[img_path]
                    self.immunize = self.load_image(immu_path)
                else:
                    ##### re-generates immunized images ########
                    modified_input = self.real_H
                    # print(self.canny_image.shape)
                    forward_stuff = self.netG(x=torch.cat((modified_input, self.canny_image), dim=1))
                    self.immunize, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                    self.immunize = self.clamp_with_grad(self.immunize)
                    # self.immunize = self.Quantization(self.immunize)
                    forward_null = self.clamp_with_grad(forward_null)

                ####### Tamper ###############
                if load_image_query["load_tampered"]: #self.attacked_image is None:
                    ### tampered image
                    attack_path = attack_path[img_path]
                    self.diffused_image = self.load_image(attack_path)
                else:
                    self.diffused_image = self.immunize * (1-self.mask) + self.another_image * self.mask
                    self.diffused_image = self.clamp_with_grad(self.diffused_image)

                ####### CV2 (real-world image postprocessing) ###############
                index = self.opt['test']['index']
                quality_idx = self.opt['test']['quality_idx'] #self.get_quality_idx_by_iteration(index=index)
                kernel = random.choice([3, 5, 7])  # 3,5,7
                resize_ratio = (int(self.random_float(0.7, 1.5) * self.width_height),
                                int(self.random_float(0.7, 1.5) * self.width_height))
                self.attacked_image = self.real_world_attacking_on_ndarray(grid=self.diffused_image[0],
                                                                           qf_after_blur=quality_idx*5,
                                                                           index=index, kernel=kernel, resize_ratio=resize_ratio)
                psnr_diff = self.psnr(self.postprocess(self.attacked_image), self.postprocess(self.diffused_image)).item()
                print(f"psnr diff: {psnr_diff}")

                self.reverse_GT = self.real_world_attacking_on_ndarray(grid=self.real_H[0],
                                                                           qf_after_blur=quality_idx*5,
                                                                           index=index, kernel=kernel, resize_ratio=resize_ratio)

                ###### mask prediction #########
                self.predicted_mask = torch.sigmoid(self.localizer(self.attacked_image))
                self.predicted_mask_binary = torch.where(self.predicted_mask > 0.5, 1.0, 0.0)
                F1, TP = self.F1score(self.predicted_mask_binary, torch.where(self.mask[:,:1]>0.5,1.0,0.0), thresh=0.5)
                F1_sum[catogory] += F1
                self.predicted_mask_binary = self.Erode_Dilate(self.predicted_mask_binary)
                # masks_GT = self.predicted_mask_binary
                self.predicted_mask_binary = self.predicted_mask_binary.repeat(1, 3, 1, 1)

                ####### image recovery #########
                self.rectified_image = self.attacked_image * (1 - self.predicted_mask_binary)
                self.rectified_image = self.clamp_with_grad(self.rectified_image)
                canny_input = torch.zeros_like(self.canny_image).cuda()
                reversed_stuff, reverse_feature = self.netG(
                    torch.cat((self.rectified_image, canny_input), dim=1), rev=True)
                reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
                reversed_ch1 = self.clamp_with_grad(reversed_ch1)
                reversed_ch2 = self.clamp_with_grad(reversed_ch2)
                self.reversed_image = reversed_ch1
                self.reversed_canny = reversed_ch2

                ####### clean up #########
                psnr_forward = self.psnr(self.postprocess(self.real_H), self.postprocess(self.immunize)).item()
                psnr_backward = self.psnr(self.postprocess(self.reverse_GT),
                                          self.postprocess(self.reversed_image)).item()
                l_percept_fw_ssim = - self.ssim_loss(self.immunize, self.real_H)
                l_percept_bk_ssim = - self.ssim_loss(self.reversed_image, self.reverse_GT)
                ssim_forward = (-l_percept_fw_ssim).item()
                ssim_backward = (-l_percept_bk_ssim).item()
                psnr_forward_sum[catogory] += psnr_forward
                psnr_backward_sum[catogory] += psnr_backward
                ssim_forward_sum[catogory] += ssim_forward
                ssim_backward_sum[catogory] += ssim_backward
                print("PF {:2f} PB {:2f} ".format(psnr_forward,psnr_backward))
                print("SF {:3f} SB {:3f} ".format(ssim_forward, ssim_backward))
                print("PFSum {:2f} SFSum {:2f} ".format(np.sum(psnr_forward_sum)/np.sum(valid_images),
                                                        np.sum(ssim_forward_sum)/np.sum(valid_images)))
                print("PB {:3f} {:3f} {:3f} {:3f} {:3f}".format(
                    psnr_backward_sum[0]/(valid_images[0]+1e-3), psnr_backward_sum[1]/(valid_images[1]+1e-3),
                    psnr_backward_sum[2]/(valid_images[2]+1e-3),
                    psnr_backward_sum[3]/(valid_images[3]+1e-3), psnr_backward_sum[4]/(valid_images[4]+1e-3)))
                print("SB {:3f} {:3f} {:3f} {:3f} {:3f}".format(
                    ssim_backward_sum[0] / (valid_images[0]+1e-3), ssim_backward_sum[1] / (valid_images[1]+1e-3),
                    ssim_backward_sum[2] / (valid_images[2]+1e-3),
                    ssim_backward_sum[3] / (valid_images[3]+1e-3), ssim_backward_sum[4] / (valid_images[4]+1e-3)))
                print("F1 {:3f} {:3f} {:3f} {:3f} {:3f}".format(
                    F1_sum[0] / (valid_images[0]+1e-3), F1_sum[1] / (valid_images[1]+1e-3),
                    F1_sum[2] / (valid_images[2]+1e-3),
                    F1_sum[3] / (valid_images[3]+1e-3), F1_sum[4] / (valid_images[4]+1e-3)))
                print("Valid {:3f} {:3f} {:3f} {:3f} {:3f}".format(valid_images[0],valid_images[1],valid_images[2],
                                                                   valid_images[3],valid_images[4]))

                ######### Save independent images #############
                save_images = True
                if save_images:
                    # eval_kind = self.opt['eval_kind'] #'copy-move/results/RESIZE'
                    # eval_attack = self.opt['eval_attack']
                    # main_folder = os.path.join(self.out_space_storage,'results', self.opt['dataset_name'], eval_kind)
                    # sub_folder = os.path.join(main_folder,eval_attack)
                    # if not os.path.exists(main_folder): os.mkdir(main_folder)
                    # if not os.path.exists(sub_folder): os .mkdir(sub_folder)
                    # if not os.path.exists(sub_folder+ '/recovered_image'): os.mkdir(sub_folder+ '/recovered_image')
                    # if not os.path.exists(sub_folder + '/predicted_masks'): os.mkdir(sub_folder + '/predicted_masks')
                    sub_folder = self.opt['path']['data_storage']

                    name = sub_folder + '/recovered_image/' + img_path
                    for image_no in range(self.reversed_image.shape[0]):
                        camera_ready = self.reversed_image[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = sub_folder + '/immunized_images/' + img_path
                    for image_no in range(self.immunize.shape[0]):
                        camera_ready = self.immunize[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = sub_folder + '/input/' + img_path
                    for image_no in range(self.real_H.shape[0]):
                        camera_ready = self.real_H[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = sub_folder + '/forged/' + img_path
                    for image_no in range(self.attacked_image.shape[0]):
                        camera_ready = self.attacked_image[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = sub_folder + '/mask_GT/' + img_path
                    for image_no in range(self.mask.shape[0]):
                        camera_ready = self.mask[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = sub_folder + '/predicted_masks/' + img_path

                    for image_no in range(self.predicted_mask.shape[0]):
                        camera_ready = self.predicted_mask[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))


    def reload(self,pretrain, network_list=['netG','localizer']):
        if 'netG' in network_list:
            load_path_G = pretrain + "_netG.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.netG, strict=True)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'KD_JPEG' in network_list:
            load_path_G = pretrain + "_KD_JPEG.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.KD_JPEG_net, strict=True)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'discriminator_mask' in network_list:
            load_path_G = pretrain + "_discriminator_mask.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.discriminator_mask, strict=True)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'qf_predict' in network_list:
            load_path_G = pretrain + "_qf_predict.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.qf_predict_network, self.opt['path']['strict_load'])
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'localizer' in network_list:
            load_path_G = pretrain + "_localizer.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.localizer, strict=True)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'generator' in network_list:
            load_path_G = pretrain + "_generator.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.generator, strict=True)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

    def save(self, iter_label, folder='model', network_list=['netG','localizer']):
        if 'netG' in network_list:
            self.save_network(self.netG, 'netG', iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/')
        if 'localizer' in network_list:
            self.save_network(self.localizer,  'localizer', iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/')
        if 'KD_JPEG' in network_list:
            self.save_network(self.KD_JPEG_net, 'KD_JPEG', iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/')
        if 'discriminator_mask' in network_list:
            self.save_network(self.discriminator_mask, 'discriminator_mask', iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/')
        if 'qf_predict' in network_list:
            self.save_network(self.qf_predict_network, 'qf_predict', iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/')
        if 'generator' in network_list:
            self.save_network(self.generator, 'generator', iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/')
        # if 'netG' in network_list:
        # self.save_training_state(epoch=0, iter_step=iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/',
        #                          network_list=network_list)

    def generate_stroke_mask(self, im_size, parts=5, parts_square=2, maxVertex=6, maxLength=64, maxBrushWidth=32,
                             maxAngle=360, percent_range=(0.0, 0.25)):
        minVertex, maxVertex = 1,4
        minLength, maxLength = int(im_size[0] *0.02), int(im_size[0] * 0.25)
        minBrushWidth, maxBrushWidth = int(im_size[0] *0.02), int(im_size[0] * 0.25)
        mask = np.zeros((im_size[0], im_size[1]), dtype=np.float32)
        lower_bound_percent = percent_range[0] + (percent_range[1] - percent_range[0]) * np.random.rand()

        while True:
            mask = mask + self.np_free_form_mask(mask, minVertex, maxVertex, minLength, maxLength, minBrushWidth, maxBrushWidth,
                                                 maxAngle, im_size[0],
                                                 im_size[1])
            mask = np.minimum(mask, 1.0)
            percent = np.mean(mask)
            if percent >= lower_bound_percent:
                break

        mask = np.maximum(mask, 0.0)
        mask_tensor = torch.from_numpy(mask).contiguous()
        # mask = Image.fromarray(mask)
        # mask_tensor = F.to_tensor(mask).float()

        return mask_tensor, np.mean(mask)

    def np_free_form_mask(self, mask_re, minVertex, maxVertex, minLength, maxLength, minBrushWidth, maxBrushWidth, maxAngle, h, w):
        mask = np.zeros_like(mask_re)
        numVertex = np.random.randint(minVertex, maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(minLength, maxLength + 1)
            brushWidth = np.random.randint(minBrushWidth, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)
            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        return mask

    def get_random_rectangle_inside(self, image_width, image_height, height_ratio_range=(0.1, 0.2),
                                    width_ratio_range=(0.1, 0.2)):

        r_float_height, r_float_width = \
            self.random_float(height_ratio_range[0], height_ratio_range[1]), self.random_float(width_ratio_range[0],
                                                                                               width_ratio_range[1])
        remaining_height = int(np.rint(r_float_height * image_height))
        remaining_width = int(np.rint(r_float_width * image_width))

        if remaining_height == image_height:
            height_start = 0
        else:
            height_start = np.random.randint(0, image_height - remaining_height)

        if remaining_width == image_width:
            width_start = 0
        else:
            width_start = np.random.randint(0, image_width - remaining_width)

        return height_start, height_start + remaining_height, width_start, width_start + remaining_width, r_float_height * r_float_width

    def random_float(self, min, max):
        return np.random.rand() * (max - min) + min

    def F1score(self, predict_image, gt_image, thresh=0.2):
        # gt_image = cv2.imread(src_image, 0)
        # predict_image = cv2.imread(dst_image, 0)
        # ret, gt_image = cv2.threshold(gt_image[0], int(255 * thresh), 255, cv2.THRESH_BINARY)
        # ret, predicted_binary = cv2.threshold(predict_image[0], int(255*thresh), 255, cv2.THRESH_BINARY)
        predicted_binary = self.tensor_to_image(predict_image[0])
        ret, predicted_binary = cv2.threshold(predicted_binary, int(255 * thresh), 255, cv2.THRESH_BINARY)
        gt_image = self.tensor_to_image(gt_image[0,:1,:,:])
        ret, gt_image = cv2.threshold(gt_image, int(255 * thresh), 255, cv2.THRESH_BINARY)

        # print(predicted_binary.shape)

        [TN, TP, FN, FP] = getLabels(predicted_binary, gt_image)
        # print("{} {} {} {}".format(TN,TP,FN,FP))
        F1 = getF1(TP, FP, FN)
        # cv2.imwrite(save_path, predicted_binary)
        return F1, TP

def getLabels(img, gt_img):
    height = img.shape[0]
    width = img.shape[1]
    #TN, TP, FN, FP
    result = [0, 0, 0 ,0]
    for row in range(height):
        for column in range(width):
            pixel = img[row, column]
            gt_pixel = gt_img[row, column]
            if pixel == gt_pixel:
                result[(pixel // 255)] += 1
            else:
                index = 2 if pixel == 0 else 3
                result[index] += 1
    return result

def getACC(TN, TP, FN, FP):
    return (TP+TN)/(TP+FP+FN+TN)
def getFPR(TN, FP):
    return FP / (FP + TN)
def getTPR(TP, FN):
    return TP/ (TP+ FN)
def getTNR(FP, TN):
    return TN/ (FP+ TN)
def getFNR(FN, TP):
    return FN / (TP + FN)
def getF1(TP, FP, FN):
    return (2*TP)/(2*TP+FP+FN)
def getBER(TN, TP, FN, FP):
    return 1/2*(getFPR(TN, FP)+FN/(FN+TP))