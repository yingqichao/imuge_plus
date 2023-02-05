import os

import torch
import torch.nn as nn
# from cycleisp_models.cycleisp import Raw2Rgb
# from MVSS.models.mvssnet import get_mvss
# from MVSS.models.resfcn import ResFCN
# from data.pipeline import pipeline_tensor2image
# import matlab.engine
from torch.nn.parallel import DistributedDataParallel
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
# from .networks import SPADE_UNet
# from .invertible_net import Inveritible_Decolorization_PAMI
# from models.networks import UNetDiscriminator
# from loss import PerceptualLoss, StyleLoss
# from .networks import SPADE_UNet
# from lama_models.HWMNet import HWMNet
# import contextual_loss as cl
# import contextual_loss.functional as F
from noise_layers import *
from utils import stitch_images
from models.IFA.base_IFA import base_IFA
from losses.emd_loss import emd_loss

class IFA_distill(base_IFA):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """
            this file is mode 4

        """
        super(IFA_distill, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants
        self.history_accuracy = 0.1


    def network_definitions(self):
        self.IFA_bc_label = None
        ### mode=8: InvISP to bi-directionally convert RGB and RAW,
        self.network_list = ['qf_predict_network']
        self.save_network_list = ['qf_predict_network']
        self.training_network_list = ['qf_predict_network']

        ### todo: network
        ### todo: network

        self.network_definitions_distill()


        if self.opt['load_predictor_models'] is not None:
            # self.load_model_wrapper(folder_name='predictor_folder', model_name='load_predictor_models',
            #                         network_lists=["qf_predict_network"], strict=False)
            load_detector_storage = self.opt['predictor_folder']
            model_path = str(self.opt['load_predictor_models'])  # last time: 10999

            print(f"loading models: {self.network_list}")
            pretrain = load_detector_storage + model_path
            load_path_G = pretrain #+"_qf_predict.pth"

            print('Loading model for class [{:s}] ...'.format(load_path_G))
            if os.path.exists(load_path_G):
                self.load_network(load_path_G, self.qf_predict_network, strict=False)

            else:
                print('Did not find model for class [{:s}] ...'.format(load_path_G))


        # ### todo: inpainting model
        # self.define_inpainting_edgeconnect()
        # self.define_inpainting_ZITS()
        # self.define_inpainting_lama()


    def network_definitions_distill(self):
        self.network_list.append('generator')
        self.save_network_list.append('generator')
        self.training_network_list.append('generator')

        self.bc_label = torch.cat([torch.ones((self.batch_size,1)).cuda(), torch.zeros((self.batch_size,1)).cuda()],dim=0)
        self.adv_label = torch.ones((2*self.batch_size,1)).cuda()

        self.detection_image, self.detection_mask = None, None
        self.timestamp = torch.zeros((1)).cuda()
        self.zero_metric = torch.zeros((self.batch_size,3,self.width_height,self.width_height)).cuda()

        ### todo: network
        self.qf_predict_network = self.define_ddpm_unet_network(out_dim=[1], use_bayar=3, dim=32, use_fft=True, use_hierarchical_class=True,
                                                                use_classification=False, use_normal_output=True)
        self.generator = self.define_ddpm_unet_network(out_dim=[3*2*1], dim=32, use_bayar=0, use_fft=True,
                                                                use_classification=False, use_middle_features=False)

        if self.opt['load_generator_models'] is not None:
            # self.load_model_wrapper(folder_name='predictor_folder', model_name='load_predictor_models',
            #                         network_lists=["qf_predict_network"], strict=False)
            load_detector_storage = self.opt['predictor_folder']
            model_path = str(self.opt['load_generator_models'])  # last time: 10999

            print(f"loading models: {self.network_list}")
            pretrain = load_detector_storage + model_path
            load_path_G = pretrain #+"_qf_predict.pth"

            print('Loading model for class [{:s}] ...'.format(load_path_G))
            if os.path.exists(load_path_G):
                self.load_network(load_path_G, self.generator, strict=False)

            else:
                print('Did not find model for class [{:s}] ...'.format(load_path_G))


    def IFA_distill(self, step=None, epoch=None):
        #### SYMBOL FOR NOTIFYING THE OUTER VAL LOADER #######
        did_val = False
        if step is not None:
            self.global_step = step

        logs, debug_logs = {}, []
        # self.real_H = self.clamp_with_grad(self.real_H)
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        stage_index = self.global_step
        warmup_stage = 10000

        ## clean up casia dataset to prevent too large or too small masks
        # filtered_detection_image, filtered_detection_mask = None, None
        # for i in range(self.batch_size):
        #     rate = torch.mean(self.detection_mask[i])
        #     if not(rate<0.01 or rate>0.35) or (self.world_size>1 and filtered_detection_image is None and i==self.batch_size-1):
        #         ## prevent empty training material that will stuck ddp training
        #         if filtered_detection_image is None:
        #             filtered_detection_mask = self.detection_mask[i:i+1]
        #             filtered_detection_image = self.detection_image[i:i+1]
        #         else:
        #             filtered_detection_mask = torch.cat([filtered_detection_mask,self.detection_mask[i:i + 1]],dim=0)
        #             filtered_detection_image = torch.cat([filtered_detection_image,self.detection_image[i:i + 1]],dim=0)
        #
        #         if filtered_detection_image.shape[0]==4:
        #             break
        #
        # self.detection_mask = filtered_detection_mask
        # self.detection_image = filtered_detection_image
        # self.batch_size = filtered_detection_image.shape[0]

        if False: ##epoch==0 and stage_index<warmup_stage and self.opt['load_predictor_models'] is None:
            ## stage 1: original
            self.distill_stage_one(logs=logs, epoch=epoch)
        elif True: #((stage_index-warmup_stage)) % 2 in [0]:
            ## stage 2
            self.distill_stage_two(logs=logs, epoch=epoch)
        else:
            ## stage 3
            self.distill_stage_three(logs=logs, epoch=epoch)

        ######## Finally ####################
        # for scheduler in self.schedulers:
        #     scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                # self.save(self.global_step, folder='model', network_list=self.save_network_list)
                self.save_network(self.qf_predict_network, 'qf_predict', f"{epoch}_{self.global_step}",
                                  model_path=f'{self.out_space_storage}/model/{self.task_name}/')
                self.save_network(self.generator, 'generator', f"{epoch}_{self.global_step}",
                                  model_path=f'{self.out_space_storage}/model/{self.task_name}/')

        self.global_step = self.global_step + 1

        return logs, None, False

    def distill_stage_one(self, *, logs, epoch):
        logs['status'] = 'One'
        self.qf_predict_network.train()
        self.generator.eval()

        with torch.enable_grad():
            no_grad_features, pred_class, pred_detection_mask = self.qf_predict_network(self.detection_image, None)

            loss_CE_detection = self.bce_with_logit_loss(pred_detection_mask, self.detection_mask)
            logs['CE_detection'] = loss_CE_detection.item()
            loss_CE_detection.backward()

            nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()

        ##### printing the images  ######
        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.detection_image),
                self.postprocess(torch.sigmoid(pred_detection_mask)),
                self.postprocess(self.detection_mask),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{epoch}_{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

    def distill_stage_two(self, *, logs, epoch):
        logs['status'] = 'Two'
        self.qf_predict_network.train()
        self.generator.train()

        # self.real_H = self.real_H[:self.batch_size]

        # masks_full, masks_GT_full, percent_range = self.mask_generation(modified_input=self.real_H,
        #                                                                 index=np.random.randint(0, 10000),
        #                                                                 percent_range=(0.0, 0.15))
        masks_GT_full = self.detection_mask

        with torch.no_grad():
            _, _, _, no_grad_features_detection, _ = self.qf_predict_network.module.feature_extract(
                self.detection_image)

        with torch.enable_grad():
            # pred_detection_mask, no_grad_features = self.qf_predict_network(self.detection_image, None)
            # loss_CE_detection = self.bce_with_logit_loss(pred_detection_mask, self.detection_mask)
            # logs['CE_detection'] = loss_CE_detection.item()

            ### generator
            _, noise_patterns = self.generator(self.real_H, None)
            which_pattern = self.global_step%2
            selected_patterns = noise_patterns[:,which_pattern*3:(which_pattern+1)*3]
            if True: #which_pattern in [0,1]:
                ## semantic based
                mixed_image = selected_patterns #self.real_H + masks_GT_full * selected_patterns
                txt = "S"
            else:
                ## noise based
                mixed_image = self.real_H * (1-masks_GT_full) + masks_GT_full * selected_patterns
                txt = "N"
            mixed_image = self.clamp_with_grad(mixed_image)
            # psnr = self.psnr(self.postprocess(mixed_image), self.postprocess(self.real_H)).item()

            # logs[f'PSNR_{txt}'] = psnr

            logs['rate'] = torch.mean(masks_GT_full).item()

            no_grad_features, pred_class, pred_mask = self.qf_predict_network(mixed_image, None)
            ## collect features for training d
            no_grad_new_feats = []
            for i in range(len(no_grad_features_detection)):
                no_grad_new_feats.append(torch.cat([no_grad_features_detection[i], no_grad_features[i].detach()], dim=0))

            loss_CE = self.bce_with_logit_loss(pred_mask, masks_GT_full)
            logs[f'CE_{txt}'] = loss_CE.item()
            loss_adv = self.bce_with_logit_loss(pred_class, self.adv_label[self.batch_size:])
            logs[f'adv_{txt}'] = loss_adv.item()

            loss = 0
            loss += loss_CE
            loss += loss_adv
            logs[f'loss_g_{txt}'] = logs[f'CE_{txt}'] + logs[f'adv_{txt}']
            if False: #not which_pattern in [0,1]:
                ## noise based
                alpha = self.exponential_weight_for_backward(value=psnr, norm=-1, alpha=1, exp=2, psnr_thresh=30,penalize=10)
                loss_L1 = self.l2_loss(mixed_image, self.real_H)  # pattern loss
                logs['L2'] = loss_L1.item()
                logs['alpha'] = alpha
                loss += alpha * loss_L1
            loss.backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(), 1)
            self.optimizer_generator.step()
            self.optimizer_generator.zero_grad()
            # mixed_image_aug = mixed_image #/ torch.mean(self.detection_mask,dim=[1,2,3]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # loss_L1 = self.l2_loss(mixed_image_aug, self.real_H)  # pattern loss
            # loss_distill = 0
            # for i in range(len(pred_features_sum)):
            #     no_grad_features, pred_features = pred_features_sum[i][:self.batch_size], pred_features_sum[i][self.batch_size:]
            #     loss_this_feature = self.l1_loss(pred_features, no_grad_features.detach())  # distill loss
            #     # loss_this_feature = -torch.mean(self.consine_similarity(pred_features, no_grad_features))
            #     loss_distill += loss_this_feature
            #     logs[f'd{i}'] = loss_this_feature.item()
            # loss_distill *= 1  # /len(pred_features))
            # if loss_distill > 100:
            #     raise StopIteration(f"gradient exploded. {loss_distill.item()}")

            pred_class = self.qf_predict_network.module.forward_detach_hierarchical_class(
                middle_feats = no_grad_new_feats)
            # loss += loss_distill
            loss_bc = self.bce_with_logit_loss(pred_class, self.bc_label)

            loss = 0
            loss += loss_bc
            logs[f'bc_{txt}'] = loss_bc.item()
            # logs['loss_d'] = logs['bc']
            # logs['distill'] = loss_distill.item()

            loss.backward()

            nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()

        ##### printing the images  ######
        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.real_H),
                self.postprocess(selected_patterns),
                self.postprocess(mixed_image),
                self.postprocess(10 * torch.abs(self.real_H - mixed_image)),
                self.postprocess(torch.sigmoid(pred_mask)),
                # self.postprocess(self.detection_image),
                # self.postprocess(torch.sigmoid(pred_detection_mask)),
                # self.postprocess(masks_GT_full),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{epoch}_{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

    def distill_stage_three(self, *, logs, epoch):
        logs['status'] = 'Three'
        self.qf_predict_network.train()
        self.generator.eval()

        self.real_H = self.real_H[:self.batch_size]
        # masks_full, masks_GT_full, percent_range = self.mask_generation(modified_input=self.real_H,
        #                                                                 index=np.random.randint(0, 10000),
        #                                                                 percent_range=(0.0, 0.15))
        masks_GT_full = self.detection_mask
        with torch.no_grad():
            _, noise_patterns = self.generator(self.real_H, None)
            idx_pattern = 0  # self.global_step%5
            selected_patterns = noise_patterns  # [:,idx_pattern*3:(1+idx_pattern)*3]
            mixed_image = self.real_H + masks_GT_full * selected_patterns
            mixed_image = self.clamp_with_grad(mixed_image)
            psnr = self.psnr(self.postprocess(mixed_image), self.postprocess(self.real_H)).item()

        with torch.enable_grad():

            no_grad_features, pred_class, pred_mask_sum = self.qf_predict_network(torch.cat([self.detection_image,mixed_image.detach()],dim=0), None)
            pred_detection_mask, pred_mask = pred_mask_sum[:self.batch_size], pred_mask_sum[self.batch_size:]

            loss_CE_detection = self.bce_with_logit_loss(pred_detection_mask, self.detection_mask)
            logs['CE_detection'] = loss_CE_detection.item()
            loss_CE = self.bce_with_logit_loss(pred_mask, masks_GT_full)
            logs['CE'] = loss_CE.item()

            loss_CE = self.bce_with_logit_loss(pred_mask_sum, torch.cat([self.detection_mask,masks_GT_full],dim=0))  # pred loss
            mixed_image_aug = mixed_image #/ torch.mean(self.detection_mask, dim=[1, 2, 3]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            loss_L1 = self.l1_loss(mixed_image_aug, self.real_H)  # pattern loss
            # loss_distill = 0
            # for i in range(len(pred_features_sum)):
            #     no_grad_features, pred_features = pred_features_sum[i][:self.batch_size], pred_features_sum[i][self.batch_size:]
            #     loss_this_feature = self.l1_loss(pred_features, no_grad_features.detach())  # distill loss
            #     # loss_this_feature = -torch.mean(self.consine_similarity(pred_features, no_grad_features))
            #     loss_distill += loss_this_feature
            #     logs[f'd{i}'] = loss_this_feature.item()
            # loss_distill *= 1  # /len(pred_features))
            # if loss_distill > 100:
            #     raise StopIteration(f"gradient exploded. {loss_distill.item()}")

            loss_bc = self.bce_with_logit_loss(pred_class, self.bc_label)
            loss_adv = self.bce_with_logit_loss(pred_class, self.adv_label)

            # alpha = self.exponential_weight_for_backward(value=psnr,norm=-1,alpha=1,exp=2, psnr_thresh=25)
            loss = 0
            loss += 1 * loss_CE
            loss += 1 * loss_bc

            logs['rate_real'] = torch.mean(self.detection_mask).item()
            logs['loss'] = loss.item()
            logs['L1'] = loss_L1.item()
            # logs['alpha'] = alpha
            logs['PSNR'] = psnr
            # logs['distill'] = loss_distill.item()
            logs['bc'] = loss_bc.item()
            logs['adv'] = loss_adv.item()

            loss.backward()

            nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()

        ##### printing the images  ######
        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.real_H),
                self.postprocess(selected_patterns),
                self.postprocess(mixed_image),
                self.postprocess(10 * torch.abs(self.real_H - mixed_image)),
                self.postprocess(torch.sigmoid(pred_mask)),
                self.postprocess(masks_GT_full),
                self.postprocess(self.detection_image),
                self.postprocess(torch.sigmoid(pred_detection_mask)),
                self.postprocess(self.detection_mask),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{epoch}_{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

    def feed_aux_data(self, detection_item):
        img, mask = detection_item
        self.detection_image = img.cuda()
        self.detection_mask = mask.cuda()
        if len(self.detection_mask.shape) == 3:
            self.detection_mask = self.detection_mask.unsqueeze(1)


    def validate_IFA_dense_prediction_postprocess(self, step=None, epoch=None):
        ### todo: downgrade model include n/2 real-world examples and n/2 restored examples
        ## self.canny_image is the authentic image in this mode
        self.qf_predict_network.eval()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        tampered_image_original, masks_GT, authentic_image_original = self.real_H_val, self.canny_image_val, self.auth_image

        batch_size = self.real_H_val.shape[0]
        # degrade_tampered, degrade_authentic = tampered_image_original, authentic_image_original
        # degrade_time = np.random.randint(1, 3)
        # for i in range(degrade_time):
        #     degrade_index = np.random.randint(1, 1000)
        #     degrade_tampered = self.benign_attacks_without_simulation(forward_image=degrade_tampered,
        #                                                               index=degrade_index)
        #     degrade_authentic = self.benign_attacks_without_simulation(forward_image=degrade_authentic,
        #                                                                index=degrade_index)

        degraded = torch.cat([tampered_image_original, authentic_image_original], dim=0)
        masks_GT = torch.cat([masks_GT, torch.zeros_like(masks_GT).cuda()], dim=0)
        ## first half: synthesized downgraded images
        # degrade_synthesize, degrade_input = degrade[:batch_size // 2], degrade[batch_size // 2:]

        # with torch.no_grad():
        #     ## second half: restored downgraded images
        #     if self.global_step % 6 == 0:  # 'unet' in self.opt['restoration_model'].lower():
        #         degrade_sum = self.restore_unet(degraded, torch.zeros((1)).cuda())
        #         degrade_sum = self.clamp_with_grad(degrade_sum)
        #         degrade_sum = self.to_jpeg(forward_image=degrade_sum)
        #         # loss = self.l1_loss(degrade_sum, self.real_H)
        #     elif self.global_step % 6 == 1:  # 'invisp' in self.opt['restoration_model'].lower():
        #         degrade_sum = self.restore_invisp(degraded)
        #         degrade_sum = self.clamp_with_grad(degrade_sum)
        #         degrade_sum = self.to_jpeg(forward_image=degrade_sum)
        #         # reverted, _ = self.qf_predict_network(degrade, rev=True)
        #         # loss = self.l1_loss(degrade_sum, self.real_H)  # + self.l1_loss(reverted, degrade.clone().detach())
        #     elif self.global_step % 6 == 2:  # restormer
        #         degrade_sum = self.restore_restormer(degraded)
        #         degrade_sum = self.clamp_with_grad(degrade_sum)
        #         degrade_sum = self.to_jpeg(forward_image=degrade_sum)
        #         # loss = self.l1_loss(degrade_sum, self.real_H)
        #     else:
        #         degrade_sum = degraded

        ## ground-truth: first 0 then 1
        if self.IFA_bc_label is None:
            self.IFA_bc_label = torch.tensor([0.] * batch_size + [1.] * batch_size).unsqueeze(1).cuda()

        with torch.no_grad():
            ## predict PSNR given degrade_sum
            predicted_mask, _ = self.qf_predict_network(degraded, self.timestamp)
            loss = self.bce_with_logit_loss(predicted_mask, masks_GT)

            logs['sum_loss'] = loss.item()

        # use_pre_post_process = np.random.randint(0, 10000) % 2 == 0
        # if use_pre_post_process:
        #     self.real_H_val, psnr = self.global_post_process(original=self.real_H_val)
        #
        # auth_non_tamper, auth_tamper = self.real_H_val[:self.batch_size//3], self.real_H_val[self.batch_size//3:]
        #
        # masks, masks_GT, percent_range = self.mask_generation(modified_input=auth_tamper, index=self.global_step,
        #                                                       percent_range=(0.0, 0.2))
        # masks_GT = torch.cat([self.zero_metric, masks_GT],dim=0)
        #
        #
        # degraded = self.benign_attacks_without_simulation(forward_image=auth_tamper, index=self.global_step)
        #
        # if self.opt['use_restore']:
        #     use_restoration = np.random.randint(0,10000) % 3 == 0
        #     if use_restoration:
        #         with torch.no_grad():
        #
        #             if use_restoration%3==0: #'unet' in self.opt['restoration_model'].lower():
        #                 degraded = self.restore_unet(degraded, self.timestamp)
        #                 degraded = self.clamp_with_grad(degraded)
        #                 # loss = self.l1_loss(predicted, self.real_H)
        #             elif use_restoration%3==1: #'invisp' in self.opt['restoration_model'].lower():
        #                 degraded = self.restore_invisp(degraded)
        #                 degraded = self.clamp_with_grad(degraded)
        #                 # reverted, _ = self.qf_predict_network(degrade, rev=True)
        #                 # loss = self.l1_loss(predicted, self.real_H)  # + self.l1_loss(reverted, degrade.clone().detach())
        #             else: # restormer
        #                 degraded = self.restore_restormer(degraded)
        #                 degraded = self.clamp_with_grad(degraded)
        #                 # loss = self.l1_loss(predicted, self.real_H)
        #
        # mixed = auth_tamper*(1-masks) + degraded*masks
        # ### contain both authentic and tampered
        # mixed_pattern_images = torch.cat([auth_non_tamper, mixed],dim=0)
        #
        #
        # if not use_pre_post_process:
        #     mixed_pattern_images_post, psnr = self.global_post_process(original=mixed_pattern_images)
        # else:
        #     mixed_pattern_images_post = mixed_pattern_images
        #
        #
        # with torch.no_grad():
        #     ## predict PSNR given degrade_sum
        #     predicted_mask = self.qf_predict_network(mixed_pattern_images_post, self.timestamp)
        #     loss = self.bce_with_logit_loss(predicted_mask, masks_GT)
        #
        #     logs['sum_loss'] = loss.item()

        if (self.global_step % 100 == 3 or self.global_step <= 10):
            images = stitch_images(
                # self.postprocess(self.real_H_val),
                self.postprocess(degraded),
                # self.postprocess(10 * torch.abs(self.real_H_val - mixed_pattern_images)),
                # self.postprocess(mixed_pattern_images_post),
                # self.postprocess(10 * torch.abs(mixed_pattern_images_post - mixed_pattern_images)),
                self.postprocess(torch.sigmoid(predicted_mask)),
                self.postprocess(masks_GT),

                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{epoch}_{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}_val.png"
            images.save(name)

        # logs['psnr'] = psnr

        self.global_step = self.global_step + 1

        return logs, None, False

    def global_post_process(self, *, original, index=None, get_label=False, local_compensate=True, global_compensate=True,
                            adjust_color=False):
        psnr_requirement = self.opt['minimum_PSNR_caused_by_attack']
        # use_global_postprocess = np.random.randint(0, 10000) % 2 == 0
        # if use_global_postprocess:
        return self.benign_attack_ndarray_auto_control(forward_image=original,
                                                    index=index,
                                                    psnr_requirement=psnr_requirement,
                                                    get_label=get_label,
                                                    local_compensate=local_compensate,
                                                    global_compensate=global_compensate,
                                                       adjust_color=adjust_color
                                                       )
        # if get_label:
        #     mixed_pattern_images_post, psnr_label = mixed_pattern_images_post
        # else:
        #     mixed_pattern_images_post = original
        #     if get_label:
        #         psnr_label = torch.ones((self.batch_size,1),device=original.device)

        # psnr = self.psnr(self.postprocess(mixed_pattern_images_post), self.postprocess(original)).item()
        # # assert (psnr > psnr_requirement), f"PSNR {psnr} is not allowed! {self.global_step}"
        # psnr = self.opt['max_psnr'] if psnr < 1 else psnr


    def IFA_binary_classification(self, step=None, epoch=None):
        ### todo: downgrade model include n/2 real-world examples and n/2 restored examples
        ## self.canny_image is the authentic image in this mode
        self.qf_predict_network.train()
        self.restore_restormer.eval()
        self.restore_unet.eval()
        self.restore_invisp.eval()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        tampered_image_original, authentic_image_original = self.real_H, self.canny_image

        batch_size = self.real_H.shape[0]
        degrade_tampered, degrade_authentic = tampered_image_original, authentic_image_original
        degrade_time = np.random.randint(1, 3)
        for i in range(degrade_time):
            degrade_index = np.random.randint(1, 1000)
            degrade_tampered = self.benign_attacks_without_simulation(forward_image=degrade_tampered, index=degrade_index)
            degrade_authentic = self.benign_attacks_without_simulation(forward_image=degrade_authentic, index=degrade_index)

        degraded = torch.cat([degrade_tampered, degrade_authentic],dim=0)
        ## first half: synthesized downgraded images
        # degrade_synthesize, degrade_input = degrade[:batch_size // 2], degrade[batch_size // 2:]

        with torch.no_grad():
            ## second half: restored downgraded images
            if self.global_step % 6 == 0:  # 'unet' in self.opt['restoration_model'].lower():
                degrade_sum = self.restore_unet(degraded, torch.zeros((1)).cuda())[0]
                degrade_sum = self.clamp_with_grad(degrade_sum)
                degrade_sum = self.to_jpeg(forward_image=degrade_sum)
                # loss = self.l1_loss(degrade_sum, self.real_H)
            elif self.global_step % 6 == 1:  # 'invisp' in self.opt['restoration_model'].lower():
                degrade_sum = self.restore_invisp(degraded)
                degrade_sum = self.clamp_with_grad(degrade_sum)
                degrade_sum = self.to_jpeg(forward_image=degrade_sum)
                # reverted, _ = self.qf_predict_network(degrade, rev=True)
                # loss = self.l1_loss(degrade_sum, self.real_H)  # + self.l1_loss(reverted, degrade.clone().detach())
            elif self.global_step % 6 == 2:  # restormer
                degrade_sum = self.restore_restormer(degraded)
                degrade_sum = self.clamp_with_grad(degrade_sum)
                degrade_sum = self.to_jpeg(forward_image=degrade_sum)
                # loss = self.l1_loss(degrade_sum, self.real_H)
            else:
                degrade_sum = degraded


            ## ground-truth: first 0 then 1
            if self.IFA_bc_label is None:
                self.IFA_bc_label = torch.tensor([0.]*batch_size+[1.]*batch_size).unsqueeze(1).cuda()


        with torch.enable_grad():
            ## predict PSNR given degrade_sum
            predicted_score = self.qf_predict_network(degrade_sum)
            loss = self.bce_with_logit_loss(predicted_score, self.IFA_bc_label)

            loss.backward()
            nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()

            logs['sum_loss'] = loss.item()

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(tampered_image_original),
                self.postprocess(authentic_image_original),
                self.postprocess(degrade_sum[:batch_size]),
                self.postprocess(degrade_sum[batch_size:]),
                self.postprocess(10 * torch.abs(tampered_image_original - degrade_sum[:batch_size])),
                self.postprocess(10 * torch.abs(authentic_image_original - degrade_sum[batch_size:])),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{epoch}_{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                # self.save(self.global_step, folder='model', network_list=self.save_network_list)
                self.save_network(self.qf_predict_network, 'qf_predict', f"{epoch}_{self.global_step}",
                                  model_path=f'{self.out_space_storage}/model/{self.task_name}/')

        self.global_step = self.global_step + 1

        return logs, None, False

    def predict_IFA_with_reference(self, step=None, epoch=None):
        # >> > triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        # >> > anchor = torch.randn(100, 128, requires_grad=True)
        # >> > positive = torch.randn(100, 128, requires_grad=True)
        # >> > negative = torch.randn(100, 128, requires_grad=True)
        # >> > output = triplet_loss(anchor, positive, negative)
        # >> > output.backward()
        ### todo: the first half of the batch is reserved as positive example, and the rest are modified as negative ones.
        self.qf_predict_network.train()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr
        # batch_size = self.real_H.shape[0]//2
        positive_examples, negative_examples = self.real_H[:].clone(), self.real_H[:].clone()

        attacked_positive, _, _, _ = self.standard_attack_layer(
            modified_input=positive_examples, skip_tamper=True
        )
        attacked_tampered_negative, _, _, _ = self.standard_attack_layer(
            modified_input=negative_examples, skip_tamper=False, percent_range=[0.00, 0.33]
        )

        # anchor = self.qf_predict_network.module.embedding.weight
        # feat_positive = self.qf_predict_network(attacked_positive)
        # feat_negative = self.qf_predict_network(attacked_tampered_negative)
        # loss = self.triplet_loss(anchor, feat_positive, feat_negative)
        losses, feats = self.qf_predict_network(attacked_positive=attacked_positive,attacked_tampered_negative=attacked_tampered_negative)
        loss, pos_similarity, neg_similarity = losses
        feat_positive, feat_pos_anchor, feat_negative, feat_neg_anchor = feats
        # l1_pos = self.l2_loss(feat_positive, anchor)
        # l1_neg = self.l2_loss(feat_negative, anchor)
        loss.backward()
        nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
        self.optimizer_qf.step()
        self.optimizer_qf.zero_grad()

        logs['sum_loss'] = loss.item()
        logs['feat_pos_anchor'] = torch.mean(feat_pos_anchor).item()
        logs['feat_neg_anchor'] = torch.mean(feat_neg_anchor).item()
        # logs['embedding'] = self.qf_predict_network.module.embedding.mean().item()
        logs['pos'] = torch.mean(pos_similarity).item()
        logs['neg'] = torch.mean(neg_similarity).item()

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(positive_examples),
                self.postprocess(negative_examples),
                self.postprocess(attacked_positive),
                self.postprocess(attacked_tampered_negative),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                # self.save(self.global_step, folder='model', network_list=self.save_network_list)
                self.save_network(self.qf_predict_network, 'qf_predict', f"{epoch}_{self.global_step}",
                                  model_path=self.out_space_storage + f'/model/{self.task_name}/')
                # pkl_path = f'{self.out_space_storage}/model/{self.task_name}/{self.global_step}_qf_predict.pkl'
                # with open(pkl_path, 'wb') as f:
                #     pickle.dump({'embedding': self.qf_predict_network.module.embedding}, f)
                #     print("Pickle saved to: {}".format(pkl_path))


        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = positive_examples

        self.global_step = self.global_step + 1

        return logs, None, False

    def predict_IFA_with_mask_prediction(self,step=None):
        ## received: real_H, canny_image
        self.generator.train()
        self.qf_predict_network.train()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        modified_input = self.real_H
        # masks_GT = self.canny_image

        ###### auto-generated tampering and post-processing
        attacked_image, attacked_forward, masks, masks_GT = self.standard_attack_layer(
            modified_input=modified_input
        )

        error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(
            attacked_forward)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
        logs['ERROR'] = error_l1
        attacked_image = attacked_image.detach().contiguous()

        ## todo: label is 0-5, representing 0-5% 5%-10% 10%-15% 15%-20% 25%-30% >30%
        label = 20*torch.mean(masks_GT, dim=[1, 2, 3])
        self.label = torch.where(label>6*torch.ones_like(label), 6*torch.ones_like(label), label).long()


        ## todo: first step: reference recovery
        estimated_mask, mid_feats = self.generator(attacked_image)
        loss_aux = self.bce_with_logit_loss(estimated_mask, masks_GT)

        output = self.qf_predict_network(torch.cat([attacked_image, estimated_mask.detach()], dim=1),
                                      mid_feats_from_recovery=mid_feats)
        loss_ce = self.ce_loss(output, self.label)

        # loss = emd_loss(labels, outputs)

        loss = loss_ce + loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
        nn.utils.clip_grad_norm_(self.generator.parameters(), 1)
        self.optimizer_qf.step()
        self.optimizer_generator.step()
        self.optimizer_qf.zero_grad()
        self.optimizer_generator.zero_grad()

        acc = (output.argmax(dim=1) == self.label).float().mean()
        logs['epoch_accuracy'] = acc
        logs['loss_ce'] = loss_ce
        logs['loss_aux'] = loss_aux

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.real_H),
                self.postprocess(attacked_image),
                self.postprocess(masks_GT),
                self.postprocess(estimated_mask),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)

        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H

        self.global_step = self.global_step + 1

        return logs, None, False


    def train_regression(self, step=None):
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

            self.qf_predict_network.train()
            modified_input = self.real_H
            # masks_GT = self.canny_image

            ###### auto-generated tampering and post-processing
            attacked_image, attacked_forward, masks, masks_GT = self.standard_attack_layer(
                modified_input=modified_input
            )

            error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(
                attacked_forward)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
            logs['ERROR'] = error_l1


            label = torch.mean(masks_GT, dim=[1, 2, 3]).float().unsqueeze(1)

            ###### just post-processing
            # index_for_postprocessing = self.global_step
            # quality_idx = self.get_quality_idx_by_iteration(index=index_for_postprocessing)
            # ## settings for attack
            # kernel = random.choice([3, 5, 7])  # 3,5,7
            # resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
            #                 int(self.random_float(0.5, 2) * self.width_height))
            # skip_robust = np.random.rand() > self.opt['skip_attack_probability']
            # if not skip_robust and self.opt['consider_robost']:
            #
            #     attacked_image, attacked_real_jpeg_simulate, _ = self.benign_attacks(attacked_forward=attacked_image,
            #                                                                          quality_idx=quality_idx,
            #                                                                          index=index_for_postprocessing,
            #                                                                          kernel_size=kernel,
            #                                                                          resize_ratio=resize_ratio
            #                                                                          )
            # else:
            #     attacked_image = attacked_image

            #######
            predictionQA, feat, quan_loss = self.qf_predict_network(attacked_image)


            l_class = self.l2_loss(predictionQA, label)

            l_sum = 0.1*quan_loss+l_class
            logs['CE'] = l_class.item()
            logs['Quan'] = quan_loss.item()

            l_sum.backward()

            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()


        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.real_H),
                self.postprocess(attacked_forward),
                self.postprocess(attacked_image),
                self.postprocess(10 * torch.abs(attacked_image - attacked_forward)),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (self.opt['model_save_period']-1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)

        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H

        self.global_step = self.global_step + 1

        return logs, debug_logs, did_val

    def inference_RR_IFA(self, val_loader, num_images=None):
        self.qf_predict_network.eval()
        epoch_pos, epoch_neg, epoch_loss = 0, 0, 0
        self.global_step = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                self.real_H, _ = batch
                self.real_H = self.real_H.cuda()
                positive_examples, negative_examples = self.real_H[:].clone(), self.real_H[:].clone()

                attacked_positive, _, _, _ = self.standard_attack_layer(
                    modified_input=positive_examples, skip_tamper=True
                )
                attacked_tampered_negative, _, _, _ = self.standard_attack_layer(
                    modified_input=negative_examples, skip_tamper=False, percent_range=[0.00, 0.33]
                )

                # anchor = self.qf_predict_network.module.embedding.weight
                # feat_positive = self.qf_predict_network(attacked_positive)
                # feat_negative = self.qf_predict_network(attacked_tampered_negative)
                # loss = self.triplet_loss(anchor, feat_positive, feat_negative)
                losses, feats = self.qf_predict_network(attacked_positive=attacked_positive, attacked_tampered_negative=attacked_tampered_negative)
                loss, pos_similarity, neg_similarity = losses
                feat_positive, feat_pos_anchor, feat_negative, feat_neg_anchor = feats

                epoch_pos += torch.mean(pos_similarity).item()
                epoch_neg += torch.mean(neg_similarity).item()
                epoch_loss += loss.item()
                print(
                    f"[{idx}] digit: pos {epoch_pos/(idx+1)} neg {epoch_neg/(idx+1)} loss {epoch_loss/(idx+1)}")

                if (idx % 1000 == 3 or self.global_step <= 10):
                    images = stitch_images(
                        self.postprocess(positive_examples),
                        self.postprocess(negative_examples),
                        self.postprocess(attacked_positive),
                        self.postprocess(attacked_tampered_negative),
                        img_per_row=1
                    )

                    name = f"{self.out_space_storage}/images/{self.task_name}/{str(idx).zfill(5)}" \
                           f"_{str(self.rank)}_val.png"
                    images.save(name)

                if self.real_H is not None:
                    if self.previous_images is not None:
                        self.previous_previous_images = self.previous_images.clone().detach()
                    self.previous_images = positive_examples

                self.global_step += 1
                if num_images is not None and idx>=num_images:
                    break

            # print(f"loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n")

    def inference_RR_IFA_regression(self,val_loader):
        epoch_loss = 0
        epoch_accuracy = 0
        self.qf_predict_network.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                self.real_H, _ = batch
                self.real_H = self.real_H.cuda()

                ###### auto-generated tampering and post-processing
                attacked_image, attacked_forward, masks, masks_GT = self.standard_attack_layer(
                    modified_input=self.real_H
                )

                # error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(
                #     attacked_forward)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
                # logs['ERROR'] = error_l1

                label = torch.mean(masks_GT, dim=[1, 2, 3]).float().unsqueeze(1)

                # masks_GT = masks_GT.unsqueeze(1).cuda()

                # label = torch.mean(masks_GT, dim=[1, 2, 3]).float().unsqueeze(1)

                # index_for_postprocessing = self.global_step
                # quality_idx = self.get_quality_idx_by_iteration(index=index_for_postprocessing)
                # ## settings for attack
                # kernel = random.choice([3, 5, 7])  # 3,5,7
                # resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
                #                 int(self.random_float(0.5, 2) * self.width_height))
                #
                # skip_robust = np.random.rand() > self.opt['skip_attack_probability']
                # if not skip_robust and self.opt['consider_robost']:
                #
                #     attacked_image, attacked_real_jpeg_simulate, _ = self.benign_attacks(attacked_forward=attacked_image,
                #                                                                          quality_idx=quality_idx,
                #                                                                          index=index_for_postprocessing,
                #                                                                          kernel_size=kernel,
                #                                                                          resize_ratio=resize_ratio
                #                                                                          )
                # else:
                #     attacked_image = attacked_image

                #######
                predictionQA, feat, quan_loss = self.qf_predict_network(attacked_image)
                l_class = self.l2_loss(predictionQA, label)
                l_sum = 0.1 * quan_loss + l_class

                label_int = (20*label).int()
                predictionQA_int = (20*predictionQA).int()

                acc = (predictionQA_int == label_int).float().mean()
                epoch_accuracy += acc.item() / len(val_loader)
                epoch_loss += l_sum.item() / len(val_loader)
                print(f"[{idx}] digit: pred {predictionQA.item()} gt {label.item()} | class: pred {predictionQA_int.item()} gt {label_int.item()}]")

            print(f"loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n")

            if epoch_accuracy > self.history_accuracy:
                print(f'Saving models and training states.')
                # self.model_save(path='checkpoint/latest', epochs=self.global_step)
                self.save(self.global_step, folder='model', network_list=self.save_network_list)
                self.history_accuracy = epoch_accuracy

