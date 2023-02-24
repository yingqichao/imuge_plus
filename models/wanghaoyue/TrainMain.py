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

class TrainMain(base_IFA):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """
            this file is mode 0

        """
        super(TrainMain, self).__init__(opt, args, train_set, val_set)
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

        self.network_definitions_dense_prediction_postprocess()



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

        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1).cuda()

    # def network_definitions_binary_classification(self):
    #
    #     ### todo: network
    #     if 'cmt' in self.opt['predict_PSNR_model'].lower():
    #         self.qf_predict_network = self.define_CMT()
    #     elif 'resnet' in self.opt['predict_PSNR_model'].lower():
    #         self.qf_predict_network = self.define_convnext(num_classes=1, size='large')
    #     else:
    #         raise NotImplementedError('用作qf_predict的网络名字是不是搞错了？')
    #
    #     self.restore_restormer = self.define_restormer()
    #     self.restore_unet = self.define_ddpm_unet_network(use_bayar=False, use_fft=False)
    #     self.restore_invisp = self.define_invISP(block_num=[4, 4, 4])
    #
    #     model_paths = [
    #         str(self.opt['load_restormer_models']),
    #         str(self.opt['load_unet_models']),
    #         str(self.opt['load_invisp_models']),
    #     ]
    #     models = [
    #         self.restore_restormer, self.restore_unet, self.restore_invisp
    #     ]
    #     folders = [
    #         'Restormer_restoration', 'Unet_restoration', 'Invisp_restoration'
    #     ]
    #
    #     print(f"loading pretrained restoration models")
    #     for idx, model_path in enumerate(model_paths):
    #         model = models[idx]
    #         pretrain = f'{self.out_space_storage}/model/{folders[idx]}/' + model_path
    #         load_path_G = pretrain
    #
    #         print('Loading model for class [{:s}] ...'.format(load_path_G))
    #         if os.path.exists(load_path_G):
    #             self.load_network(load_path_G, model, strict=False)
    #         else:
    #             print('Did not find model for class [{:s}] ...'.format(load_path_G))


    def network_definitions_dense_prediction_postprocess(self):
        self.timestamp = torch.zeros((1)).cuda()
        self.zero_metric = torch.zeros((self.batch_size//3,1,self.width_height,self.width_height)).cuda()

        ### todo: network
        self.qf_predict_network = self.define_ddpm_unet_network(use_bayar=3, use_fft=True, use_SRM=False, use_hierarchical_segment=[1],
                                                                use_classification=[6], use_hierarchical_class=[],
                                                                use_normal_output=[])

        if self.opt['use_restore']:
            self.restore_restormer = self.define_restormer()
            self.restore_unet = self.define_ddpm_unet_network(use_bayar=False, use_fft=False)
            self.restore_invisp = self.define_invISP(block_num=[4, 4, 4])

            model_paths = [
                str(self.opt['load_restormer_models']),
                str(self.opt['load_unet_models']),
                str(self.opt['load_invisp_models']),
            ]
            models = [
                self.restore_restormer, self.restore_unet, self.restore_invisp
            ]
            folders = [
                'Restormer_restoration', 'Unet_restoration', 'Invisp_restoration'
            ]

            print(f"loading pretrained restoration models")
            for idx, model_path in enumerate(model_paths):
                model = models[idx]
                pretrain = f'{self.out_space_storage}/model/{folders[idx]}/' + model_path
                load_path_G = pretrain

                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, model, strict=False)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))


    def feed_aux_data(self, detection_item):
        img, mask = detection_item
        self.detection_image = img.cuda()
        self.detection_mask = mask.cuda()
        if len(self.detection_mask.shape) == 3:
            self.detection_mask = self.detection_mask.unsqueeze(1)

    def IFA_dense_prediction_postprocess(self, step=None, epoch=None):
        ### todo: downgrade model include n/2 real-world examples and n/2 restored examples
        ## self.canny_image is the authentic image in this mode
        self.qf_predict_network.train()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr
        use_post_process = False #np.random.randint(0, 10000) % 3 != 0
        use_pre_post_process = True #np.random.randint(0, 10000) % 2 == 0

        with torch.enable_grad():
            index = np.random.randint(0, 10000) % self.amount_of_benign_attack

            #self.l2_loss(self.real_H, outsize_pattern) # *torch.mean(((255*predicted_mse_map[i]).float()) ** 2)

            ## outside pattern: reserve the last 2 images
            # auth_non_tamper, auth_tamper = self.real_H[3 * self.batch_size // 4:], self.real_H[:3 * self.batch_size // 4]
            index_label = torch.tensor([index] * (self.batch_size), device=self.real_H.device).long()

            compressed_real_H = self.global_post_process(original=self.real_H, get_label=False, index=index)

            outside_pattern = compressed_real_H #torch.cat([compressed_real_H, auth_non_tamper], dim=0)

            # mse_gt = torch.mean((self.real_H - mixed_pattern_images) ** 2, dim=[1, 2, 3]).unsqueeze(1)
            psnr_distort, _ = self.psnr.with_mse(self.postprocess(outside_pattern),
                                                 self.postprocess(self.real_H))
            # labels: <25 25-30 30-35 35-40 40-45 >45
            psnr_label = torch.zeros((self.batch_size, 6), device=self.real_H.device)
            for j in range(self.batch_size):
                i = psnr_distort[j]
                psnr_label[j, int(min(max((i // 5) - 4, 0), 5))] = 1
            # psnr_label = torch.tensor([min(max((i//5)-4,0),5) for i in psnr_distort],device=self.real_H.device).long()
            logs['psnr_distort'] = sum(psnr_distort) / self.batch_size

            x_cls = self.qf_predict_network.module.forward_classification(
                outside_pattern, self.timestamp)
            # predicted_mask, predicted_mse_map = predicted_items[:,:1], predicted_items[:,1:]
            predicted_mse = x_cls[0] #, x_cls[1]
            ### predict global error map
            loss_psnr_regress = emd_loss(predicted_mse, psnr_label)
            # loss_post_class = self.ce_loss(predicted_post_class, index_label)

            loss_global = 0
            # loss += 2 * loss_mask
            loss_global += loss_psnr_regress
            # loss_global += loss_post_class
            loss_global.backward()
            logs['loss_global'] = loss_global.item()
            logs['psnr_loss'] = loss_psnr_regress.item()
            # logs['post_class'] = loss_post_class.item()
            # nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            # self.optimizer_qf.step()
            # self.optimizer_qf.zero_grad()

        # degrade_index = np.random.randint(0, 10000)
        # if self.opt['use_restore'] and degrade_index%10 in [3,6,9]:
        #     use_restoration = np.random.randint(0,10000) % 3
        #     with torch.no_grad():
        #         if use_restoration%3==0: #'unet' in self.opt['restoration_model'].lower():
        #             degraded = self.restore_unet(auth_tamper, self.timestamp)[0]
        #             degraded = self.clamp_with_grad(degraded)
        #         elif use_restoration%3==1: #'invisp' in self.opt['restoration_model'].lower():
        #             degraded = self.restore_invisp(auth_tamper)
        #             degraded = self.clamp_with_grad(degraded)
        #         else: # restormer
        #             degraded = self.restore_restormer(auth_tamper)
        #             degraded = self.clamp_with_grad(degraded)
        # else:
        # degraded = self.benign_attacks_without_simulation(forward_image=auth_tamper, index=degrade_index)


        # if self.opt['do_augment'] and np.random.rand()>0.5:
        #     degraded = self.data_augmentation_on_rendered_rgb(degraded, index=np.random.randint(0,10000), scale=2)


        # if not use_pre_post_process and use_post_process:
        #     mixed_pattern_images_post, psnr_distort, psnr_label = self.global_post_process(original=mixed_pattern_images, get_label=True,
        #                                                                            local_compensate=True, global_compensate=True)
        # else:
            ## inside pattern: reserve the first two images
            # auth_non_tamper, auth_tamper = outside_pattern[:self.batch_size//4], outside_pattern[self.batch_size//4:]
            masks_full, masks_GT_full, percent_range = self.mask_generation(modified_input=self.real_H,
                                                                            index=np.random.randint(0, 10000),
                                                                            percent_range=(0.1, 0.2))
            masks_GT = masks_GT_full #torch.cat([self.zero_metric, masks_GT_full[self.batch_size // 4:]], dim=0)

            index_fg = np.random.randint(0, 10000) % self.amount_of_benign_attack
            while index_fg % self.amount_of_benign_attack == (index % self.amount_of_benign_attack):
                index_fg = np.random.randint(0, 10000) % self.amount_of_benign_attack
            degraded = self.global_post_process(original=self.real_H, get_label=False, index=index_fg,
                                                adjust_color=True)
            inside_pattern = degraded  # torch.cat([auth_non_tamper, degraded], dim=0)

            mixed_pattern_images = self.real_H * (1 - masks_GT) + inside_pattern * masks_GT
            # mixed_pattern_images_post = mixed_pattern_images

            ## predict PSNR given degrade_sum
            middle_feats, hier_class_output, hier_seg_output, x_cls, outs = self.qf_predict_network(mixed_pattern_images, self.timestamp)
            # predicted_mask, predicted_mse_map = predicted_items[:,:1], predicted_items[:,1:]
            # predicted_mse, predicted_post_class = x_cls[0], x_cls[1]
            predicted_mask = hier_seg_output[0]
            ### predict global error map
            # predicted_mse_map = self.clamp_with_grad(predicted_mse_map)
            # loss_psnr_regress = emd_loss(predicted_mse, psnr_label)
            # loss_post_class = self.ce_loss(predicted_post_class, index_label)
            # predicted_mse = torch.mean(predicted_mse_map**2,dim=[1,2,3])

            # predicted_psnr = self.psnr.from_mse_to_psnr(predicted_mse)

            ### predict local noise inconsistency
            loss_mask = self.bce_with_logit_loss(predicted_mask, masks_GT)

            loss_local = 0
            loss_local += loss_mask
            # loss += loss_psnr_regress
            # loss += loss_post_class
            loss_local.backward()
            nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()


            # logs['loss_local'] = loss_local.item()
            logs['mask_loss'] = loss_mask.item()
            # logs['psnr_pred'] = sum(predicted_psnr) / len(predicted_psnr)
            # logs['psnr_diff'] = sum([abs(predicted_psnr[i]-psnr_distort[i]) for i in range(len(psnr_distort))])/len(psnr_distort)

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.real_H),
                self.postprocess(outside_pattern),
                self.postprocess(10 * torch.abs(outside_pattern - self.real_H)),
                self.postprocess(mixed_pattern_images),
                self.postprocess(10 * torch.abs(self.real_H-mixed_pattern_images)),
                # self.postprocess(10*torch.abs(predicted_mse_map)),
                # self.postprocess(mixed_pattern_images_post),
                self.postprocess(torch.sigmoid(predicted_mask)),
                self.postprocess(masks_GT),

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

        with torch.no_grad():
            ## predict PSNR given degrade_sum
            middle_feats, hier_class_output, hier_seg_output, x_cls, outs = self.qf_predict_network(
                degraded, self.timestamp)
            # predicted_mask, predicted_mse_map = predicted_items[:,:1], predicted_items[:,1:]
            # predicted_mse, predicted_post_class = x_cls[0], x_cls[1]
            predicted_mask = hier_seg_output[0]
            loss_mask = self.bce_with_logit_loss(predicted_mask, masks_GT)

            predicted_mse = x_cls[0]  # , x_cls[1]
            ### predict global error map
            _, argmax = torch.max(predicted_mse, 1)
            counts = [0] * 6
            for i in range(self.batch_size):
                counts[argmax[i]] += 1

            logs['sum_loss'] = loss_mask.item()
            for i in range(6):
                logs[f'count_{i}'] = counts[i]
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

