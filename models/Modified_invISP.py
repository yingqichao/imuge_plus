import copy
import logging
import os
import math
import cv2
import torch
import torch.nn as nn

from noise_layers import *
from utils import stitch_images
from .base_model import BaseModel


# import matlab.engine
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
# import contextual_loss as cl
# import contextual_loss.functional as F

class Modified_invISP(BaseModel):
    def __init__(self, opt, args, train_set=None, val_set=None):
        super(Modified_invISP, self).__init__(opt, args, train_set, val_set)


        print(f"network list:{self.network_list}")
        # self.save_network_list = self.network_list
        print(f"Current mode: {self.args.mode}")
        print(f"Function: {self.mode_dict[self.args.mode]}")

        if self.args.mode in [0, 1]:
            ### mode=0: generating protected images (val)
            ### mode=1: tampering localization on generating protected images (val)

            self.network_list = self.default_ISP_networks + self.default_RAW_to_RAW_networks + self.default_detection_networks
            self.save_network_list = []
            self.training_network_list = []
        elif self.args.mode in [2] and "UNet" in self.task_name:
            ### mode=2: regular training (UNet), including ISP, RAW2RAW and localization (train)
            self.network_list = self.default_ISP_networks + self.default_RAW_to_RAW_networks + self.default_detection_networks
            self.save_network_list = self.network_list
            self.training_network_list = ["KD_JPEG", "discriminator_mask"]
        elif self.args.mode in [2] and "elastic" in self.task_name:
            ### mode=2: regular training (our network design), including ISP, RAW2RAW and localization (train)
            self.network_list = self.default_ISP_networks + self.default_RAW_to_RAW_networks + self.default_detection_networks
            self.save_network_list = ["KD_JPEG", "discriminator_mask"]
            self.training_network_list = ["KD_JPEG", "discriminator_mask"]
        elif self.args.mode in [3]:
            ### regular training for ablation (RGB protection)
            self.network_list = ["KD_JPEG", "discriminator_mask"]
            self.save_network_list = self.network_list
            self.training_network_list = self.network_list
        elif self.args.mode in [4]:
            ### OSN performance (val)
            self.network_list = self.default_ISP_networks + self.default_RAW_to_RAW_networks + self.default_detection_networks
                                # + self.default_customized_networks
            if self.opt['activate_OSN'] or self.opt['test_restormer']==2:
                self.network_list.append('localizer')
            self.save_network_list = []
            self.training_network_list = []
        elif self.args.mode in [5]:
            ### train a ISP using restormer for validation (train)
            self.network_list = ['localizer']
            self.save_network_list = self.network_list
            self.training_network_list = self.network_list
        elif self.args.mode in [6]:
            ### train a ISP using restormer for validation (train)
            self.network_list = ['discriminator_mask']
            self.save_network_list = self.network_list
            self.training_network_list = self.network_list
        else:
            raise NotImplementedError("从1012开始，需要指定一下读取哪些模型")

        print(f"network list: {self.network_list}")

        self.out_space_storage = f"{self.opt['name']}/complete_results"

        if 'generator' in self.network_list:
            ####################################################################################################
            # todo: ISP networks will be loaded
            # todo: invISP: generator
            ####################################################################################################
            self.define_ISP_network_training()

            self.model_storage = f'/model/{self.opt["task_name_ISP_model"]}/'

            ### loading ISP
            self.load_space_storage = f"{self.opt['name']}/complete_results"
            self.load_storage = f'/model/{self.opt["task_name_ISP_model"]}/'
            self.model_path = str(self.opt['load_ISP_models'])  # last time: 10999
            load_models = self.opt['load_ISP_models'] > 0
            if load_models:
                print(f"loading tampering/ISP models: {self.network_list}")
                self.pretrain = self.load_space_storage + self.load_storage + self.model_path
                self.reload(self.pretrain, network_list=self.default_ISP_networks)

        if 'discriminator_mask' in self.network_list:

            self.define_tampering_localization_network()

            ### loading discriminator
            self.load_space_storage = f"{self.opt['name']}/complete_results"
            self.load_storage = f'/model/{self.opt["task_name_discriminator_model"]}/'
            self.model_path = str(self.opt['load_discriminator_models'])  # last time: 10999
            load_models = self.opt['load_discriminator_models'] > 0
            if load_models:
                print(f"loading models: {self.network_list}")
                self.pretrain = self.load_space_storage + self.load_storage + self.model_path
                self.reload(self.pretrain, network_list=self.default_detection_networks)

            # self.discriminator = HWMNet(in_chn=3, out_chn=1, wf=32, depth=4, subtask=0,
            #                             style_control=False, use_dwt=False, use_norm_conv=True).cuda()
            #     # UNetDiscriminator(in_channels=3, out_channels=1, residual_blocks=2, use_SRM=False, subtask=self.opt['raw_classes']).cuda() #UNetDiscriminator(use_SRM=False).cuda()
            # self.discriminator = DistributedDataParallel(self.discriminator,
            #                                                   device_ids=[torch.cuda.current_device()],
            #                                                   find_unused_parameters=True)

            # self.scaler_G = torch.cuda.amp.GradScaler()
            # self.scaler_generator = torch.cuda.amp.GradScaler()
            # self.scaler_qf = torch.cuda.amp.GradScaler()

        if 'localizer' in self.network_list:
            self.define_localizer()
            if self.args.mode in [5.0] or self.opt['test_restormer'] == 2:
                ### loading localizer
                self.load_space_storage = f"{self.opt['name']}/complete_results"
                self.load_storage = f'/model/{self.opt["task_name_customized_model"]}/'
                self.model_path = str(self.opt['load_customized_models'])  # last time: 10999
                load_models = self.opt['load_customized_models'] > 0
                if load_models:
                    print(f"loading models: {self.network_list}")
                    self.pretrain = self.load_space_storage + self.load_storage + self.model_path
                    self.reload(self.pretrain, network_list=self.default_customized_networks)

        if 'KD_JPEG' in self.network_list:
            ####################################################################################################
            # todo: RAW2RAW network will be loaded
            # todo:
            ####################################################################################################
            self.define_RAW2RAW_network()

            ### loading RAW2RAW
            self.load_space_storage = f"{self.opt['name']}/complete_results"
            self.load_storage = f'/model/{self.opt["task_name_KD_JPEG_model"]}/'
            self.model_path = str(self.opt['load_RAW_models'])  # last time: 10999
            load_models = self.opt['load_RAW_models'] > 0
            if load_models:
                print(f"loading models: {self.network_list}")
                self.pretrain = self.load_space_storage + self.load_storage + self.model_path
                self.reload(self.pretrain, network_list=self.default_RAW_to_RAW_networks)

            # self.scaler_kd_jpeg = torch.cuda.amp.GradScaler()

        ####################################################################################################
        # todo: Optimizers
        # todo: invISP
        ####################################################################################################
        self.define_optimizers()

        ####################################################################################################
        # todo: Scheduler
        # todo: invISP
        ####################################################################################################
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=118287))

        ####################################################################################################
        # todo: creating dirs
        # todo:
        ####################################################################################################
        self.create_folders_for_the_experiment()

        # ## load states, deprecated, because there can be too many models from different paths.
        # state_path = self.load_space_storage + self.load_storage + '{}.state'.format(self.model_path)
        # if load_state:
        #     print('Loading training state')
        #     if os.path.exists(state_path):
        #         self.resume_training(state_path, self.network_list)
        #     else:
        #         print('Did not find state [{:s}] ...'.format(state_path))



    def feed_data_router(self, batch, mode):
        if mode in [0.0]:
            # self.feed_data_COCO_like(batch, mode='train') # feed_data_COCO_like(batch)
            self.feed_data_ISP(batch, mode='train')
        elif mode in [1.0,6.0]:
            self.feed_data_COCO_like(batch, mode='train')
        elif mode in [2.0,3.0,4.0,5.0]:
            self.feed_data_ISP(batch, mode='train')

    def feed_data_val_router(self, batch, mode):
        if mode in [0.0]:
            self.feed_data_ISP(batch, mode='val')
        elif mode in [1.0,6.0]:
            self.feed_data_COCO_like(batch, mode='val')
        elif mode in [2.0,3.0,4.0,5.0]:
            self.feed_data_ISP(batch, mode='val')


    def feed_data_ISP(self, batch, mode='train'):
        if mode=='train':
            self.real_H = batch['input_raw'].cuda()
            self.label = batch['target_rgb'].cuda()
            self.file_name = batch['file_name']
            self.camera_white_balance = batch['camera_whitebalance'].cuda()
            self.bayer_pattern = batch['bayer_pattern'].cuda()
            self.camera_name = batch['camera_name']
        else:
            self.real_H_val = batch['input_raw'].cuda()
            self.label_val = batch['target_rgb'].cuda()
            self.file_name_val = batch['file_name']
            self.camera_white_balance_val = batch['camera_whitebalance'].cuda()
            self.bayer_pattern_val = batch['bayer_pattern'].cuda()
            self.camera_name_val = batch['camera_name']

    def feed_data_COCO_like(self, batch, mode='train'):
        if mode == 'train':
            img, mask = batch
            self.real_H = img.cuda()
            self.canny_image = mask.unsqueeze(1).cuda()
        else:
            img, mask = batch
            self.real_H_val = img.cuda()
            self.canny_image_val = mask.unsqueeze(1).cuda()

    def optimize_parameters_router(self, mode, step=None):
        if mode == 0.0:
            return self.get_protected_RAW_and_corresponding_images(step=step)
        elif mode==1.0:
            return self.get_predicted_mask(step=step)
        elif mode==2.0:
            return self.optimize_parameters_main(step=step)
        elif mode==3.0:
            return self.optimize_parameters_ablation_on_RAW(step=step)
        elif mode==4.0:
            return self.get_performance_of_OSN(step=step)
        elif mode==5.0:
            return self.train_ISP_using_rstormer(step=step)
        elif mode==6.0:
            return self.train_resfcn(step=step)

    def train_ISP_using_rstormer(self, step=None):
        ####################################################################################################
        # todo: Image Manipulation Detection Network (Downstream task)
        # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
        ####################################################################################################
        #### SYMBOL FOR NOTIFYING THE OUTER VAL LOADER #######
        did_val = False
        if step is not None:
            self.global_step = step

        logs, debug_logs = {}, []
        # self.real_H = self.clamp_with_grad(self.real_H)
        lr = self.get_current_learning_rate()
        logs['lr'] = lr


        #### DIVIDE THE BATCH INTO CLIPS AS MINI-BATCHES ###
        sum_batch_size = self.real_H.shape[0]
        num_per_clip = int(sum_batch_size // self.opt['step_acumulate'])

        for idx_clip in range(self.opt['step_acumulate']):
            ### camera_white_balance SIZE (B,3)
            camera_white_balance = self.camera_white_balance[
                                   idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
            file_name = self.file_name[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip]
            ### bayer_pattern sized (B,1) ranging from [0,3]
            bayer_pattern = self.bayer_pattern[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()

            input_raw_one_dim = self.real_H[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
            gt_rgb = self.label[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
            input_raw = self.visualize_raw(input_raw_one_dim, bayer_pattern=bayer_pattern,
                                           white_balance=camera_white_balance, eval=not self.opt['train_isp_networks'])
            ### NEW BATCH_SIZE AFTER CLIPPING
            batch_size, num_channels, height_width, _ = input_raw.shape
            # input_raw = self.clamp_with_grad(input_raw)


            with torch.enable_grad():

                self.localizer.train()
                ####################################################################################################
                # todo: Image ISP training
                # todo: we first train several nn-based ISP networks BEFORE TRAINING THE PIPELINE
                ####################################################################################################

                ####### UNetDiscriminator ##############
                modified_input_qf_predict, CYCLE_loss = self.ISP_image_generation_general(
                    network=self.localizer,
                    input_raw=input_raw.detach().contiguous(),
                    target=gt_rgb)

                modified_input_qf_predict_detach = self.clamp_with_grad(modified_input_qf_predict.detach())
                CYCLE_PSNR = self.psnr(self.postprocess(modified_input_qf_predict_detach),
                                       self.postprocess(gt_rgb)).item()
                logs['CYCLE_PSNR'] = CYCLE_PSNR
                logs['CYCLE_L1'] = CYCLE_loss.item()


                (CYCLE_loss / self.opt['step_acumulate']).backward()
                # self.scaler_qf.scale(CYCLE_loss).backward()
                if idx_clip % self.opt['step_acumulate'] == self.opt['step_acumulate'] - 1:
                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.localizer.parameters(), 1)
                    self.optimizer_localizer.step()
                    # self.scaler_qf.step(self.optimizer_qf)
                    # self.scaler_qf.update()
                    self.optimizer_localizer.zero_grad()


            if (self.global_step % 1000 == 3 or self.global_step <= 10):
                images = stitch_images(
                    self.postprocess(input_raw),
                    self.postprocess(modified_input_qf_predict_detach),
                    self.postprocess(gt_rgb),
                    img_per_row=1
                )

                name = f"{self.out_space_storage}/isp_images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                       f"_{idx_clip}_ {str(self.rank)}.png"
                print(f'Bayer: {bayer_pattern}. Saving sample {name}')
                images.save(name)

        ####################################################################################################
        # todo: updating the training stage
        # todo:
        ####################################################################################################
        ######## Finally ####################
        if self.global_step % (self.opt['model_save_period']) == (self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)

        self.global_step = self.global_step + 1

        # print(logs)
        # print(debug_logs)
        return logs, debug_logs, did_val


    def optimize_parameters_main(self, step=None):
        ####################################################################################################
        # todo: Image Manipulation Detection Network (Downstream task)
        # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
        ####################################################################################################
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
            #### DIVIDE THE BATCH INTO CLIPS AS MINI-BATCHES ###
            sum_batch_size = self.real_H.shape[0]
            num_per_clip = int(sum_batch_size//self.opt['step_acumulate'])

            for idx_clip in range(self.opt['step_acumulate']):
                ### camera_white_balance SIZE (B,3)
                camera_white_balance = self.camera_white_balance[
                                       idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
                file_name = self.file_name[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip]
                ### bayer_pattern sized (B,1) ranging from [0,3]
                bayer_pattern = self.bayer_pattern[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()

                input_raw_one_dim = self.real_H[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
                gt_rgb = self.label[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
                input_raw = self.visualize_raw(input_raw_one_dim, bayer_pattern=bayer_pattern,
                                               white_balance=camera_white_balance, eval=not self.opt['train_isp_networks'])
                ### NEW BATCH_SIZE AFTER CLIPPING
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
                        ####################################################################################################
                        # todo: Image ISP training
                        # todo: we first train several nn-based ISP networks BEFORE TRAINING THE PIPELINE
                        ####################################################################################################

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
                            torch.cat((stored_image_qf_predict, modified_input_qf_predict_detach), dim=0)

                        # self.optimizer_generator.zero_grad()
                        if self.opt['train_isp_networks']:
                            (CYCLE_loss / self.opt['step_acumulate']).backward()
                            # self.scaler_qf.scale(CYCLE_loss).backward()
                            if idx_clip % self.opt['step_acumulate'] == self.opt['step_acumulate'] - 1:
                                if self.train_opt['gradient_clipping']:
                                    nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
                                self.optimizer_qf.step()
                                # self.scaler_qf.step(self.optimizer_qf)
                                # self.scaler_qf.update()
                                self.optimizer_qf.zero_grad()


                        #### HWMNET ####
                        modified_input_netG, THIRD_loss = self.ISP_image_generation_general(network=self.netG,
                                                                                               input_raw=input_raw.detach().contiguous(),
                                                                                               target=gt_rgb)

                        modified_input_netG_detach = self.clamp_with_grad(modified_input_netG.detach())
                        PIPE_PSNR = self.psnr(self.postprocess(modified_input_netG_detach),self.postprocess(gt_rgb)).item()
                        logs['PIPE_PSNR'] = PIPE_PSNR
                        logs['PIPE_L1'] = THIRD_loss.item()
                        ## STORE THE RESULT FOR LATER USE
                        stored_image_netG = modified_input_netG_detach if stored_image_netG is None else \
                            torch.cat((stored_image_netG, modified_input_netG_detach), dim=0)

                        if self.opt['train_isp_networks']:
                            # self.optimizer_generator.zero_grad()
                            (THIRD_loss/self.opt['step_acumulate']).backward()
                            # self.scaler_G.scale(THIRD_loss).backward()
                            if idx_clip % self.opt['step_acumulate'] == self.opt['step_acumulate'] - 1:
                                if self.train_opt['gradient_clipping']:
                                    nn.utils.clip_grad_norm_(self.netG.parameters(), 1)
                                self.optimizer_G.step()
                                # self.scaler_G.step(self.optimizer_G)
                                # self.scaler_G.update()
                                self.optimizer_G.zero_grad()

                        #### InvISP #####
                        modified_input_generator, ISP_loss = self.ISP_image_generation_general(network=self.generator,
                                                                                               input_raw=input_raw.detach().contiguous(),
                                                                                               target=gt_rgb)

                        modified_input_generator_detach = modified_input_generator.detach()
                        ISP_PSNR = self.psnr(self.postprocess(modified_input_generator_detach), self.postprocess(gt_rgb)).item()
                        logs['ISP_PSNR'] = ISP_PSNR
                        logs['ISP_L1'] = ISP_loss.item()
                        stored_image_generator = modified_input_generator_detach if stored_image_generator is None else \
                            torch.cat((stored_image_generator, modified_input_generator_detach), dim=0)

                        if self.opt['train_isp_networks']:
                            ####################################################################################################
                            # todo: Grad Accumulation
                            # todo: added 20220919, steo==0, do not update, step==1 update
                            ####################################################################################################
                            # self.optimizer_generator.zero_grad()
                            (ISP_loss/self.opt['step_acumulate']).backward()
                            # self.scaler_generator.scale(ISP_loss).backward()
                            if idx_clip % self.opt['step_acumulate']==self.opt['step_acumulate']-1:
                                if self.train_opt['gradient_clipping']:
                                    nn.utils.clip_grad_norm_(self.generator.parameters(), 1)
                                self.optimizer_generator.step()
                                # self.scaler_generator.step(self.optimizer_generator)
                                # self.scaler_generator.update()
                                self.optimizer_generator.zero_grad()

                ####################################################################################################
                # todo: emptying cache to save memory
                # todo: https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/25
                ####################################################################################################
                # torch.cuda.empty_cache()


                if self.opt['train_isp_networks'] and (self.global_step % 200 == 3 or self.global_step <= 10):
                    images = stitch_images(
                        self.postprocess(input_raw),
                        self.postprocess(modified_input_generator_detach),
                        self.postprocess(modified_input_qf_predict_detach),
                        self.postprocess(modified_input_netG_detach),
                        self.postprocess(gt_rgb),
                        img_per_row=1
                    )

                    name = f"{self.out_space_storage}/isp_images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                           f"_{idx_clip}_ {str(self.rank)}.png"
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
                self.discriminator_mask.train() if "discriminator_mask" in self.training_network_list else self.discriminator_mask.eval()
                self.discriminator.eval()
                self.qf_predict_network.eval()
                # self.localizer.train()

                for idx_clip in range(self.opt['step_acumulate']):

                    input_raw_one_dim = self.real_H[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
                    file_name = self.file_name[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip]
                    camera_name = self.camera_name[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip]
                    gt_rgb = self.label[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
                    ### tensor sized (B,3)
                    camera_white_balance = self.camera_white_balance[
                                           idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
                    ### tensor sized (B,1) ranging from [0,3]
                    bayer_pattern = self.bayer_pattern[
                                    idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()

                    input_raw = self.visualize_raw(input_raw_one_dim, bayer_pattern=bayer_pattern,
                                                   white_balance=camera_white_balance)
                    batch_size, num_channels, height_width, _ = input_raw.shape

                    with torch.enable_grad():
                        ####################################################################################################
                        # todo: Generation of protected RAW
                        # todo: next, we protect RAW for tampering detection
                        ####################################################################################################
                        #### condition for RAW2RAW ####
                        # label_array = np.random.choice(range(self.opt['raw_classes']),batch_size)
                        # label_control = torch.tensor(label_array).long().cuda()
                        # label_input = torch.tensor(label_array).float().cuda().unsqueeze(1)
                        # label_input = label_input / self.opt['raw_classes']

                        ### RAW PROTECTION ###
                        if self.task_name == "my_own_elastic":
                            modified_raw_one_dim = self.RAW_protection_by_my_own_elastic(input_raw_one_dim=input_raw_one_dim)

                        else:
                            modified_raw_one_dim = self.KD_JPEG(input_raw_one_dim)
                        # raw_reversed, _ = self.KD_JPEG(modified_raw_one_dim, rev=True)

                        modified_raw = self.visualize_raw(modified_raw_one_dim, bayer_pattern=bayer_pattern, white_balance=camera_white_balance)
                        RAW_L1 = self.l1_loss(input=modified_raw, target=input_raw)
                        # RAW_L1_REV = self.l1_loss(input=raw_reversed, target=input_raw_one_dim)
                        modified_raw = self.clamp_with_grad(modified_raw)

                        RAW_PSNR = self.psnr(self.postprocess(modified_raw), self.postprocess(input_raw)).item()
                        logs['RAW_PSNR'] = RAW_PSNR
                        logs['RAW_L1'] = RAW_L1.item()

                        ####################################################################################################
                        # todo: RAW2RGB pipelines
                        # todo: note: our goal is that the rendered rgb by the protected RAW should be close to that rendered by unprotected RAW
                        # todo: thus, we are not let the ISP network approaching the ground-truth RGB.
                        ####################################################################################################

                        ### model selection，shuffle the gts to enable color control
                        if self.global_step%3==0:
                            isp_model_0, isp_model_1 = self.generator, self.qf_predict_network
                            stored_list_0, stored_list_1 = stored_image_generator, stored_image_qf_predict
                        elif self.global_step%3==1:
                            isp_model_0, isp_model_1 = self.netG, self.qf_predict_network
                            stored_list_0, stored_list_1 = stored_image_netG, stored_image_qf_predict
                        else: #if self.global_step%3==2:
                            isp_model_0, isp_model_1 = self.netG, self.generator
                            stored_list_0, stored_list_1 = stored_image_netG, stored_image_generator

                        #### invISP AS SUBSEQUENT ISP####
                        modified_input_0 = isp_model_0(modified_raw)
                        if self.opt['use_gamma_correction']:
                            modified_input_0 = self.gamma_correction(modified_input_0)
                        tamper_source_0 = stored_list_0[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
                        ISP_L1_0 = self.l1_loss(input=modified_input_0, target=tamper_source_0)
                        ISP_SSIM_0 = - self.ssim_loss(modified_input_0, tamper_source_0)
                        ISP_percept_0, ISP_style_0 = self.perceptual_loss(modified_input_0, tamper_source_0, with_gram=True)
                        modified_input_0 = self.clamp_with_grad(modified_input_0)

                        modified_input_1 = isp_model_1(modified_raw)
                        if self.opt['use_gamma_correction']:
                            modified_input_1 = self.gamma_correction(modified_input_1)
                        tamper_source_1 = stored_list_1[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
                        ISP_L1_1 = self.l1_loss(input=modified_input_1, target=tamper_source_1)
                        ISP_SSIM_1 = - self.ssim_loss(modified_input_1, tamper_source_1)
                        ISP_percept_1, ISP_style_1 = self.perceptual_loss(modified_input_1, tamper_source_1, with_gram=True)
                        modified_input_1 = self.clamp_with_grad(modified_input_1)

                        # #### HWMNET AS SUBSEQUENT ISP####
                        # modified_input_2 = self.netG(modified_raw)
                        # if self.opt['use_gamma_correction']:
                        #     modified_input_2 = self.gamma_correction(modified_input_2)
                        # tamper_source_2 = stored_image_netG[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()
                        # ISP_L1_2 = self.l1_loss(input=modified_input_2, target=tamper_source_2)
                        # ISP_SSIM_2 = - self.ssim_loss(modified_input_2, tamper_source_2)
                        # # ISP_percept_2, ISP_style_2 = self.perceptual_loss(modified_input_2, tamper_source_2, with_gram=True)
                        # # ISP_style_2 = self.style_loss(modified_input_2, tamper_source_2)
                        # modified_input_2 = self.clamp_with_grad(modified_input_2)



                        # #### my_own_pipeline ON PROTECTED RAW ######
                        # numpy_rgb = isp_tensor2image(raw_image=input_raw, metadata=None, file_name=file_name[0],
                        #                              camera_name=camera_name[0])
                        #
                        # modified_input_1 = torch.zeros_like(modified_input_2)
                        # for idx_pipeline in range(num_per_clip):
                        #     metadata = self.train_set.metadata_list[file_name[idx_pipeline]]
                        #     flip_val = metadata['flip_val']
                        #     metadata = metadata['metadata']
                        #     # 在metadata中加入要用的flip_val和camera_name
                        #     metadata['flip_val'] = flip_val
                        #     metadata['camera_name'] = camera_name
                        #     # [B C H W]->[H,W]
                        #     raw_1 = modified_raw_one_dim[idx_pipeline].permute(1, 2, 0).squeeze(2)
                        #     numpy_rgb = pipeline_tensor2image(raw_image=raw_1, metadata=metadata, input_stage='normal', output_stage='gamma')
                        #     modified_input_1[idx_pipeline:idx_pipeline+1] = torch.from_numpy(np.ascontiguousarray(np.transpose(numpy_rgb, (2, 0, 1)))).contiguous().float()
                        # #### my_own_pipeline ON ORIGINAL RAW ######
                        # tamper_source_1 = torch.zeros_like(modified_input_2)
                        # for idx_pipeline in range(num_per_clip):
                        #     metadata = self.train_set.metadata_list[file_name[idx_pipeline]]
                        #     flip_val = metadata['flip_val']
                        #     metadata = metadata['metadata']
                        #     # 在metadata中加入要用的flip_val和camera_name
                        #     metadata['flip_val'] = flip_val
                        #     metadata['camera_name'] = camera_name
                        #     # [B C H W]->[H,W]
                        #     raw_1 = input_raw_one_dim[idx_pipeline].permute(1, 2, 0).squeeze(2)
                        #     numpy_rgb = pipeline_tensor2image(raw_image=raw_1, metadata=metadata, input_stage='normal', output_stage='gamma')
                        #     tamper_source_1[idx_pipeline:idx_pipeline + 1] = torch.from_numpy(
                        #         np.ascontiguousarray(np.transpose(numpy_rgb, (2, 0, 1)))).contiguous().float()

                        ####################################################################################################
                        # todo: doing mixup on the images
                        # todo: note: our goal is that the rendered rgb by the protected RAW should be close to that rendered by unprotected RAW
                        # todo: thus, we are not let the ISP network approaching the ground-truth RGB.
                        ####################################################################################################
                        skip_the_second = np.random.rand() > 0.8
                        alpha_0 = 1.0 if skip_the_second else np.random.rand()
                        alpha_1 = 1 - alpha_0
                        # alpha_0 = np.random.rand()*0.66
                        # alpha_1 = np.random.rand()*0.66
                        # alpha_1 = min(alpha_1,1-alpha_0)
                        # alpha_1 = max(0, alpha_1)
                        # alpha_2 = 1 - alpha_0 - alpha_1

                        modified_input = alpha_0*modified_input_0
                        # modified_input += alpha_2*modified_input_2
                        modified_input += alpha_1*modified_input_1
                        tamper_source = alpha_0*tamper_source_0
                        # tamper_source += alpha_2*tamper_source_2
                        tamper_source += alpha_1*tamper_source_1
                        tamper_source = tamper_source.detach()

                        # ISP_L1_sum = self.l1_loss(input=modified_input, target=tamper_source)
                        # ISP_SSIM_sum = - self.ssim_loss(modified_input, tamper_source)

                        ### collect the protected images
                        modified_input = self.clamp_with_grad(modified_input)
                        tamper_source = self.clamp_with_grad(tamper_source)
                        PSNR_DIFF = self.psnr(self.postprocess(modified_input), self.postprocess(tamper_source)).item()
                        ISP_PSNR = self.psnr(self.postprocess(modified_input), self.postprocess(gt_rgb)).item()
                        logs['PSNR_DIFF'] = PSNR_DIFF
                        logs['ISP_PSNR_NOW'] = ISP_PSNR

                        collected_protected_image = modified_input_netG_detach if collected_protected_image is None else \
                            torch.cat((collected_protected_image, modified_input.detach()), dim=0)

                        ####################################################################################################
                        # todo: attack layer
                        # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
                        ####################################################################################################
                        attacked_image, attacked_adjusted, attacked_forward, masks, masks_GT = self.standard_attack_layer(
                            modified_input=modified_input, gt_rgb=gt_rgb, logs=logs, idx_clip=idx_clip,
                            num_per_clip=num_per_clip
                        )

                        # ERROR = attacked_image-attacked_forward
                        error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(attacked_forward)).item() #self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
                        logs['ERROR'] = error_l1
                        ####################################################################################################
                        # todo: Image Manipulation Detection Network (Downstream task)
                        # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
                        ####################################################################################################
                        # _, pred_mvss = self.netG(attacked_image)
                        # CE_MVSS = self.bce_with_logit_loss(pred_mvss, masks_GT)
                        # logs['CE_MVSS'] = CE_MVSS.item()
                        # pred_mantra = self.localizer(attacked_image)
                        # CE_mantra = self.bce_with_logit_loss(pred_mantra, masks_GT)
                        # logs['CE_mantra'] = CE_mantra.item()
                        ### why contiguous? https://discuss.pytorch.org/t/runtimeerror-set-sizes-and-strides-is-not-allowed-on-a-tensor-created-from-data-or-detach/116910/10

                        ####################################################################################################
                        # todo: cropping
                        # todo: cropped: original-sized cropped image, scaled_cropped: resized cropped image, masks, masks_GT
                        ####################################################################################################

                        if self.opt['conduct_cropping'] and np.random.rand() > 0.66:
                            # print("crop...")
                            locs, cropped, attacked_image = self.cropping_mask_generation(
                                forward_image=attacked_image, min_rate=0.7, max_rate=1.0, logs=logs)
                            h_start, h_end, w_start, w_end = locs
                            _, _, masks_GT = self.cropping_mask_generation(forward_image=masks_GT, locs=locs, logs=logs)

                        ### get mean and std of mask_GT
                        std_gt, mean_gt = torch.std_mean(masks_GT, dim=(2, 3))

                        ### UPDATE discriminator_mask AND LATER AFFECT THE MOMENTUM LOCALIZER
                        if "discriminator_mask" in self.training_network_list:
                            pred_resfcn, post_resfcn = self.discriminator_mask(attacked_image.detach().contiguous())

                            CE_resfcn = self.bce_loss(torch.sigmoid(pred_resfcn), masks_GT)
                            l1_resfcn = self.bce_loss(torch.sigmoid(post_resfcn), masks_GT)

                            CE_loss = CE_resfcn + l1_resfcn
                            logs['CE'] = CE_resfcn.item()
                            logs['CE_ema'] = CE_resfcn.item()
                            logs['CEL1'] = l1_resfcn.item()
                            logs['l1_ema'] = l1_resfcn.item()
                            # logs['Mean'] = l1_mean.item()
                            # logs['Std'] = l1_std.item()
                            (CE_loss/self.opt['step_acumulate']).backward()
                            if idx_clip % self.opt['step_acumulate'] == self.opt['step_acumulate']-1:
                                # self.optimizer_generator.zero_grad()
                                # loss.backward()
                                # self.scaler_generator.scale(loss).backward()
                                if self.train_opt['gradient_clipping']:
                                    nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
                                self.optimizer_discriminator_mask.step()
                                self.optimizer_discriminator_mask.zero_grad()


                        ### USING THE MOMENTUM LOCALIZER TO TRAIN THE PIPELINE
                        if "KD_JPEG" in self.training_network_list:
                            pred_resfcn, post_resfcn = self.discriminator_mask(attacked_image)

                            CE_resfcn = self.bce_loss(torch.sigmoid(pred_resfcn), masks_GT)
                            l1_resfcn = self.bce_loss(torch.sigmoid(post_resfcn), masks_GT)
                            # l1_mean = self.l2_loss(mean_pred, mean_gt)
                            # l1_std = self.l2_loss(std_pred, std_gt)

                            # CE_control = self.CE_loss(pred_control, label_control)
                            CE_loss = CE_resfcn
                            logs['CE_ema'] = CE_resfcn.item()
                            logs['l1_ema'] = l1_resfcn.item()
                            # logs['Mean'] = l1_mean.item()
                            # logs['Std'] = l1_std.item()


                            loss = 0
                            loss_l1 = self.opt['L1_hyper_param'] * (ISP_L1_0+ISP_L1_1)/2
                            loss += loss_l1
                            hyper_param_raw = self.opt['RAW_L1_hyper_param'] if (ISP_PSNR < self.opt['psnr_thresh']) else self.opt['RAW_L1_hyper_param']/5
                            loss += hyper_param_raw * RAW_L1
                            loss_ssim = self.opt['ssim_hyper_param'] * (ISP_SSIM_0+ISP_SSIM_1)/2
                            loss += loss_ssim
                            hyper_param_percept = self.opt['perceptual_hyper_param'] if (ISP_PSNR < self.opt['psnr_thresh']) else self.opt['perceptual_hyper_param'] / 4
                            loss_percept = hyper_param_percept * (ISP_percept_0+ISP_percept_1)/2
                            loss += loss_percept
                            loss_style = self.opt['style_hyper_param'] * (ISP_style_0 +ISP_style_1) / 2
                            # loss += loss_style
                            hyper_param = self.opt['CE_hyper_param'] if (ISP_PSNR>=self.opt['psnr_thresh']) else self.opt['CE_hyper_param']/10
                            loss += hyper_param * CE_loss  # (CE_MVSS+CE_mantra+CE_resfcn)/3

                            logs['ISP_SSIM_NOW'] = -loss_ssim.item()
                            logs['Percept'] = loss_percept.item()
                            logs['Style'] = loss_style.item()
                            logs['Gray'] = loss_l1.item()
                            logs['loss'] = loss.item()

                            ####################################################################################################
                            # todo: Grad Accumulation
                            # todo: added 20220919, steo==0, do not update, step==1 update
                            ####################################################################################################
                            (loss/self.opt['step_acumulate']).backward()
                            # self.scaler_kd_jpeg.scale(loss).backward()
                            if idx_clip % self.opt['step_acumulate'] == self.opt['step_acumulate']-1:
                                # self.optimizer_generator.zero_grad()
                                # loss.backward()
                                # self.scaler_generator.scale(loss).backward()
                                if self.train_opt['gradient_clipping']:
                                    nn.utils.clip_grad_norm_(self.KD_JPEG.parameters(), 1)
                                    # nn.utils.clip_grad_norm_(self.netG.parameters(), 1)
                                    # nn.utils.clip_grad_norm_(self.localizer.parameters(), 1)
                                    # nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
                                    # nn.utils.clip_grad_norm_(self.generator.parameters(), 1)
                                self.optimizer_KD_JPEG.step()
                                # self.optimizer_discriminator_mask.step()
                                # self.scaler_kd_jpeg.step(self.optimizer_KD_JPEG)
                                # self.scaler_kd_jpeg.step(self.optimizer_G)
                                # self.scaler_kd_jpeg.step(self.optimizer_localizer)
                                # self.scaler_kd_jpeg.step(self.optimizer_discriminator_mask)
                                # self.scaler_kd_jpeg.update()
                                self.optimizer_KD_JPEG.zero_grad()
                                self.optimizer_G.zero_grad()
                                self.optimizer_localizer.zero_grad()
                                self.optimizer_discriminator_mask.zero_grad()
                                self.optimizer_generator.zero_grad()
                                self.optimizer_qf.zero_grad()

                    ####################################################################################################
                    # todo: printing the images
                    # todo: invISP
                    ####################################################################################################
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
                            # self.postprocess(modified_input_2),
                            # self.postprocess(tamper_source_2),
                            # self.postprocess(10 * torch.abs(modified_input_2 - tamper_source_2)),
                            # self.postprocess(inpainted_image),
                            self.postprocess(gt_rgb),

                            ### RAW2RGB
                            # self.postprocess(modified_input),
                            # self.postprocess(tamper_source),
                            # self.postprocess(10 * torch.abs(modified_input - tamper_source)),
                            ### tampering and benign attack
                            self.postprocess(attacked_forward),
                            self.postprocess(attacked_adjusted),
                            self.postprocess(attacked_image),
                            self.postprocess(10 * torch.abs(attacked_forward - attacked_image)),
                            ### tampering detection
                            self.postprocess(masks_GT),
                            # self.postprocess(torch.sigmoid(pred_mvss)),
                            # self.postprocess(10 * torch.abs(masks_GT - torch.sigmoid(pred_mvss))),
                            # self.postprocess(torch.sigmoid(pred_mantra)),
                            # self.postprocess(10 * torch.abs(masks_GT - torch.sigmoid(pred_mantra))),
                            self.postprocess(torch.sigmoid(pred_resfcn)),
                            self.postprocess(torch.sigmoid(post_resfcn)),
                            # self.postprocess(refined_resfcn),
                            # norm_pred, adaptive_pred, diff_pred
                            # self.postprocess(norm_pred),
                            # self.postprocess(adaptive_pred),
                            # self.postprocess(diff_pred),
                            # self.postprocess(10 * torch.abs(masks_GT - torch.sigmoid(pred_resfcn))),
                            img_per_row=1
                        )

                        name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                                   f"_{idx_clip}_ {str(self.rank)}.png"
                        print('\nsaving sample ' + name)
                        images.save(name)

                ####################################################################################################
                # todo: doing ema average
                # todo:
                ####################################################################################################
                # if self.begin_using_momentum:
                #     print("Moving average...")
                #     self._momentum_update_key_encoder()

                ####################################################################################################
                # todo: inference single image for testing
                # todo:
                ####################################################################################################
                # if self.global_step % 199 == 3:
                #     did_val = True
                #     self.inference_single_image()#input_raw_one_dim=input_raw_one_dim, input_raw=input_raw, gt_rgb=gt_rgb,
                #                                 # camera_white_balance=camera_white_balance, file_name=file_name,
                #                                 # camera_name=camera_name, bayer_pattern=bayer_pattern)

        ####################################################################################################
        # todo: updating the training stage
        # todo:
        ####################################################################################################
        ######## Finally ####################
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

    def optimize_parameters_ablation_on_RAW(self, step=None):
        ####################################################################################################
        # todo: Image Manipulation Detection Network (Downstream task)
        # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
        ####################################################################################################
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
            #### DIVIDE THE BATCH INTO CLIPS AS MINI-BATCHES ###
            sum_batch_size = self.real_H.shape[0]
            num_per_clip = int(sum_batch_size//self.opt['step_acumulate'])


            self.KD_JPEG.train()
            self.discriminator_mask.train()

            for idx_clip in range(self.opt['step_acumulate']):

                gt_rgb = self.label[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous()

                batch_size, num_channels, height_width, _ = gt_rgb.shape

                with torch.enable_grad():
                    ####################################################################################################
                    # todo: Generation of protected RAW
                    # todo: next, we protect RAW for tampering detection
                    ####################################################################################################

                    ### RGB PROTECTION ###
                    modified_input = self.baseline_generate_protected_rgb(gt_rgb=gt_rgb)

                    RAW_L1 = self.l1_loss(input=modified_input, target=gt_rgb)
                    ISP_percept, ISP_style = self.perceptual_loss(modified_input, gt_rgb,
                                                                      with_gram=True)

                    modified_input = self.clamp_with_grad(modified_input)

                    RAW_PSNR = self.psnr(self.postprocess(modified_input), self.postprocess(gt_rgb)).item()
                    logs['RAW_PSNR'] = RAW_PSNR
                    logs['RAW_L1'] = RAW_L1.item()
                    logs['Percept'] = ISP_percept.item()


                    collected_protected_image = modified_input.detach() if collected_protected_image is None else \
                        torch.cat((collected_protected_image, modified_input.detach()), dim=0)

                    ####################################################################################################
                    # todo: attack layer
                    # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
                    ####################################################################################################
                    attacked_image, attacked_adjusted, attacked_forward, masks, masks_GT = self.standard_attack_layer(
                        modified_input=modified_input, gt_rgb=gt_rgb, logs=logs, idx_clip=idx_clip, num_per_clip=num_per_clip
                    )

                    # ERROR = attacked_image-attacked_forward
                    error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(attacked_forward)).item() #self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
                    logs['ERROR'] = error_l1
                    ####################################################################################################
                    # todo: Image Manipulation Detection Network (Downstream task)
                    # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
                    ####################################################################################################
                    ### get mean and std of mask_GT
                    std_gt, mean_gt = torch.std_mean(masks_GT, dim=(2, 3))

                    ### UPDATE discriminator_mask AND LATER AFFECT THE MOMENTUM LOCALIZER

                    pred_resfcn, post_resfcn = self.discriminator_mask(attacked_image.detach().contiguous())
                    CE_resfcn = self.bce_loss(torch.sigmoid(pred_resfcn), masks_GT)
                    l1_resfcn = self.bce_loss(torch.sigmoid(post_resfcn), masks_GT)

                    # CE_control = self.CE_loss(pred_control, label_control)
                    CE_loss = CE_resfcn + l1_resfcn #+ l1_resfcn + 10 * (l1_mean + l1_std)  # + CE_control
                    logs['CE'] = CE_resfcn.item()
                    logs['CEL1'] = l1_resfcn.item()
                    # logs['l1_ema'] = l1_resfcn.item()
                    # logs['Mean'] = l1_mean.item()
                    # logs['Std'] = l1_std.item()
                    # logs['CE_control'] = CE_control.item()
                    (CE_loss/self.opt['step_acumulate']).backward()
                    if idx_clip % self.opt['step_acumulate'] == self.opt['step_acumulate']-1:
                        # self.optimizer_generator.zero_grad()
                        # loss.backward()
                        # self.scaler_generator.scale(loss).backward()
                        if self.train_opt['gradient_clipping']:
                            nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
                        self.optimizer_discriminator_mask.step()
                        self.optimizer_discriminator_mask.zero_grad()


                    ### USING THE MOMENTUM LOCALIZER TO TRAIN THE PIPELINE

                    pred_resfcn, post_resfcn = self.discriminator_mask(attacked_image)
                    CE_resfcn = self.bce_loss(torch.sigmoid(pred_resfcn), masks_GT)
                    l1_resfcn = self.bce_loss(torch.sigmoid(post_resfcn), masks_GT)

                    # CE_control = self.CE_loss(pred_control, label_control)
                    CE_loss = CE_resfcn + l1_resfcn  # + l1_resfcn + 10 * (l1_mean + l1_std)  # + CE_control
                    logs['CE_ema'] = CE_resfcn.item()
                    logs['l1_ema'] = l1_resfcn.item()


                    loss = 0
                    loss += self.opt['L1_hyper_param'] * RAW_L1
                    hyper_param_percept = self.opt['perceptual_hyper_param'] if (RAW_PSNR < self.opt['psnr_thresh']) else self.opt['perceptual_hyper_param'] / 4
                    loss_percept = hyper_param_percept * ISP_percept
                    loss += loss_percept
                    hyper_param = self.opt['CE_hyper_param'] if (RAW_PSNR>=self.opt['psnr_thresh']) else self.opt['CE_hyper_param']/10
                    loss += hyper_param * CE_loss  # (CE_MVSS+CE_mantra+CE_resfcn)/3
                    logs['loss'] = loss.item()

                    ####################################################################################################
                    # todo: Grad Accumulation
                    # todo: added 20220919, steo==0, do not update, step==1 update
                    ####################################################################################################
                    (loss/self.opt['step_acumulate']).backward()
                    # self.scaler_kd_jpeg.scale(loss).backward()
                    if idx_clip % self.opt['step_acumulate'] == self.opt['step_acumulate']-1:
                        # self.optimizer_generator.zero_grad()
                        # loss.backward()
                        # self.scaler_generator.scale(loss).backward()
                        if self.train_opt['gradient_clipping']:
                            nn.utils.clip_grad_norm_(self.KD_JPEG.parameters(), 1)

                        self.optimizer_KD_JPEG.step()
                        self.optimizer_KD_JPEG.zero_grad()
                        self.optimizer_discriminator_mask.zero_grad()


                ####################################################################################################
                # todo: printing the images
                # todo: invISP
                ####################################################################################################
                anomalies = False  # CE_recall.item()>0.5
                if anomalies or self.global_step % 200 == 3 or self.global_step <= 10:
                    images = stitch_images(

                        self.postprocess(modified_input),

                        self.postprocess(gt_rgb),

                        self.postprocess(attacked_forward),
                        self.postprocess(attacked_adjusted),
                        self.postprocess(attacked_image),
                        self.postprocess(10 * torch.abs(attacked_forward - attacked_image)),
                        ### tampering detection
                        self.postprocess(masks_GT),

                        self.postprocess(torch.sigmoid(pred_resfcn)),
                        # self.postprocess(refined_resfcn),
                        # norm_pred, adaptive_pred, diff_pred
                        # self.postprocess(norm_pred),
                        # self.postprocess(adaptive_pred),
                        # self.postprocess(diff_pred),
                        # self.postprocess(10 * torch.abs(masks_GT - torch.sigmoid(pred_resfcn))),
                        img_per_row=1
                    )

                    name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                               f"_{idx_clip}_ {str(self.rank)}.png"
                    print('\nsaving sample ' + name)
                    images.save(name)

        ####################################################################################################
        # todo: updating the training stage
        # todo:
        ####################################################################################################
        ######## Finally ####################
        if self.global_step % 1000 == 999 or self.global_step == 9:
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

    @torch.no_grad()
    def inference_single_image(self):
        pass

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

        ####################################################################################################
        # todo: loading model: discriminator_mask is our model, localizer is OSN
        # todo: IMPORTANT: restormer cannot be loaded simultaneously with OSN network, because they share the same variable
        ####################################################################################################
        test_model = self.discriminator_mask if "localizer" not in self.opt['using_which_model_for_test'] else self.localizer
        test_model.eval()

        ####################################################################################################
        # todo: if the model requires image protection?
        ####################################################################################################

        ### generate gt_rgb using ISP instead of directly loading
        gt_rgb = self.render_image_using_ISP(input_raw=input_raw, input_raw_one_dim=input_raw_one_dim, gt_rgb=gt_rgb,
                                     file_name=file_name, camera_name=camera_name)

        ### generate non-tampered protected (or not) image
        if "localizer" not in self.opt['using_which_model_for_test']:
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
                non_tampered_image = self.render_image_using_ISP(input_raw=input_raw, input_raw_one_dim=input_raw_one_dim, gt_rgb=gt_rgb,
                                     file_name=file_name, camera_name=camera_name)

            ## RGB protection ##
            else:
                self.KD_JPEG.eval()
                # print("rgb protection")
                non_tampered_image = self.baseline_generate_protected_rgb(gt_rgb=gt_rgb)
        ## test on OSN, skip image protection
        else:
            # print('test osn/cat/mvss/resfcn')
            non_tampered_image = gt_rgb

            # print("remember to remove this gaussian blur, we use it to beat CAT!!!")
            # non_tampered_image = self.median_blur(non_tampered_image)

        RGB_PSNR = self.psnr(self.postprocess(non_tampered_image), self.postprocess(gt_rgb)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
        logs['RGB_PSNR'] = RGB_PSNR

        # for i in range(batch_size):
        #     step = step % 130
        #     file_name = "%05d.png" % ((step * batch_size + i) % 758)
        ####################################################################################################
        # todo: get tampering source and mask
        ####################################################################################################
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
            ## tampered image
            test_input = non_tampered_image * (1 - mask_GT) + tamper_source * mask_GT



        else:
            #### auto generated
            if self.previous_protected is not None:
                self.previous_protected = self.label[0:1]
            self.previous_images = self.label[0:1]

            percent_range = (0.05, 0.2) if self.using_copy_move() else (0.05, 0.25)
            masks, masks_GT = self.mask_generation(modified_input=gt_rgb,
                                                   percent_range=percent_range, logs=logs)

            self.previous_protected = gt_rgb

            ### conduct tampering
            test_input, masks, mask_GT = self.tampering(
                forward_image=gt_rgb, masks=masks, masks_GT=masks_GT,
                modified_input=non_tampered_image, percent_range=percent_range, logs=logs,
                idx_clip=None, num_per_clip=None, index=self.opt['inference_tamper_index'],
            )


        ### attacks generate them all
        # attack_lists = [
        #     (None, None, None), (0, None, None), (1, None, None), (2, None, None),
        #     (3, 18, None), (3, 14, None), (3, 10, None), (4, None, None),
        #     (None, None, 0), (None, None, 1), (None, None, 2), (None, None, 3),
        # ]
        attack_lists = [
            (0, 20, 1),  (1, 20, 3), (3, 18, 0),
            (None,None,None),(None,None,0),(None,None,1),
            (0, 20, None), (0, 20, 2),
            (1, 20, None), (1, 20, 0),
            (2, 20, None), (2, 20, 1), (2, 20, 2),
            (3, 10, None), (3, 10, 3), (3, 10, 0),
            (3, 14, None), (3, 14, 2), (3, 14, 1),
            (3, 18, None), (3, 18, 3),
            (4, 20, None), (4, 20, 1), (4, 20, 2),
        ]

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

        ####################################################################################################
        # todo: ISP attack
        # todo:
        ####################################################################################################
        if do_augment is not None:
            modified_adjusted = self.data_augmentation_on_rendered_rgb(modified_input, index=do_augment)
        else:
            modified_adjusted = modified_input

        attacked_forward = modified_adjusted

        ####################################################################################################
        # todo: Benign attacks
        # todo: including JPEG compression Gaussian Blurring, Median blurring and resizing
        ####################################################################################################
        if do_attack is not None:
            if quality_idx is None:
                if do_attack % 8 in {0,1,2,4,6,7}:
                    quality_idx = np.random.randint(20, 21)
                else:
                    quality_idx = np.random.randint(10, 21)
            attacked_image = self.benign_attacks(attacked_forward=attacked_forward, logs=logs,
                                                 quality_idx=quality_idx, index=do_attack)
        else:
            attacked_image = attacked_forward

        # ERROR = attacked_image-attacked_forward
        error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(
            attacked_forward)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
        logs['ERROR'] = error_l1
        ####################################################################################################
        # todo: Image Manipulation Detection Network (Downstream task)
        # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
        ####################################################################################################
        if "localizer" in self.opt['using_which_model_for_test']:
            if 'CAT' in self.opt['using_which_model_for_test']:
                pred_resfcn = target_model(attacked_image.detach().contiguous(), None)
                import torch.nn.functional as F
                pred_resfcn = F.interpolate(pred_resfcn, size=(512, 512), mode='bilinear')
                pred_resfcn = F.softmax(pred_resfcn, dim=1)
                _, pred_resfcn = torch.split(pred_resfcn, 1, dim=1)
            elif 'MVSS' in self.opt['using_which_model_for_test']:
                _, pred_resfcn = target_model(attacked_image.detach().contiguous())
                pred_resfcn = torch.sigmoid(pred_resfcn)
            elif 'Resfcn' or 'Mantra' in self.opt['using_which_model_for_test']:
                pred_resfcn = target_model(attacked_image.detach().contiguous())
                pred_resfcn = torch.sigmoid(pred_resfcn)
            else:
                pred_resfcn = target_model(attacked_image.detach().contiguous())
        else:
            pred_resfcn = target_model(attacked_image.detach().contiguous())
            if isinstance(pred_resfcn, (tuple)):
                # pred_resfcn,  _ = pred_resfcn
                _, pred_resfcn = pred_resfcn
            pred_resfcn = torch.sigmoid(pred_resfcn)

        # refined_resfcn, std_pred, mean_pred = post_pack
        # CE_resfcn = self.bce_loss(torch.sigmoid(pred_resfcn), masks_GT)
        # # l1_resfcn = self.bce_loss(self.clamp_with_grad(refined_resfcn), masks_GT)
        # logs['CE'] = CE_resfcn.item()
        # # logs['CEL1'] = l1_resfcn.item()
        # pred_resfcn = torch.sigmoid(pred_resfcn)
        CE_resfcn = self.bce_loss(pred_resfcn, masks_GT)
        # l1_resfcn = self.bce_loss(self.clamp_with_grad(refined_resfcn), masks_GT)
        logs['CE'] = CE_resfcn.item()
        # logs['CEL1'] = l1_resfcn.item()


        # refined_resfcn_bn = torch.where(refined_resfcn > 0.5, 1.0, 0.0)

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
            name = f"{self.out_space_storage}/test_predicted_masks/{self.task_name}"
            print('saving sample ' + name)
            for image_no in range(batch_size):
                # self.print_this_image(modified_input[image_no],
                #                       f"{name}/{str(step).zfill(5)}_{filename_append}tampered_{self.model_path}_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")

                self.print_this_image(pred_resfcn[image_no],
                                      f"{name}/{str(step).zfill(5)}_{image_no}_{filename_append}pred_ce_{self.model_path}_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                self.print_this_image(pred_resfcn_bn[image_no],
                                      f"{name}/{str(step).zfill(5)}_{image_no}_{filename_append}pred_cebn_{self.model_path}_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                # self.print_this_image(refined_resfcn[image_no],
                #                       f"{name}/{str(step).zfill(5)}_{filename_append}pred_L1_{self.model_path}_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                # self.print_this_image(refined_resfcn_bn[image_no],
                #                       f"{name}/{str(step).zfill(5)}_{filename_append}pred_L1bn_{self.model_path}_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                self.print_this_image(attacked_image[image_no],
                                      f"{name}/{str(step).zfill(5)}_{image_no}_{filename_append}tamper_{self.model_path}_{str(do_attack)}_{str(quality_idx)}_{str(do_augment)}.png")
                self.print_this_image(masks_GT[image_no],
                                      f"{name}/{str(step).zfill(5)}_{image_no}_gt.png")
                print("tampering localization saved at:{}".format(f"{name}/{str(step).zfill(5)}_{image_no}"))

        return logs, (pred_resfcn), False

    @torch.no_grad()
    def get_protected_RAW_and_corresponding_images(self, step=None):
        ####################################################################################################
        # todo: inference single image
        # todo: what is tamper_source? used for simulated inpainting, only activated if self.global_step%3==2
        ####################################################################################################

        input_raw_one_dim = self.real_H_val
        file_name = self.file_name_val
        camera_name = self.camera_name_val
        gt_rgb = self.label_val
        ### tensor sized (B,3)
        camera_white_balance = self.camera_white_balance_val
        ### tensor sized (B,1) ranging from [0,3]
        bayer_pattern = self.bayer_pattern_val

        input_raw = self.visualize_raw(input_raw_one_dim, bayer_pattern=bayer_pattern,
                                       white_balance=camera_white_balance)
        batch_size, num_channels, height_width, _ = input_raw.shape

        logs= {}
        logs['lr'] = 0

        self.KD_JPEG.eval()
        self.netG.eval()
        self.qf_predict_network.eval()
        self.generator.eval()
        ### RAW PROTECTION ###
        modified_raw_one_dim = self.RAW_protection_by_my_own_elastic(input_raw_one_dim=input_raw_one_dim)
        # raw_reversed, _ = self.KD_JPEG(modified_raw_one_dim, rev=True)

        modified_raw = self.visualize_raw(modified_raw_one_dim, bayer_pattern=bayer_pattern,
                                          white_balance=camera_white_balance)
        RAW_L1 = self.l1_loss(input=modified_raw, target=input_raw)
        modified_raw = self.clamp_with_grad(modified_raw)

        RAW_PSNR = self.psnr(self.postprocess(modified_raw), self.postprocess(input_raw)).item()
        logs['RAW_PSNR'] = RAW_PSNR
        logs['RAW_L1'] = RAW_L1.item()

        ####################################################################################################
        # todo: RAW2RGB pipelines
        ####################################################################################################
        #### invISP AS SUBSEQUENT ISP####
        modified_input_0 = self.generator(modified_raw)
        if self.opt['use_gamma_correction']:
            modified_input_0 = self.gamma_correction(modified_input_0)
        modified_input_0 = self.clamp_with_grad(modified_input_0)

        original_0 = self.generator(input_raw)
        if self.opt['use_gamma_correction']:
            original_0 = self.gamma_correction(original_0)
        original_0 = self.clamp_with_grad(original_0)
        RAW_PSNR = self.psnr(self.postprocess(original_0), self.postprocess(modified_input_0)).item()
        logs['RGB_PSNR_0'] = RAW_PSNR

        modified_input_1 = self.qf_predict_network(modified_raw)
        if self.opt['use_gamma_correction']:
            modified_input_1 = self.gamma_correction(modified_input_1)
        modified_input_1 = self.clamp_with_grad(modified_input_1)

        original_1 = self.qf_predict_network(input_raw)
        if self.opt['use_gamma_correction']:
            original_1 = self.gamma_correction(original_1)
        original_1 = self.clamp_with_grad(original_1)
        RAW_PSNR = self.psnr(self.postprocess(original_1), self.postprocess(modified_input_1)).item()
        logs['RGB_PSNR_1'] = RAW_PSNR

        modified_input_2 = self.netG(modified_raw)
        if self.opt['use_gamma_correction']:
            modified_input_2 = self.gamma_correction(modified_input_2)
        modified_input_2 = self.clamp_with_grad(modified_input_2)

        original_2 = self.qf_predict_network(input_raw)
        if self.opt['use_gamma_correction']:
            original_2 = self.gamma_correction(original_2)
        original_2 = self.clamp_with_grad(original_2)
        RAW_PSNR = self.psnr(self.postprocess(original_2), self.postprocess(modified_input_2)).item()
        logs['RGB_PSNR_2'] = RAW_PSNR


        name = f"{self.out_space_storage}/test_protected_images/{self.task_name}"
        # print('\nsaving sample ' + name)
        for image_no in range(batch_size):
            if self.opt['inference_save_image']:
                self.print_this_image(modified_raw[image_no], f"{name}/{str(step).zfill(5)}_protect_raw.png")
                self.print_this_image(input_raw[image_no], f"{name}/{str(step).zfill(5)}_ori_raw.png")
                # self.print_this_image((10*torch.abs(input_raw[image_no]-modified_raw[image_no])).unsqueeze(0),
                #                       f"{name}/{str(step).zfill(5)}_diff_raw.png")
                self.print_this_image(modified_input_0[image_no], f"{name}/{str(step).zfill(5)}_0.png")
                self.print_this_image(original_0[image_no], f"{name}/{str(step).zfill(5)}_0_ori.png")
                # self.print_this_image((10 * torch.abs(modified_input_0[image_no] - original_0[image_no])),
                #                       f"{name}/{str(step).zfill(5)}_0_diff.png")
                self.print_this_image(modified_input_1[image_no], f"{name}/{str(step).zfill(5)}_1.png")
                self.print_this_image(original_1[image_no], f"{name}/{str(step).zfill(5)}_1_ori.png")
                # self.print_this_image((10 * torch.abs(modified_input_1[image_no] - original_1[image_no])),
                #                       f"{name}/{str(step).zfill(5)}_1_diff.png")
                self.print_this_image(modified_input_2[image_no], f"{name}/{str(step).zfill(5)}_2.png")
                self.print_this_image(original_2[image_no], f"{name}/{str(step).zfill(5)}_2_ori.png")
                # self.print_this_image((10 * torch.abs(modified_input_2[image_no] - original_2[image_no])),
                #                       f"{name}/{str(step).zfill(5)}_2_diff.png")
                self.print_this_image(gt_rgb[image_no], f"{name}/{str(step).zfill(5)}_gt.png")
                np.save(f"{name}/{str(step).zfill(5)}_gt", modified_raw.detach().cpu().numpy())

                print("Saved:{}".format(f"{name}/{str(step).zfill(5)}"))

            if self.opt['inference_do_subsequent_prediction']:
                logs_pred_accu = {}
                for idx_isp in range(3):
                    source_image = eval(f"modified_input_{idx_isp}")[image_no:image_no+1]
                    ### get tampering source and mask
                    if self.opt['inference_load_real_world_tamper']:
                        file_name = f"{str(step).zfill(5)}_{idx_isp}_{str(self.rank)}.png"
                        folder_name = f'/groupshare/ISP_results/xxhu_test/{self.task_name}/FORGERY_{idx_isp}/'
                        mask_file_name = f"{str(step).zfill(5)}_0_{str(self.rank)}.png"
                        mask_folder_name = f'/groupshare/ISP_results/xxhu_test/{self.task_name}/MASK/'
                        # print(f"reading {folder_name+file_name}")
                        img_GT = cv2.imread(folder_name+file_name, cv2.IMREAD_COLOR)
                        # img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]
                        # print(f"reading {mask_folder_name + file_name}")
                        mask_GT = cv2.imread(mask_folder_name+mask_file_name, cv2.IMREAD_GRAYSCALE)

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

                        tamper_source = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float().unsqueeze(0).cuda()

                        test_input = source_image * (1 - mask_GT) + tamper_source * mask_GT

                    else: ## using simulated tampering

                        #### tampering source from the training set

                        if self.previous_protected is not None:
                            self.previous_protected = self.label[0:1]
                        self.previous_images = self.label[0:1]


                        percent_range = (0.05, 0.2) if self.using_copy_move() else (0.05, 0.25)
                        masks, masks_GT = self.mask_generation(modified_input=source_image,
                                                               percent_range=percent_range, logs=logs)

                        test_input, masks, mask_GT = self.tampering(
                            forward_image=gt_rgb, masks=masks, masks_GT=masks_GT,
                            modified_input=source_image, percent_range=percent_range, logs=logs,
                            idx_clip=None, num_per_clip=None, index=self.opt['inference_tamper_index'],
                        )

                        self.previous_protected = source_image.clone().detach()



                    ### attacks generate them all
                    attack_lists = [
                        (None, None, None), (0, None, None), (1, None, None), (2, None, None), (3, 18, None), (3, 14, None), (4, None, None),
                        (None, None, 0), (None, None, 1), (None, None, 2), (None, None, 3),
                    ]
                    # attack_lists = [
                    #     (None,None,None),(None,None,0),(None,None,1),
                    #     (0, 20, None), (0, 20, 2), (0, 20, 1),
                    #     (1, 20, None), (1, 20, 3), (1, 20, 0),
                    #     (2, 20, None), (2, 20, 1), (2, 20, 2),
                    #     (3, 10, None), (3, 10, 3), (3, 10, 0),
                    #     (3, 14, None), (3, 14, 2), (3, 14, 1),
                    #     (3, 18, None), (3, 18, 0), (3, 18, 3),
                    #     (4, 20, None), (4, 20, 1), (4, 20, 2),
                    # ]
                    begin_idx = self.opt['inference_benign_attack_begin_idx']
                    for idx_attacks in range(begin_idx, begin_idx+1): # len(attack_lists)
                        do_attack, quality_idx, do_augment = attack_lists[idx_attacks]
                        logs_pred, pred_resfcn, _ = self.get_predicted_mask(modified_input=test_input,
                                                                            masks_GT=mask_GT, do_attack=do_attack,
                                                                            quality_idx=quality_idx,
                                                                            do_augment=do_augment,
                                                                            step=step,
                                                                            filename_append=str(idx_isp),
                                                                            save_image=self.opt['inference_save_image']
                                                                            )
                        if len(logs_pred_accu)==0:
                            logs_pred_accu.update(logs_pred)
                        else:
                            for key in logs_pred:
                                logs_pred_accu[key] += logs_pred[key]

                for key in logs_pred_accu:
                    logs_pred_accu[key] = logs_pred_accu[key]/3/1
                logs.update(logs_pred_accu)

        return logs, (modified_raw, modified_input_0, modified_input_1, modified_input_2), True


    def train_resfcn(self, step=None):
        self.discriminator_mask.train()

        logs, debug_logs = {}, []

        self.real_H = self.clamp_with_grad(self.real_H)
        batch_size, num_channels, height_width, _ = self.real_H.shape
        lr = self.get_current_learning_rate()
        logs['lr'] = lr


        if not (self.previous_images is None or self.previous_previous_images is None):

            with torch.enable_grad(): #cuda.amp.autocast():

                attacked_image, attacked_adjusted, attacked_forward, masks, masks_GT = self.standard_attack_layer(modified_input=self.real_H, gt_rgb=self.real_H, logs=logs,
                                           idx_clip=None, num_per_clip=None)

                ##############################################
                # todo: Image Manipulation Detection Network (Downstream task)
                # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
                ####################################################################################################

                pred_resfcn = self.discriminator_mask(attacked_image.detach().contiguous())
                CE_resfcn = self.bce_with_logit_loss(pred_resfcn, masks_GT)

                logs['CE'] = CE_resfcn.item()


                self.optimizer_discriminator_mask.zero_grad()
                CE_resfcn.backward()
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
                self.optimizer_discriminator_mask.step()
                self.optimizer_discriminator_mask.zero_grad()

                if self.global_step % 1000 == 3 or self.global_step <= 10:
                    images = stitch_images(
                        self.postprocess(self.real_H),
                        self.postprocess(attacked_image),
                        self.postprocess(attacked_adjusted),
                        self.postprocess(attacked_forward),
                        self.postprocess(torch.sigmoid(pred_resfcn)),
                        self.postprocess(masks_GT),
                        img_per_row=1
                    )

                    name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                           f"_0_ {str(self.rank)}.png"
                    print('\nsaving sample ' + name)
                    images.save(name)


        ####################################################################################################
        # todo: updating the training stage
        # todo:
        ####################################################################################################
        ######## Finally ####################
        if self.global_step % (self.opt['model_save_period']) == (self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.network_list)
        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H
            self.previous_protected = self.real_H
        self.global_step = self.global_step + 1

        # print(logs)
        # print(debug_logs)
        return logs, debug_logs, False


    def reload(self, pretrain, network_list=['netG', 'localizer']):
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
                    self.load_network(load_path_G, self.KD_JPEG, strict=True)
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

        if 'discriminator' in network_list:
            load_path_G = pretrain + "_discriminator.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.discriminator, strict=True)
                else:
                    print('Did not find momentum model for class [{:s}] ... we load the discriminator_mask instead'.format(load_path_G))
                    load_path_G = pretrain + "_discriminator_mask.pth"
                    print('Loading model for class [{:s}] ...'.format(load_path_G))
                    if os.path.exists(load_path_G):
                        self.load_network(load_path_G, self.discriminator_mask, strict=True)
                    else:
                        print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'qf_predict_network' in network_list:
            load_path_G = pretrain + "_qf_predict.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.qf_predict_network, strict=True)
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

    def save(self, iter_label, folder='model', network_list=['netG', 'localizer']):
        if 'netG' in network_list:
            self.save_network(self.netG, 'netG', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'localizer' in network_list:
            self.save_network(self.localizer, 'localizer', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'KD_JPEG' in network_list:
            self.save_network(self.KD_JPEG, 'KD_JPEG', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'discriminator_mask' in network_list:
            self.save_network(self.discriminator_mask, 'discriminator_mask', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'discriminator' in network_list:
            self.save_network(self.discriminator, 'discriminator', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'qf_predict_network' in network_list:
            self.save_network(self.qf_predict_network, 'qf_predict', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'generator' in network_list:
            self.save_network(self.generator, 'generator', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')


if __name__ == '__main__':
    pass