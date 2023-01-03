import torch.nn as nn
# from cycleisp_models.cycleisp import Raw2Rgb
# from MVSS.models.mvssnet import get_mvss
# from MVSS.models.resfcn import ResFCN
# from data.pipeline import pipeline_tensor2image
# import matlab.engine
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
from models.ISP.Modified_invISP import Modified_invISP


class Test(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """
            this file is mode 2

        """
        super(Test, self).__init__(opt, args, train_set, val_set)
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
        print(f"using {self.opt['finetune_detector_name']} as discriminator_mask.")

        self.discriminator_mask = self.define_detector(opt_name='finetune_detector_name') #self.define_my_own_elastic_as_detector()
        self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
                                network_lists=['discriminator_mask'])

        ## inpainting model
        self.define_inpainting_edgeconnect()
        self.define_inpainting_ZITS()
        self.define_inpainting_lama()

    def main_test(self, step=None):
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

                    #### InvISP #####
                    # modified_input_generator, ISP_loss = self.ISP_image_generation_general(network=self.generator,
                    #                                                                        input_raw=input_raw.detach().contiguous(),
                    #                                                                        target=gt_rgb)
                    modified_input_generator = self.generator(input_raw)
                    # ISP_SSIM = - self.ssim_loss(modified_input_generator, gt_rgb)
                    ISP_forward = self.l1_loss(input=modified_input_generator, target=gt_rgb)
                    modified_input_generator = self.clamp_with_grad(modified_input_generator)
                    rev_RAW, _ = self.generator(modified_input_generator, rev=True)

                    ISP_backward = self.l1_loss(input=rev_RAW, target=input_raw)
                    ISP_loss = ISP_backward + ISP_forward

                    modified_input_generator_detach = modified_input_generator.detach()
                    ISP_PSNR = self.psnr(self.postprocess(modified_input_generator_detach), self.postprocess(gt_rgb)).item()
                    logs['ISP_PSNR'] = ISP_PSNR
                    logs['ISP_L1'] = ISP_loss.item()
                    stored_image_generator = modified_input_generator_detach if stored_image_generator is None else \
                        torch.cat([stored_image_generator, modified_input_generator_detach], dim=0)

                    if self.opt['train_isp_networks']:
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

            if self.opt['train_RAW2RAW'] or self.opt['train_detector']:
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


                with torch.no_grad():

                    ### RAW PROTECTION ###

                    modified_raw_one_dim = self.RAW_protection_by_my_own_elastic(input_raw_one_dim=input_raw_one_dim)

                    modified_raw = self.visualize_raw(modified_raw_one_dim, bayer_pattern=bayer_pattern, white_balance=camera_white_balance)
                    RAW_L1 = self.l1_loss(input=modified_raw, target=input_raw)

                    modified_raw = self.clamp_with_grad(modified_raw)

                    RAW_PSNR = self.psnr(self.postprocess(modified_raw), self.postprocess(input_raw)).item()
                    logs['RAW_PSNR'] = RAW_PSNR
                    logs['RAW_L1'] = RAW_L1.item()


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
                    results_list = []
                    buffer_index = self.global_step


                    # while rate_mask < 0.05 or rate_mask >= 0.33:  # prevent too small or too big
                    masks, masks_GT, percent_range = self.mask_generation(modified_input=modified_input,
                                                                          percent_range=(0,0.2),
                                                                          index=self.global_step)

                    kernel = 3  # 3,5,7
                    resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
                                    int(self.random_float(0.5, 2) * self.width_height))

                    tamper_shifted, masks, masks_GT = self.get_shifted_image_for_copymove(forward_image=modified_input,
                                                                                          percent_range=percent_range,
                                                                                          masks=masks)
                    attacked_forward = modified_input * (1 - masks) + tamper_shifted.clone().detach() * masks

                    for i in range(5):
                        self.global_step = i


                        index_for_postprocessing = self.global_step

                        quality_idx = self.get_quality_idx_by_iteration(index=index_for_postprocessing)

                        ###############   TAMPERING   ##################################################################################

                        # attacked_forward, masks, masks_GT = self.copymove(forward_image=modified_input, masks=masks,
                        #                                                   masks_GT=masks_GT,
                        #                                                   percent_range=percent_range)

                        # attacked_forward = self.inpainting_ZITS(forward_image=modified_input,
                        #                                                     masks=masks_GT)
                        attacked_forward = self.clamp_with_grad(attacked_forward)

                        attacked_image, attacked_real_jpeg_simulate, _ = self.benign_attacks(
                            attacked_forward=attacked_forward,
                            quality_idx=quality_idx,
                            index=index_for_postprocessing,
                            kernel_size=kernel,
                            resize_ratio=resize_ratio
                            )

                        error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(attacked_forward)).item() #self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
                        logs['ERROR'] = error_l1


                        pred_resfcn, CE_resfcn = self.detector_predict(model=self.discriminator_mask,
                                                                       attacked_image=attacked_image,
                                                                       opt_name='finetune_detector_name',
                                                                       masks_GT=masks_GT)

                        CE_loss = CE_resfcn
                        logs['CE_ema'] = CE_resfcn.item()

                        results_list.append(pred_resfcn)

                    self.global_step = buffer_index


                #########################    printing the images   #################################################
                anomalies = False  # CE_recall.item()>0.5
                if True:
                    images = stitch_images(
                        self.postprocess(input_raw),
                        ### RAW2RAW
                        self.postprocess(modified_raw),
                        self.postprocess(10 * torch.abs(modified_raw - input_raw)),
                        ### rendered images and protected images
                        self.postprocess(modified_input),
                        self.postprocess(gt_rgb),

                        ### tampering and benign attack
                        self.postprocess(attacked_forward),
                        # self.postprocess(attacked_adjusted),
                        self.postprocess(attacked_image),
                        ### tampering detection
                        # self.postprocess(attacked_cannied),
                        self.postprocess(masks_GT),

                        self.postprocess(results_list[0]),
                        self.postprocess(results_list[1]),
                        self.postprocess(results_list[2]),
                        self.postprocess(results_list[3]),
                        self.postprocess(results_list[4]),
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
