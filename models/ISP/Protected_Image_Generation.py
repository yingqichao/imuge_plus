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
from models.ISP.Modified_invISP import Modified_invISP
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from data.demo_test import Bayer_demosaic


class Protected_Image_Generation(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """

            this file is mode 0

        """
        super(Protected_Image_Generation, self).__init__(opt, args, train_set, val_set)
        self.bilinear_demosaic = Bayer_demosaic(512)
        self.train_set = train_set
        self.val_set = val_set
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        self.network_list = self.default_ISP_networks + self.default_RAW_to_RAW_networks
        self.save_network_list = []
        self.training_network_list = []

        ### ISP networks
        self.define_ISP_network_training()
        self.load_model_wrapper(folder_name='ISP_folder', model_name='load_ISP_models',
                                network_lists=self.default_ISP_networks, strict=False)
        ### RAW2RAW network
        self.define_RAW2RAW_network()
        self.load_model_wrapper(folder_name='protection_folder', model_name='load_RAW_models',
                                network_lists=self.default_RAW_to_RAW_networks)
        ### detector networks: 这个可能需要改一下，要看到底加载哪个模型，现在是MPF
        # print("using my_own_elastic as discriminator_mask.")
        # self.discriminator_mask = self.define_CATNET()
        # self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
        #                         network_lists=['discriminator_mask'])

        ## inpainting model
        self.define_inpainting_edgeconnect()
        self.define_inpainting_ZITS()
        self.define_inpainting_lama()



    def get_error_map(self, pred, filepath):
        width = pred.shape[1]  # in pixels
        fig = plt.figure(frameon=False)
        dpi = 40  # fig.dpi
        fig.set_size_inches(width / dpi, ((width * pred.shape[0]) / pred.shape[1]) / dpi)
        sns.heatmap(pred*10, vmin=0, cbar=True, cmap='jet', )
        plt.axis('off')
        plt.savefig(filepath, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close(fig)


    @torch.no_grad()
    def get_protected_RAW_and_corresponding_images(self, step=None):
        ##########################    inference single image   ########################
        ### what is tamper_source? used for simulated inpainting, only activated if self.global_step%3==2
        tamper_index = self.opt['tamper_index']
        postprocess_index = self.opt['postprocess_index']

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

        logs = {}
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

        ###########################    RAW2RGB pipelines    ########################################
        #### invISP AS SUBSEQUENT ISP####
        # invISP
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

        # unet
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

        # # rawpy
        # modified_input_2 = self.pipeline_ISP_gathering(modified_raw_one_dim=modified_raw_one_dim,
        #                                                   file_name=file_name, gt_rgb=gt_rgb, using_rawpy=True, camera_name=camera_name)
        # modified_input_2 = self.clamp_with_grad(modified_input_2)
        #
        # original_2 = self.pipeline_ISP_gathering(modified_raw_one_dim=input_raw_one_dim, file_name=file_name,
        #                                          gt_rgb=gt_rgb, using_rawpy=True, camera_name=camera_name)
        # original_2 = self.clamp_with_grad(original_2)
        #
        # # tradISP
        # modified_input_3 = self.pipeline_ISP_gathering(modified_raw_one_dim=modified_raw_one_dim,
        #                                                file_name=file_name, gt_rgb=gt_rgb)
        # modified_input_3 = self.clamp_with_grad(modified_input_3)
        #
        # original_3 = self.pipeline_ISP_gathering(modified_raw_one_dim=input_raw_one_dim, file_name=file_name,
        #                                          gt_rgb=gt_rgb)
        # original_3 = self.clamp_with_grad(original_3)


        # modified_input_2 = self.netG(modified_raw)
        # if self.opt['use_gamma_correction']:
        #     modified_input_2 = self.gamma_correction(modified_input_2)
        # modified_input_2 = self.clamp_with_grad(modified_input_2)
        #
        # original_2 = self.qf_predict_network(input_raw)
        # if self.opt['use_gamma_correction']:
        #     original_2 = self.gamma_correction(original_2)
        # original_2 = self.clamp_with_grad(original_2)
        # RAW_PSNR = self.psnr(self.postprocess(original_2), self.postprocess(modified_input_2)).item()
        # logs['RGB_PSNR_2'] = RAW_PSNR

        ###############   TAMPERING   ##################################################################################
        attacked_forward = None
        if tamper_index is not None:

            masks, masks_GT, percent_range = self.mask_generation(modified_input=modified_input_1,
                                                                  percent_range=(0.0,0.15), index=self.global_step)

            if not (tamper_index in self.opt['simulated_splicing_indices'] and self.previous_previous_images is None):
                attacked_forward, _, _ = self.tampering_RAW(
                    masks=masks, masks_GT=masks_GT, gt_rgb=gt_rgb,
                    modified_input=modified_input_1, percent_range=percent_range, index=tamper_index
                )

            self.previous_previous_images = gt_rgb

        ###############   Post-process   ##################################################################################
        attacked_image = None
        if attacked_forward is not None and postprocess_index is not None:
            kernel = 7
            resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
                            int(self.random_float(0.5, 2) * self.width_height))
            quality_idx = 18

            attacked_image, _, _ = self.benign_attacks(attacked_forward=attacked_forward,
                                                                                 quality_idx=quality_idx,
                                                                                 index=postprocess_index,
                                                                                 kernel_size=kernel,
                                                                                 resize_ratio=resize_ratio
                                                                                 )

        ################################ print images ###############################
        ori_raw_visualized = self.bilinear_demosaic.bilinear_demosaic(input_raw_one_dim, bayer_pattern)
        modified_raw_visualized = self.bilinear_demosaic.bilinear_demosaic(modified_raw_one_dim, bayer_pattern)
        name = f"{self.out_space_storage}/hand_forged_images/test_rebuttal/"
        # print('\nsaving sample ' + name)
        for image_no in range(batch_size):
            if self.opt['inference_save_image']:
                self.print_this_image(modified_raw_visualized[image_no], f"{name}/{str(step).zfill(5)}_protect_raw.png")
                self.print_this_image(ori_raw_visualized[image_no], f"{name}/{str(step).zfill(5)}_ori_raw.png")
                # diff_raw_path = f"{name}/{str(step).zfill(5)}_diff_raw.png"
                # self.get_error_map((modified_raw_one_dim[image_no]-input_raw_one_dim[image_no]).squeeze().cpu().numpy(),
                #                    diff_raw_path)

                self.print_this_image((50*torch.abs(ori_raw_visualized[image_no]-modified_raw_visualized[image_no])),
                                      f"{name}/{str(step).zfill(5)}_diff_raw.png")

                self.print_this_image(modified_input_0[image_no], f"{name}/{str(step).zfill(5)}_0.png")
                self.print_this_image(original_0[image_no], f"{name}/{str(step).zfill(5)}_0_ori.png")

                self.print_this_image((20 * torch.abs(modified_input_0[image_no] - original_0[image_no])),
                                      f"{name}/{str(step).zfill(5)}_0_diff.png")

                self.print_this_image(modified_input_1[image_no], f"{name}/{str(step).zfill(5)}_1.png")
                self.print_this_image(original_1[image_no], f"{name}/{str(step).zfill(5)}_1_ori.png")

                self.print_this_image((20 * torch.abs(modified_input_1[image_no] - original_1[image_no])),
                                      f"{name}/{str(step).zfill(5)}_1_diff.png")
                # self.print_this_image((10 * torch.abs(modified_input_2[image_no] - original_2[image_no])),
                #                       f"{name}/{str(step).zfill(5)}_2_diff.png")

                # self.print_this_image(modified_input_2[image_no], f"{name}/{str(step).zfill(5)}_2.png")
                # self.print_this_image(original_2[image_no], f"{name}/{str(step).zfill(5)}_2_ori.png")
                #
                # self.print_this_image((10 * torch.abs(modified_input_2[image_no] - original_2[image_no])),
                #                       f"{name}/{str(step).zfill(5)}_2_diff.png")
                #
                # self.print_this_image(modified_input_3[image_no], f"{name}/{str(step).zfill(5)}_3.png")
                # self.print_this_image(original_3[image_no], f"{name}/{str(step).zfill(5)}_3_ori.png")
                #
                # self.print_this_image((10 * torch.abs(modified_input_3[image_no] - original_3[image_no])),
                #                       f"{name}/{str(step).zfill(5)}_3_diff.png")

                self.print_this_image(gt_rgb[image_no], f"{name}/{str(step).zfill(5)}_gt.png")
                # np.save(f"{name}/{str(step).zfill(5)}_gt", modified_raw.detach().cpu().numpy())

                if attacked_forward is not None:
                    self.print_this_image(attacked_forward[image_no], f"{name}/{str(step).zfill(5)}_tampered_{tamper_index}.png")

                if attacked_image is not None:
                    self.print_this_image(attacked_image[image_no], f"{name}/{str(step).zfill(5)}_postprocess_{tamper_index}.png")

                print("Saved:{}".format(f"{name}/{str(step).zfill(5)}"))

            # if self.opt['inference_do_subsequent_prediction']:
            #     logs_pred_accu = {}
            #     for idx_isp in range(3):
            #         source_image = eval(f"modified_input_{idx_isp}")[image_no:image_no + 1]
            #         ### get tampering source and mask
            #         if self.opt['inference_load_real_world_tamper']:
            #             file_name = f"{str(step).zfill(5)}_{idx_isp}_{str(self.rank)}.png"
            #             folder_name = f'/groupshare/ISP_results/xxhu_test/{self.task_name}/FORGERY_{idx_isp}/'
            #             mask_file_name = f"{str(step).zfill(5)}_0_{str(self.rank)}.png"
            #             mask_folder_name = f'/groupshare/ISP_results/xxhu_test/{self.task_name}/MASK/'
            #             # print(f"reading {folder_name+file_name}")
            #             img_GT = cv2.imread(folder_name + file_name, cv2.IMREAD_COLOR)
            #             # img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]
            #             # print(f"reading {mask_folder_name + file_name}")
            #             mask_GT = cv2.imread(mask_folder_name + mask_file_name, cv2.IMREAD_GRAYSCALE)
            #
            #             img_GT = img_GT.astype(np.float32) / 255.
            #             if img_GT.ndim == 2:
            #                 img_GT = np.expand_dims(img_GT, axis=2)
            #             # some images have 4 channels
            #             if img_GT.shape[2] > 3:
            #                 img_GT = img_GT[:, :, :3]
            #             mask_GT = mask_GT.astype(np.float32) / 255.
            #
            #             orig_height, orig_width, _ = img_GT.shape
            #             H, W, _ = img_GT.shape
            #
            #             mask_GT = torch.from_numpy(np.ascontiguousarray(mask_GT)).float().unsqueeze(0).unsqueeze(
            #                 0).cuda()
            #
            #             # BGR to RGB, HWC to CHW, numpy to tensor
            #             if img_GT.shape[2] == 3:
            #                 img_GT = img_GT[:, :, [2, 1, 0]]
            #
            #             tamper_source = torch.from_numpy(
            #                 np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float().unsqueeze(0).cuda()
            #
            #             test_input = source_image * (1 - mask_GT) + tamper_source * mask_GT
            #
            #         else:  ## using simulated tampering
            #
            #             #### tampering source from the training set
            #
            #             if self.previous_protected is not None:
            #                 self.previous_protected = self.label[0:1]
            #             self.previous_images = self.label[0:1]
            #
            #             masks, masks_GT, percent_range = self.mask_generation(modified_input=source_image,
            #                                                                   percent_range=None)
            #
            #             test_input, masks, mask_GT = self.tampering_RAW(
            #                 masks=masks, masks_GT=masks_GT,
            #                 modified_input=source_image, percent_range=percent_range,
            #                 index=self.opt['inference_tamper_index'],
            #             )
            #
            #             self.previous_protected = source_image.clone().detach()
            #
            #         ### attacks generate them all
            #         attack_lists = [
            #             (None, None, None), (0, None, None), (1, None, None), (2, None, None), (3, 18, None),
            #             (3, 14, None), (4, None, None),
            #             (None, None, 0), (None, None, 1), (None, None, 2), (None, None, 3),
            #         ]
            #         # attack_lists = [
            #         #     (None,None,None),(None,None,0),(None,None,1),
            #         #     (0, 20, None), (0, 20, 2), (0, 20, 1),
            #         #     (1, 20, None), (1, 20, 3), (1, 20, 0),
            #         #     (2, 20, None), (2, 20, 1), (2, 20, 2),
            #         #     (3, 10, None), (3, 10, 3), (3, 10, 0),
            #         #     (3, 14, None), (3, 14, 2), (3, 14, 1),
            #         #     (3, 18, None), (3, 18, 0), (3, 18, 3),
            #         #     (4, 20, None), (4, 20, 1), (4, 20, 2),
            #         # ]
            #         begin_idx = self.opt['inference_benign_attack_begin_idx']
            #         for idx_attacks in range(begin_idx, begin_idx + 1):  # len(attack_lists)
            #             do_attack, quality_idx, do_augment = attack_lists[idx_attacks]
            #             logs_pred, pred_resfcn, _ = self.get_predicted_mask(modified_input=test_input,
            #                                                                 masks_GT=mask_GT, do_attack=do_attack,
            #                                                                 quality_idx=quality_idx,
            #                                                                 do_augment=do_augment,
            #                                                                 step=step,
            #                                                                 filename_append=str(idx_isp),
            #                                                                 save_image=self.opt['inference_save_image']
            #                                                                 )
            #             if len(logs_pred_accu) == 0:
            #                 logs_pred_accu.update(logs_pred)
            #             else:
            #                 for key in logs_pred:
            #                     logs_pred_accu[key] += logs_pred[key]
            #
        return logs, (modified_raw, modified_input_0, modified_input_1), True