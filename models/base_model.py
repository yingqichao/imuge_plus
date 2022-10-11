import copy
import logging
import os
import math
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from collections import OrderedDict
import torchvision.transforms.functional as F
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.nn.parallel import DistributedDataParallel
from cycleisp_models.cycleisp import Raw2Rgb
import pytorch_ssim
from MVSS.models.mvssnet import get_mvss
from MVSS.models.resfcn import ResFCN
from metrics import PSNR
from models.modules.Quantization import diff_round
from noise_layers import *
from noise_layers.crop import Crop
from noise_layers.dropout import Dropout
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.middle_filter import MiddleBlur
from noise_layers.resize import Resize
from utils import stitch_images
from utils.JPEG import DiffJPEG
from data.pipeline import pipeline_tensor2image
# import matlab.engine
import torch.nn.functional as Functional
from utils.commons import create_folder
from data.pipeline import rawpy_tensor2image
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
from .invertible_net import Inveritible_Decolorization_PAMI
from models.networks import UNetDiscriminator
from loss import PerceptualLoss, StyleLoss
from .networks import SPADE_UNet
from lama_models.HWMNet import HWMNet
# import contextual_loss as cl
# import contextual_loss.functional as F
from loss import GrayscaleLoss
from .invertible_net import Inveritible_Decolorization_PAMI
from models.networks import UNetDiscriminator
from loss import PerceptualLoss, StyleLoss
from .networks import SPADE_UNet
from lama_models.HWMNet import HWMNet
from lama_models.my_own_elastic_dtcwt import my_own_elastic

class BaseModel():
    def __init__(self, opt,  args, train_set=None):

        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

        self.train_set = train_set
        self.rank = torch.distributed.get_rank()
        self.opt = opt
        self.args = args
        self.train_opt = opt['train']
        self.test_opt = opt['test']

        self.task_name = args.task_name  # self.opt['datasets']['train']['name']  # self.train_opt['task_name']
        self.loading_from = args.loading_from
        self.is_load_raw_models = self.opt['load_RAW_models']
        self.is_load_localizer_models = self.opt['load_RAW_models']
        self.is_load_RAW_models = self.opt['load_RAW_models']
        self.is_load_ISP_models = self.opt['load_ISP_models']
        print("Task Name: {}".format(self.task_name))
        self.global_step = 0
        self.new_task = self.train_opt['new_task']
        self.use_gamma_correction = self.opt['use_gamma_correction']
        self.conduct_augmentation = self.opt['conduct_augmentation']
        self.lower_false_positive = self.opt['lower_false_positive']
        self.conduct_cropping = self.opt['conduct_cropping']
        self.consider_robost = self.opt['consider_robost']
        self.CE_hyper_param = self.opt['CE_hyper_param']
        self.perceptual_hyper_param = self.opt['perceptual_hyper_param']
        self.ssim_hyper_param = self.opt['ssim_hyper_param']
        self.L1_hyper_param = self.opt["L1_hyper_param"]
        self.RAW_L1_hyper_param = self.opt['RAW_L1_hyper_param']
        self.style_hyper_param = self.opt['style_hyper_param']
        self.psnr_thresh = self.opt['psnr_thresh']
        self.raw_classes = self.opt['raw_classes']
        self.train_isp_networks = self.opt["train_isp_networks"]
        self.train_full_pipeline = self.opt["train_full_pipeline"]
        self.train_inpainting_surrogate_model = self.opt["train_inpainting_surrogate_model"]
        self.include_isp_inference = self.opt["include_isp_inference"]
        self.step_acumulate = self.opt["step_acumulate"]
        self.dtcwt_layers = self.opt['dtcwt_layers']
        self.model_save_period = self.opt['model_save_period']

        ####################################################################################################
        # todo: constants
        ####################################################################################################
        self.width_height = opt['datasets']['train']['GT_size']
        self.kernel_RAW_k0 = torch.tensor([[[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]], device="cuda",
                                          requires_grad=False).unsqueeze(0)
        self.kernel_RAW_k1 = torch.tensor([[[0, 0], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [0, 0]]], device="cuda",
                                          requires_grad=False).unsqueeze(0)
        self.kernel_RAW_k2 = torch.tensor([[[0, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, 0]]], device="cuda",
                                          requires_grad=False).unsqueeze(0)
        self.kernel_RAW_k3 = torch.tensor([[[0, 1], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [1, 0]]], device="cuda",
                                          requires_grad=False)
        expand_times = int(self.width_height // 2)
        self.kernel_RAW_k0 = self.kernel_RAW_k0.repeat(1, 1, expand_times, expand_times)
        self.kernel_RAW_k1 = self.kernel_RAW_k1.repeat(1, 1, expand_times, expand_times)
        self.kernel_RAW_k2 = self.kernel_RAW_k2.repeat(1, 1, expand_times, expand_times)
        self.kernel_RAW_k3 = self.kernel_RAW_k3.repeat(1, 1, expand_times, expand_times)

        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

        ####################################################################################################
        # todo: losses and attack layers
        # todo: JPEG attack rescaling deblurring
        ####################################################################################################
        self.psup = nn.PixelShuffle(upscale_factor=2).cuda()
        self.psdown = nn.PixelUnshuffle(downscale_factor=2).cuda()
        self.tanh = nn.Tanh().cuda()
        self.psnr = PSNR(255.0).cuda()
        # self.lpips_vgg = lpips.LPIPS(net="vgg").cuda()
        # self.exclusion_loss = ExclusionLoss().type(torch.cuda.FloatTensor).cuda()
        self.ssim_loss = pytorch_ssim.SSIM().cuda()
        self.crop = Crop().cuda()
        self.dropout = Dropout().cuda()
        self.gaussian = Gaussian().cuda()
        self.salt_pepper = SaltPepper(prob=0.01).cuda()
        self.gaussian_blur = GaussianBlur().cuda()
        self.median_blur = MiddleBlur().cuda()
        self.resize = Resize().cuda()
        self.identity = Identity().cuda()
        self.gray_scale_loss = GrayscaleLoss().cuda()
        self.jpeg_simulate = [
            [DiffJPEG(50, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(55, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(60, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(65, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(70, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(75, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(80, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(85, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(90, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(95, height=self.width_height, width=self.width_height).cuda(), ]
        ]

        self.bce_loss = nn.BCELoss().cuda()
        self.bce_with_logit_loss = nn.BCEWithLogitsLoss().cuda()
        self.l1_loss = nn.SmoothL1Loss(beta=0.5).cuda()  # reduction="sum"
        self.l2_loss = nn.MSELoss().cuda()  # reduction="sum"
        self.perceptual_loss = PerceptualLoss().cuda()
        self.style_loss = StyleLoss().cuda()
        self.Quantization = diff_round
        # self.Quantization = Quantization().cuda()
        # self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw']).cuda()
        # self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back']).cuda()
        # self.criterion_adv = CWLoss().cuda()  # loss for fooling target model
        self.CE_loss = nn.CrossEntropyLoss().cuda()
        self.width_height = opt['datasets']['train']['GT_size']
        self.init_gaussian = None
        # self.adversarial_loss = AdversarialLoss(type="nsgan").cuda()

        self.real_H, self.real_H_path, self.previous_images, self.previous_previous_images = None, None, None, None
        self.previous_protected = None
        self.previous_canny = None

        self.optimizer_G = None
        self.optimizer_localizer = None
        self.optimizer_discriminator_mask = None
        self.optimizer_discriminator = None
        self.optimizer_KD_JPEG = None
        self.optimizer_generator = None
        self.optimizer_qf_predict = None
        self.netG = None
        self.localizer = None
        self.discriminator = None
        self.discriminator_mask = None
        self.KD_JPEG = None
        self.global_step = 0
        self.updated_step = 0
        self.out_space_storage = ""

    ####################################################################################################
    # todo: using which ISP network?
    ####################################################################################################
    # def using_invISP(self):
    #     return self.global_step % 4 == 0
    # def using_cycleISP(self):
    #     return self.global_step % 4 == 1
    # def using_my_own_pipeline(self):
    #     return self.global_step % 4 == 2

    def create_folders_for_the_experiment(self):
        create_folder(self.out_space_storage)
        create_folder(self.out_space_storage + "/model")
        create_folder(self.out_space_storage + "/images")
        create_folder(self.out_space_storage + "/isp_images/")
        create_folder(self.out_space_storage + "/model/" + self.task_name)
        create_folder(self.out_space_storage + "/images/" + self.task_name)
        create_folder(self.out_space_storage + "/isp_images/" + self.task_name)

    def define_RAW2RAW_network(self):
        if 'UNet' not in self.task_name:
            print("using my_own_elastic as KD_JPEG.")
            n_channels = 3 if "ablation" in self.task_name else 4 # 36/48 12
            self.KD_JPEG = my_own_elastic(nin=n_channels, nout=n_channels, depth=4, nch=48, num_blocks=self.dtcwt_layers,
                                          use_norm_conv=False).cuda()
        else:
            self.KD_JPEG = HWMNet(in_chn=1, out_chn=1, wf=32, depth=4, subtask=0, style_control=False,
                                  use_dwt=False).cuda()
            # SPADE_UNet(in_channels=1, out_channels=1, residual_blocks=2).cuda()
            # Inveritible_Decolorization_PAMI(dims_in=[[1, 64, 64]], block_num=[2, 2, 2], augment=False, ).cuda()
        # InvISPNet(channel_in=3, channel_out=3, block_num=4, network="ResNet").cuda() HWMNet(in_chn=1, wf=32, depth=4).cuda() # UNetDiscriminator(in_channels=1,use_SRM=False).cuda()
        self.KD_JPEG = DistributedDataParallel(self.KD_JPEG, device_ids=[torch.cuda.current_device()],
                                               find_unused_parameters=True)

    def define_tampering_localization_network(self):
        if 'UNet' not in self.task_name:
            print("using my_own_elastic as discriminator_mask.")
            self.discriminator_mask = my_own_elastic(nin=3, nout=1, depth=4, nch=36, num_blocks=self.dtcwt_layers,
                                                     use_norm_conv=True).cuda()
        else:
            self.discriminator_mask = HWMNet(in_chn=3, out_chn=1, wf=32, depth=4, subtask=0,
                                             style_control=False, use_dwt=False, use_norm_conv=True).cuda()
            # UNetDiscriminator(in_channels=3, out_channels=1, residual_blocks=2, use_SRM=False, subtask=self.raw_classes).cuda() #UNetDiscriminator(use_SRM=False).cuda() #
        self.discriminator_mask = DistributedDataParallel(self.discriminator_mask,
                                                          device_ids=[torch.cuda.current_device()],
                                                          find_unused_parameters=True)

    ####################################################################################################
    # todo: using which tampering attacks?
    ####################################################################################################
    def using_simulated_inpainting(self):
        return self.global_step % 9 in [1,5,7]
    def using_splicing(self):
        return self.global_step % 9 in [0,4,6,9]
    def using_copy_move(self):
        return self.global_step % 9 in [2,3,8]

    ####################################################################################################
    # todo: using which image processing attacks?
    ####################################################################################################
    def using_weak_jpeg_plus_blurring_etc(self):
        return self.global_step % 8 in {0,1,2,4,6,7}

    def begin_using_momentum(self):
        return False #self.global_step>=0

    ####################################################################################################
    # todo: settings for beginning training
    ####################################################################################################
    def data_augmentation_on_rendered_rgb(self, modified_input, index=None):
        if index is None:
            index = self.global_step % 7

        is_stronger = np.random.rand() > 0.5
        if index % 4 in [0]:
            ## careful!
            strength = np.random.rand() * (0.2 if is_stronger>0 else -0.2)
            modified_adjusted = F.adjust_hue(modified_input, hue_factor=0+strength)  # 0.5 ave
        elif index % 4 in [1,4]:
            strength = np.random.rand() * (1.0 if is_stronger > 0 else -0.5)
            modified_adjusted = F.adjust_contrast(modified_input, contrast_factor=1+strength)  # 1 ave
        # elif self.global_step%5==2:
        ## not applicable
        # modified_adjusted = F.adjust_gamma(modified_input,gamma=0.5+1*np.random.rand()) # 1 ave
        elif index % 4 in [2,5]:
            strength = np.random.rand() * (1.0 if is_stronger > 0 else -0.5)
            modified_adjusted = F.adjust_saturation(modified_input, saturation_factor=1+strength)
        elif index % 4 in [3,6]:
            strength = np.random.rand() * (1.0 if is_stronger > 0 else -0.5)
            modified_adjusted = F.adjust_brightness(modified_input,
                                                    brightness_factor=1+strength)  # 1 ave
        modified_adjusted = self.clamp_with_grad(modified_adjusted)

        return modified_adjusted #modified_input + (modified_adjusted - modified_input).detach()

    def gamma_correction(self, tensor, avg=4095, digit=2.2):
    ## gamma correction
    #             norm_value = np.power(4095, 1 / 2.2) if self.camera_name == 'Canon_EOS_5D' else np.power(16383, 1 / 2.2)
    #             input_raw_img = np.power(input_raw_img, 1 / 2.2)
        norm = math.pow(avg, 1 / digit)
        tensor = torch.pow(tensor*avg, 1/digit)
        tensor = tensor / norm

        return tensor

    def do_aug_train(self, *, attacked_forward):
        skip_augment = np.random.rand() > 0.85
        if not skip_augment and self.conduct_augmentation:
            attacked_adjusted = self.data_augmentation_on_rendered_rgb(attacked_forward)
        else:
            attacked_adjusted = attacked_forward

        return attacked_adjusted

    def do_postprocess_train(self, *, attacked_adjusted, logs):
        skip_robust = np.random.rand() > 0.85
        if not skip_robust and self.consider_robost:
            if self.using_weak_jpeg_plus_blurring_etc():
                quality_idx = np.random.randint(20, 21)
            else:
                quality_idx = np.random.randint(10, 20)
            attacked_image = self.benign_attacks(attacked_forward=attacked_adjusted, logs=logs,
                                                 quality_idx=quality_idx)
        else:
            attacked_image = attacked_adjusted

        return attacked_image

    @torch.no_grad()
    def _momentum_update_key_encoder(self, momentum=0.9):
        ####################################################################################################
        # todo:  Momentum update of the key encoder
        # todo: param_k: momentum
        ####################################################################################################

        for param_q, param_k in zip(self.discriminator_mask.parameters(), self.discriminator.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def visualize_raw(self, raw_to_raw_tensor, bayer_pattern, white_balance=None, eval=False):
        batch_size, height_width = raw_to_raw_tensor.shape[0], raw_to_raw_tensor.shape[2]
        # 两个相机都是RGGB
        # im = np.expand_dims(raw, axis=2)
        # if self.kernel_RAW_k0.ndim!=4:
        #     self.kernel_RAW_k0 = self.kernel_RAW_k0.unsqueeze(0).repeat(batch_size,1,1,1)
        #     self.kernel_RAW_k1 = self.kernel_RAW_k1.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #     self.kernel_RAW_k2 = self.kernel_RAW_k2.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #     self.kernel_RAW_k3 = self.kernel_RAW_k3.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #     print(f"visualize_raw is inited. Current shape {self.kernel_RAW_k0.shape}")
        out_tensor = None
        for idx in range(batch_size):

            used_kernel = getattr(self, f"kernel_RAW_k{bayer_pattern[idx]}")
            v_im = raw_to_raw_tensor[idx:idx+1].repeat(1,3,1,1) * used_kernel

            if white_balance is not None:
                # v_im (1,3,512,512) white_balance (1,3)
                # print(white_balance[idx:idx+1].unsqueeze(2).unsqueeze(3).shape)
                # print(v_im.shape)
                v_im = v_im * white_balance[idx:idx+1].unsqueeze(2).unsqueeze(3)

            out_tensor = v_im if out_tensor is None else torch.cat((out_tensor, v_im), dim=0)

        return out_tensor.float() #.half() if not eval else out_tensor

    def pack_raw(self, raw_to_raw_tensor):
        # 两个相机都是RGGB
        batch_size, num_channels, height_width = raw_to_raw_tensor.shape[0], raw_to_raw_tensor.shape[1], raw_to_raw_tensor.shape[2]

        H, W = raw_to_raw_tensor.shape[2], raw_to_raw_tensor.shape[3]
        R = raw_to_raw_tensor[:,:, 0:H:2, 0:W:2]
        Gr = raw_to_raw_tensor[:,:, 0:H:2, 1:W:2]
        Gb = raw_to_raw_tensor[:,:, 1:H:2, 0:W:2]
        B = raw_to_raw_tensor[:,:, 1:H:2, 1:W:2]
        G_avg = (Gr + Gb) / 2
        out = torch.cat((R, G_avg, B), dim=1)
        print(out.shape)
        out = Functional.interpolate(
            out,
            size=[height_width, height_width],
            mode='bilinear')
        return out

    def cropping_mask_generation(self, forward_image, locs=None, min_rate=0.6, max_rate=1.0, logs=None):
        ####################################################################################################
        # todo: cropping
        # todo: cropped: original-sized cropped image, scaled_cropped: resized cropped image, masks, masks_GT
        ####################################################################################################
        # batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        # masks_GT = torch.ones_like(self.canny_image)

        self.height_ratio = min_rate + (max_rate - min_rate) * np.random.rand()
        self.width_ratio = min_rate + (max_rate - min_rate) * np.random.rand()

        self.height_ratio = min(self.height_ratio, self.width_ratio + 0.2)
        self.width_ratio = min(self.width_ratio, self.height_ratio + 0.2)

        if locs==None:
            h_start, h_end, w_start, w_end = self.crop.get_random_rectangle_inside(forward_image.shape,
                                                                                   self.height_ratio,
                                                                                   self.width_ratio)
        else:
            h_start, h_end, w_start, w_end = locs
        # masks_GT[:, :, h_start: h_end, w_start: w_end] = 0
        # masks = masks_GT.repeat(1, 3, 1, 1)

        cropped = forward_image[:, :, h_start: h_end, w_start: w_end]

        scaled_cropped = Functional.interpolate(
            cropped,
            size=[forward_image.shape[2], forward_image.shape[3]],
            mode='bilinear')
        scaled_cropped = self.clamp_with_grad(scaled_cropped)

        return (h_start, h_end, w_start, w_end), cropped, scaled_cropped #, masks, masks_GT

    def tamper_based_augmentation(self, modified_input, modified_canny, masks, masks_GT, logs):
        # tamper-based data augmentation
        batch_size, height_width = modified_input.shape[0], modified_input.shape[2]
        for imgs in range(batch_size):
            if imgs % 3 != 2:
                modified_input[imgs, :, :, :] = (
                            modified_input[imgs, :, :, :] * (1 - masks[imgs, :, :, :]) + self.previous_images[imgs, :,
                                                                                         :, :] * masks[imgs, :, :,
                                                                                                 :]).clone().detach()
                modified_canny[imgs, :, :, :] = (
                            modified_canny[imgs, :, :, :] * (1 - masks_GT[imgs, :, :, :]) + self.previous_canny[imgs, :,
                                                                                            :, :] * masks_GT[imgs, :, :,
                                                                                                    :]).clone().detach()

        return modified_input, modified_canny

    def print_this_image(self, image, filename):
        camera_ready = image.unsqueeze(0)
        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                     filename, nrow=1,
                                     padding=0, normalize=False)

    def mask_generation(self, modified_input, percent_range, logs):
        batch_size, height_width = modified_input.shape[0], modified_input.shape[2]
        masks_GT = torch.zeros(batch_size, 1, self.real_H.shape[2], self.real_H.shape[3]).cuda()
        ## THE RECOVERY STAGE WILL ONLY WORK UNDER LARGE TAMPERING
        ## TO LOCALIZE SMALL TAMPERING, WE ONLY UPDATE LOCALIZER NETWORK

        for imgs in range(batch_size):
            if imgs % 3 == 2:
                ## copy-move will not be too large
                percent_range = (0.00, 0.15)
            masks_origin, _ = self.generate_stroke_mask(
                [self.real_H.shape[2], self.real_H.shape[3]], percent_range=percent_range)
            masks_GT[imgs, :, :, :] = masks_origin.cuda()
        masks = masks_GT.repeat(1, 3, 1, 1)

        # masks is just 3-channel-version masks_GT
        return masks, masks_GT

    def tampering(self, forward_image, masks, masks_GT, modified_input, percent_range, idx_clip, num_per_clip, logs, index=None):
        batch_size, height_width = modified_input.shape[0], modified_input.shape[2]
        ####### Tamper ###############
        # attacked_forward = torch.zeros_like(modified_input)
        # for img_idx in range(batch_size):
        if index is None:
            index = self.global_step % 9

        if index in [0,4,6,9]: #self.using_splicing():
            ####################################################################################################
            # todo: splicing
            # todo: invISP
            ####################################################################################################
            attacked_forward = modified_input * (1 - masks) + (self.previous_protected if idx_clip is None else self.previous_protected[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous() * masks)
            # attack_name = "splicing"

        elif index in [2,3,8]: #self.using_copy_move():
            ####################################################################################################
            # todo: copy-move
            # todo: invISP
            ####################################################################################################
            lower_bound_percent = percent_range[0] + (percent_range[1] - percent_range[0]) * np.random.rand()
            ###### IMPORTANT NOTE: for ideal copy-mopv, here should be modified_input. If you want to ease the condition, can be changed to forward_iamge
            tamper = modified_input.clone().detach()
            x_shift, y_shift, valid, retried, max_valid, mask_buff = 0, 0, 0, 0, 0, None
            while retried<20 and not (valid>lower_bound_percent and (abs(x_shift)>(modified_input.shape[2]/3) or abs(y_shift)>(modified_input.shape[3]/3))):
                x_shift = int((modified_input.shape[2]) * (np.random.rand() - 0.5))
                y_shift = int((modified_input.shape[3]) * (np.random.rand() - 0.5))

                ### two times padding ###
                mask_buff = torch.zeros((masks.shape[0], masks.shape[1],
                                            masks.shape[2] + abs(2 * x_shift),
                                            masks.shape[3] + abs(2 * y_shift))).cuda()

                mask_buff[:, :,
                abs(x_shift) + x_shift:abs(x_shift) + x_shift + modified_input.shape[2],
                abs(y_shift) + y_shift:abs(y_shift) + y_shift + modified_input.shape[3]] = masks

                mask_buff = mask_buff[:, :,
                                    abs(x_shift):abs(x_shift) + modified_input.shape[2],
                                    abs(y_shift):abs(y_shift) + modified_input.shape[3]]

                valid = torch.mean(mask_buff)
                retried += 1
                if valid>=max_valid:
                    max_valid = valid
                    self.mask_shifted = mask_buff
                    self.x_shift, self.y_shift = x_shift, y_shift

            self.tamper_shifted = torch.zeros((modified_input.shape[0], modified_input.shape[1],
                                               modified_input.shape[2] + abs(2 * self.x_shift),
                                               modified_input.shape[3] + abs(2 * self.y_shift))).cuda()
            self.tamper_shifted[:, :, abs(self.x_shift) + self.x_shift: abs(self.x_shift) + self.x_shift + modified_input.shape[2],
            abs(self.y_shift) + self.y_shift: abs(self.y_shift) + self.y_shift + modified_input.shape[3]] = tamper


            self.tamper_shifted = self.tamper_shifted[:, :,
                             abs(self.x_shift): abs(self.x_shift) + modified_input.shape[2],
                             abs(self.y_shift): abs(self.y_shift) + modified_input.shape[3]]

            masks = self.mask_shifted.clone().detach()
            masks = self.clamp_with_grad(masks)
            valid = torch.mean(masks)

            masks_GT = masks[:, :1, :, :]
            attacked_forward = modified_input * (1 - masks) + self.tamper_shifted.clone().detach() * masks
            # del self.tamper_shifted
            # del self.mask_shifted
            # torch.cuda.empty_cache()

        elif index in [1,5,7]: #self.using_simulated_inpainting:
            ####################################################################################################
            # todo: simulated inpainting
            # todo: it is important, without protection, though the tampering can be close, it should also be detected.
            ####################################################################################################
            # attacked_forward = modified_input * (1 - masks) + forward_image * masks
            attacked_forward = modified_input * (1 - masks) + (self.previous_images if idx_clip is None else self.previous_images[idx_clip * num_per_clip:(idx_clip + 1) * num_per_clip].contiguous() * masks)

        attacked_forward = self.clamp_with_grad(attacked_forward)
        # attacked_forward = self.Quantization(attacked_forward)

        return attacked_forward, masks, masks_GT

    def benign_attacks(self, attacked_forward, quality_idx, logs, index=None):
        batch_size, height_width = attacked_forward.shape[0], attacked_forward.shape[2]
        attacked_real_jpeg = torch.rand_like(attacked_forward).cuda()
        if index is None:
            index = self.global_step % 8

        ## id of weak JPEG: 0,1,2,4,6,7
        if index in [0,5]:
            blurring_layer = self.resize
        elif index in [1,6]:
            blurring_layer = self.gaussian_blur
        elif index in [2,7]:
            blurring_layer = self.median_blur
        elif index in [4]:
            blurring_layer = self.gaussian
        elif index in [3]:
            blurring_layer = self.identity

        quality = int(quality_idx * 5)

        jpeg_layer_after_blurring = self.jpeg_simulate[quality_idx - 10][0] if quality < 100 else self.identity
        attacked_real_jpeg_simulate = self.clamp_with_grad(jpeg_layer_after_blurring(blurring_layer(attacked_forward)))
        # if self.using_jpeg_simulation_only():
        #     attacked_image = attacked_real_jpeg_simulate
        # else:  # if self.global_step%5==3:
        for idx_atkimg in range(batch_size):
            grid = attacked_forward[idx_atkimg]
            realworld_attack = self.real_world_attacking_on_ndarray(grid, quality, index)
            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        attacked_real_jpeg = attacked_real_jpeg.clone().detach()
        attacked_image = attacked_real_jpeg_simulate + (
                    attacked_real_jpeg - attacked_real_jpeg_simulate).clone().detach()

        # error_scratch = attacked_real_jpeg - attacked_forward
        # l_scratch = self.l1_loss(error_scratch, torch.zeros_like(error_scratch).cuda())
        # logs.append(('SCRATCH', l_scratch.item()))
        return attacked_image

    def benign_attacks_without_simulation(self, forward_image, quality_idx, logs, index=None):
        batch_size, height_width = forward_image.shape[0], forward_image.shape[2]
        attacked_real_jpeg = torch.rand_like(forward_image).cuda()

        quality = int(quality_idx * 5)

        for idx_atkimg in range(batch_size):
            grid = forward_image[idx_atkimg]
            realworld_attack = self.real_world_attacking_on_ndarray(grid, quality, index)
            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        return attacked_real_jpeg

    def real_world_attacking_on_ndarray(self, grid, qf_after_blur, index=None):
        # batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        if index is None:
            index = self.global_step % 8

        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to('cpu', torch.uint8).numpy()
        if index in [0,5]:
            # grid = self.resize(grid.unsqueeze(0))[0]
            newH, newW = int((0.7+0.6*np.random.rand())*self.width_height), int((0.7+0.6*np.random.rand())*self.width_height)
            realworld_attack = cv2.resize(np.copy(ndarr), (newH,newW),
                                          interpolation=cv2.INTER_LINEAR)
            realworld_attack = cv2.resize(np.copy(realworld_attack), (self.width_height, self.width_height),
                                interpolation=cv2.INTER_LINEAR)
        elif index in [1,6]:
            kernel_list = [3, 5, 7]
            kernel = random.choice(kernel_list)
            realworld_attack = cv2.GaussianBlur(ndarr, (kernel, kernel), 0)
        elif index in [2,7]:
            kernel_list = [3, 5, 7]
            kernel = random.choice(kernel_list)
            realworld_attack = cv2.medianBlur(ndarr, kernel)
        elif index in [4]:
            mean, sigma = 0, 1
            gauss = np.random.normal(mean, sigma, (self.width_height, self.width_height, 3))
            # 给图片添加高斯噪声
            realworld_attack = ndarr + gauss
        elif index in [3]:
            realworld_attack = ndarr

        _, realworld_attack = cv2.imencode('.jpeg', realworld_attack,
                                           (int(cv2.IMWRITE_JPEG_QUALITY), qf_after_blur))
        realworld_attack = cv2.imdecode(realworld_attack, cv2.IMREAD_UNCHANGED)
        # realworld_attack = data.util.channel_convert(realworld_attack.shape[2], 'RGB', [realworld_attack])[0]
        # realworld_attack = cv2.resize(copy.deepcopy(realworld_attack), (height_width, height_width),
        #                               interpolation=cv2.INTER_LINEAR)

        # ### jpeg in the file
        # cv2.imwrite('./temp.jpeg', realworld_attack,
        #                                    (int(cv2.IMWRITE_JPEG_QUALITY), qf_after_blur))
        # realworld_attack = cv2.imread('./temp.jpeg', cv2.IMREAD_COLOR)
        # realworld_attack = realworld_attack.astype(np.float32) / 255.
        # if realworld_attack.ndim == 2:
        #     realworld_attack = np.expand_dims(realworld_attack, axis=2)
        # # some images have 4 channels
        # if realworld_attack.shape[2] > 3:
        #     realworld_attack = realworld_attack[:, :, :3]
        # orig_height, orig_width, _ = realworld_attack.shape
        # H, W, _ = realworld_attack.shape
        # # BGR to RGB, HWC to CHW, numpy to tensor
        # if realworld_attack.shape[2] == 3:
        #     realworld_attack = realworld_attack[:, :, [2, 1, 0]]

        realworld_attack = realworld_attack.astype(np.float32) / 255.
        realworld_attack = torch.from_numpy(
            np.ascontiguousarray(np.transpose(realworld_attack, (2, 0, 1)))).contiguous().float()
        realworld_attack = realworld_attack.unsqueeze(0).cuda()
        return realworld_attack


    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def get_paths_from_images(self, path):
        '''get image path list from image folder'''
        assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
        images = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if self.is_image_file(fname):
                    # img_path = os.path.join(dirpath, fname)
                    images.append((path, dirpath[len(path) + 1:], fname))
        assert images, '{:s} has no valid image file'.format(path)
        return images

    def print_individual_image(self, cropped_GT, name):
        for image_no in range(cropped_GT.shape[0]):
            camera_ready = cropped_GT[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         name, nrow=1, padding=0, normalize=False)

    def load_image(self, path, readimg=False, Height=608, Width=608, grayscale=False):
        import data.util as util
        GT_path = path

        img_GT = util.read_img(GT_path)

        # change color space if necessary
        # img_GT = util.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]
        if grayscale:
            img_GT = rgb2gray(img_GT)

        img_GT = cv2.resize(copy.deepcopy(img_GT), (Width, Height), interpolation=cv2.INTER_LINEAR)
        return img_GT

    def img_random_crop(self, img_GT, Height=608, Width=608, grayscale=False):
        # # randomly crop
        # H, W = img_GT.shape[0], img_GT.shape[1]
        # rnd_h = random.randint(0, max(0, H - Height))
        # rnd_w = random.randint(0, max(0, W - Width))
        #
        # img_GT = img_GT[rnd_h:rnd_h + Height, rnd_w:rnd_w + Width, :]
        #
        # orig_height, orig_width, _ = img_GT.shape
        # H, W = img_GT.shape[0], img_GT.shape[1]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if not grayscale:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_GT = torch.from_numpy(
                np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).contiguous().float()
        else:
            img_GT = self.image_to_tensor(img_GT)

        return img_GT.cuda()

    def tensor_to_image(self, tensor):

        tensor = tensor * 255.0
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
        # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def tensor_to_image_batch(self, tensor):

        tensor = tensor * 255.0
        image = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def image_to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(np.asarray(img)).float()
        return img_t



    def clamp_with_grad(self, tensor):
        tensor_clamp = torch.clamp(tensor, 0, 1)
        return tensor + (tensor_clamp - tensor).clone().detach()

    def gaussian_batch(self, dims):
        return self.clamp_with_grad(torch.randn(tuple(dims)).cuda())

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label,save_dir=None, model_path=None):
        # save_dir = '../experiments/pretrained_models/'
        if model_path == None:
            model_path = self.opt['path']['models']
        if save_dir is None:
            save_filename = '{}_{}.pth'.format(iter_label, network_label)
            save_path = os.path.join(model_path, save_filename)
        else:
            save_filename = '{}_latest.pth'.format(network_label)
            save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        print("Model saved to: {}".format(save_path))
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step, model_path, network_list):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        if 'localizer' in network_list:
            state['optimizer_localizer'] = self.optimizer_localizer.state_dict()
        if 'discriminator_mask' in network_list:
            state['optimizer_discriminator_mask'] = self.optimizer_discriminator_mask.state_dict()

        if 'discriminator' in network_list:
            state['optimizer_discriminator'] = self.optimizer_discriminator.state_dict()

        if 'netG' in network_list:
            state['optimizer_G'] = self.optimizer_G.state_dict()
            state['clock'] = self.netG.module.clock

        if 'generator' in network_list:
            state['optimizer_generator'] = self.optimizer_generator.state_dict()
        if 'KD_JPEG' in network_list:
            state['optimizer_KD_JPEG'] = self.optimizer_KD_JPEG.state_dict()
        if 'qf_predict' in network_list:
            state['optimizer_qf_predict'] = self.optimizer_qf_predict.state_dict()
        # for s in self.schedulers:
        #     state['schedulers'].append(s.state_dict())
        # for o in self.optimizers:
        #     state['optimizers'].append(o.state_dict())

        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(model_path , save_filename)
        print("State saved to: {}".format(save_path))
        torch.save(state, save_path)

    def resume_training(self, state_path, network_list):
        resume_state = torch.load(state_path)
        if 'clock' in resume_state and 'netG' in network_list:
            self.localizer.module.clock = resume_state['clock']
        ##  Resume the optimizers and schedulers for training
        if 'optimizer_G' in resume_state and 'netG' in network_list:
            self.optimizer_G.load_state_dict(resume_state['optimizer_G'])
        if 'optimizer_localizer' in resume_state and 'localizer' in network_list:
            self.optimizer_localizer.load_state_dict(resume_state['optimizer_localizer'])
        if 'optimizer_discriminator_mask' in resume_state and 'discriminator_mask' in network_list:
            self.optimizer_discriminator_mask.load_state_dict(resume_state['optimizer_discriminator_mask'])
        if 'optimizer_discriminator' in resume_state and 'discriminator' in network_list:
            self.optimizer_discriminator.load_state_dict(resume_state['optimizer_discriminator'])
        if 'optimizer_qf_predict' in resume_state and 'qf_predict' in network_list:
            self.optimizer_qf_predict.load_state_dict(resume_state['optimizer_qf_predict'])
        if 'optimizer_generator' in resume_state and 'generator' in network_list:
            self.optimizer_generator.load_state_dict(resume_state['optimizer_generator'])
        if 'optimizer_KD_JPEG' in resume_state and 'KD_JPEG' in network_list:
            self.optimizer_KD_JPEG.load_state_dict(resume_state['optimizer_KD_JPEG'])

        # resume_optimizers = resume_state['optimizers']
        # resume_schedulers = resume_state['schedulers']
        # assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        # assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        # for i, o in enumerate(resume_optimizers):
        #     self.optimizers[i].load_state_dict(o)
        # for i, s in enumerate(resume_schedulers):
        #     self.schedulers[i].load_state_dict(s)

    def generate_stroke_mask(self, im_size, parts=5, parts_square=2, maxVertex=6, maxLength=64, maxBrushWidth=32,
                             maxAngle=360, percent_range=(0.0, 0.25)):
        minVertex, maxVertex = 1, 8
        minLength, maxLength = int(im_size[0] * 0.02), int(im_size[0] * 0.2)
        minBrushWidth, maxBrushWidth = int(im_size[0] * 0.02), int(im_size[0] * 0.2)
        mask = np.zeros((im_size[0], im_size[1]), dtype=np.float32)
        lower_bound_percent = percent_range[0] + (percent_range[1] - percent_range[0]) * np.random.rand()

        while True:
            mask = mask + self.np_free_form_mask(mask, minVertex, maxVertex, minLength, maxLength, minBrushWidth,
                                                 maxBrushWidth,
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

    def np_free_form_mask(self, mask_re, minVertex, maxVertex, minLength, maxLength, minBrushWidth, maxBrushWidth,
                          maxAngle, h, w):
        mask = np.zeros_like(mask_re)
        numVertex = np.random.randint(minVertex, maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        use_rect = False  # np.random.rand()<0.5
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
            ## drawing: https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
            if use_rect:
                cv2.rectangle(mask, (startY, startX), (startY + brushWidth, startX + brushWidth), 2)
            else:
                cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX

        if use_rect:
            cv2.rectangle(mask, (startY, startX), (startY + brushWidth, startX + brushWidth), 2)
        else:
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

    def F1score(self, predict_image, gt_image, thresh=0.2, get_auc=False):
        # gt_image = cv2.imread(src_image, 0)
        # predict_image = cv2.imread(dst_image, 0)
        # ret, gt_image = cv2.threshold(gt_image[0], int(255 * thresh), 255, cv2.THRESH_BINARY)
        # ret, predicted_binary = cv2.threshold(predict_image[0], int(255*thresh), 255, cv2.THRESH_BINARY)
        predicted_binary = self.tensor_to_image(predict_image[0])
        gt_image = self.tensor_to_image(gt_image[0, :1, :, :])
        if get_auc:
            AUC = getAUC(predicted_binary/255, gt_image/255)
        ret, predicted_binary = cv2.threshold(predicted_binary, int(255 * thresh), 255, cv2.THRESH_BINARY)
        ret, gt_image = cv2.threshold(gt_image, int(255 * thresh), 255, cv2.THRESH_BINARY)
        if get_auc:
            IoU = getIOU(predicted_binary/255, gt_image/255)
        # print(predicted_binary.shape)

        [TN, TP, FN, FP] = getLabels(predicted_binary, gt_image)
        # print("{} {} {} {}".format(TN,TP,FN,FP))
        F1 = getF1(TP, FP, FN)
        RECALL = getTPR(TP, FN)
        # cv2.imwrite(save_path, predicted_binary)
        return (F1, RECALL, AUC, IoU) if get_auc else (F1, RECALL)

def getLabels(img, gt_img):
    height = img.shape[0]
    width = img.shape[1]
    # TN, TP, FN, FP
    result = [0, 0, 0, 0]
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
    return (TP + TN) / (TP + FP + FN + TN)

def getFPR(TN, FP):
    return FP / (FP + TN)

def getTPR(TP, FN):
    return TP / (TP + FN)

def getTNR(FP, TN):
    return TN / (FP + TN)

def getFNR(FN, TP):
    return FN / (TP + FN)

def getF1(TP, FP, FN):
    return (2 * TP) / (2 * TP + FP + FN)

def getBER(TN, TP, FN, FP):
    return 1 / 2 * (getFPR(TN, FP) + FN / (FN + TP))

def getAUC(pre, gt):
    # 输入都是0-1区间内的 mask
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(gt.flatten(), pre.flatten())
    return auc

def getIOU(pre, gt):
    # 输入是二值化之后的 0 或 1
    union = np.logical_or(pre, gt)
    cross = np.logical_and(pre, gt)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return iou

## here test iou and auc
if __name__ == '__main__':
    gt = np.random.choice(2, [256, 256])
    pre = np.random.choice(2, [256, 256])
    iou = getIOU(pre, gt)
    print(iou)
    from sklearn.metrics import f1_score
    api_f1 = f1_score(gt.flatten(), pre.flatten())
    print(api_f1)
    auc = getAUC(pre.flatten(), gt.flatten())
    # auc = getAUC(pre, gt)
    print(auc)
    [TN, TP, FN, FP] = getLabels(pre*255, gt*255)
    # print("{} {} {} {}".format(TN,TP,FN,FP))
    our_f1 = getF1(TP, FP, FN)
    print(our_f1)