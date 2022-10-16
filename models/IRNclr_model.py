import logging
from collections import OrderedDict
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as Functional
from skimage.feature import canny
import torchvision
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from skimage.color import rgb2gray
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, CWLoss
from models.modules.Quantization import diff_round
import torch.distributed as dist
from utils.JPEG import DiffJPEG
from losses.loss import AdversarialLoss, PerceptualLoss, StyleLoss
import cv2
from utils.metrics import PSNR
from .invertible_net import Inveritible_Decolorization_PAMI
from .conditional_jpeg_generator import Crop_predictor
from utils import stitch_images
import os
import pytorch_ssim
from noise_layers import *
from noise_layers.dropout import Dropout
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.middle_filter import MiddleBlur
from noise_layers.resize import Resize
from noise_layers.crop import Crop
# import matlab.engine
from losses.loss import ExclusionLoss
from losses.fourier_loss import fft_L1_loss_color

# print("Starting MATLAB engine...")
# engine = matlab.engine.start_matlab()
# print("MATLAB engine loaded successful.")
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

"""
    ImugeV2 includes: netG localizer

    zxy     includes: generator attack_net localizer 


"""


class IRNclrModel(BaseModel):
    def __init__(self, opt,args):
        super(IRNclrModel, self).__init__(opt)
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        ########### CONSTANTS ###############
        self.TASK_IMUGEV2 = "ImugeV2"
        self.TASK_TEST = "Test"
        self.TASK_CropLocalize = "CropLocalize"
        self.TASK_RHI3 = "RHI3"
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        self.gpu_id = self.opt['gpu_ids'][0]
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.args = args
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.real_H, self.real_H_path, self.previous_images, self.previous_previous_images = None, None, None, None
        self.previous_canny = None
        self.task_name = self.opt['datasets']['train']['name']  # self.train_opt['task_name']
        print("Task Name: {}".format(self.task_name))
        self.global_step = 0
        self.new_task = self.train_opt['new_task']

        ############## Metrics and attacks #############
        self.tanh = nn.Tanh().cuda()
        self.psnr = PSNR(255.0).cuda()
        self.exclusion_loss = ExclusionLoss().type(torch.cuda.FloatTensor).cuda()
        self.ssim_loss = pytorch_ssim.SSIM().cuda()
        self.width_height = opt['datasets']['train']['GT_size']
        self.jpeg_simulate = [
            # [DiffJPEG(50, height=self.width_height, width=self.width_height).cuda(), ]
            # , [DiffJPEG(55, height=self.width_height, width=self.width_height).cuda(), ]
            [DiffJPEG(60, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(65, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(70, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(75, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(80, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(85, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(90, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(95, height=self.width_height, width=self.width_height).cuda(), ]
        ]
        self.crop = Crop().cuda()
        self.dropout = Dropout().cuda()
        self.gaussian = Gaussian().cuda()
        self.salt_pepper = SaltPepper(prob=0.01).cuda()
        self.gaussian_blur = GaussianBlur().cuda()
        self.median_blur = MiddleBlur(kernel=5).cuda()
        self.resize = Resize().cuda()
        self.identity = Identity().cuda()

        self.combined_jpeg_weak = Combined(
            [JpegMask(80), Jpeg(80), JpegMask(90), Jpeg(90), JpegMask(70), Jpeg(70), JpegMask(60), Jpeg(60)]).cuda()
        self.combined_jpeg_strong = Combined(
            [JpegMask(50), Jpeg(50), JpegMask(40), Jpeg(40), JpegMask(30), Jpeg(30), JpegMask(20), Jpeg(20)]).cuda()
        self.combined_diffjpeg = Combined([DiffJPEG(90), DiffJPEG(80), DiffJPEG(60), DiffJPEG(70)]).cuda()

        self.bce_loss = nn.BCELoss().cuda()
        self.l1_loss = nn.SmoothL1Loss().cuda()  # reduction="sum"
        self.l2_loss = nn.MSELoss().cuda()  # reduction="sum"
        self.perceptual_loss = PerceptualLoss().cuda()
        self.style_loss = StyleLoss().cuda()
        self.Quantization = diff_round #Quantization().cuda()
        self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw']).cuda()
        self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back']).cuda()
        self.criterion_adv = CWLoss().cuda()  # loss for fooling target model
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.width_height = opt['datasets']['train']['GT_size']
        self.init_gaussian = None
        self.adversarial_loss = AdversarialLoss(type="nsgan").cuda()

        ######### SCALER ##########################
        self.scaler_discriminator = torch.cuda.amp.GradScaler()
        self.scaler_G = torch.cuda.amp.GradScaler()
        self.scaler_generator = torch.cuda.amp.GradScaler()

        ############## Nets ################################
        self.network_list = ['netG', 'discriminator']

        # self.generator = Inveritible_Decolorization_PAMI(dims_in=[[4, 64, 64]], block_num=[2,2,2],augment=False,
        #                                             ).cuda()
        # self.generator = DistributedDataParallel(self.generator, device_ids=[torch.cuda.current_device()],
        #                                          )

        # self.localizer = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,
        #                                    use_spectral_norm=True, dim=16).cuda()

        # self.discriminator = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,use_spectral_norm=True, use_SRM=False).cuda()
        self.discriminator = Crop_predictor(in_nc=3, classes=4,
                                            crop_pred=True).cuda()
        self.discriminator = DistributedDataParallel(self.discriminator, device_ids=[torch.cuda.current_device()],
                                                     )
        # self.netG = Inveritible_Decolorization_PAMI(dims_in=[[4, 64, 64]], block_num=[4, 4, 4], augment=False,
        #                                             middle_block_constructor=ResBlock_light, middile_block=4,
        #                                             subnet_constructor=ResBlock_light).cuda()
        self.netG = Inveritible_Decolorization_PAMI(dims_in=[[4, 64, 64]], block_num=[2,2,2],augment=False,
                                                    ).cuda()
        self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],
                                            )
        # self.generator = Discriminator(in_channels=3, use_sigmoid=True).cuda()
        # self.generator = DistributedDataParallel(self.generator, device_ids=[torch.cuda.current_device()])
        # self.discriminator_mask = Discriminator(in_channels=3, use_SRM=False).cuda()
        # self.dis_adv_cov = Discriminator(in_channels=1, use_SRM=False).cuda()

        # self.localizer = DistributedDataParallel(self.localizer, device_ids=[torch.cuda.current_device()],
        #                                          find_unused_parameters=True)

        # self.dis_adv_cov = DistributedDataParallel(self.dis_adv_cov, device_ids=[torch.cuda.current_device()],
        #                                            find_unused_parameters=True)
        # self.discriminator_mask = DistributedDataParallel(self.discriminator_mask,
        #                                                   device_ids=[torch.cuda.current_device()],
        #                                                   find_unused_parameters=True)

        ########### For Crop localization ############
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

        ########## Load pre-trained ##################
        # self.load()

        self.log_dict = OrderedDict()

        ########## optimizers ##################
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

        optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                            weight_decay=wd_G,
                                            betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_G)

        # # # for generator
        # optim_params = []
        # for k, v in self.generator.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         if self.rank <= 0:
        #             logger.warning('Params [{:s}] will not optimize.'.format(k))
        # self.optimizer_generator = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
        #                                             weight_decay=wd_G,
        #                                             betas=(train_opt['beta1'], train_opt['beta2']))
        # self.optimizers.append(self.optimizer_generator)

        # # for mask discriminator
        # optim_params = []
        # for k, v in self.discriminator_mask.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         if self.rank <= 0:
        #             logger.warning('Params [{:s}] will not optimize.'.format(k))
        # self.optimizer_discriminator_mask = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
        #                                                      weight_decay=wd_G,
        #                                                      betas=(train_opt['beta1'], train_opt['beta2']))
        # self.optimizers.append(self.optimizer_discriminator_mask)
        #
        # # for mask discriminator
        # optim_params = []
        # for k, v in self.dis_adv_cov.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         if self.rank <= 0:
        #             logger.warning('Params [{:s}] will not optimize.'.format(k))
        # self.optimizer_dis_adv_cov = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
        #                                               weight_decay=wd_G,
        #                                               betas=(train_opt['beta1'], train_opt['beta2']))
        # self.optimizers.append(self.optimizer_dis_adv_cov)

        # for discriminator
        optim_params = []
        for k, v in self.discriminator.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_discriminator = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
                                                        weight_decay=wd_G,
                                                        betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_discriminator)

        # # localizer
        # optim_params = []
        # for k, v in self.localizer.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         if self.rank <= 0:
        #             logger.warning('Params [{:s}] will not optimize.'.format(k))
        # self.optimizer_localizer = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
        #                                             weight_decay=wd_G,
        #                                             betas=(train_opt['beta1'], train_opt['beta2']))
        # self.optimizers.append(self.optimizer_localizer)

        # ############## schedulers #########################
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state']))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

        ########## WELCOME BACK!!!
        ########## THE FOLLOWING MODELS SHOULD BE KEEP INTACT!!!
        self.out_space_storage = '/home/qcying/20220106_CLRNet'
        # pretrain = "/home/qcying/20220106_CLRNet/models/21010"
        self.model_storage = '/models/'
        self.model_path = str(21010)  # 29999

        # self.model_storage = '/models/COCO_4/'
        # self.model_path = str(1010)  # 29999
        load_models = True
        load_state = False
        load_network_list = ['netG']
        if load_models:
            self.pretrain = self.out_space_storage + self.model_storage + self.model_path
            self.reload(self.pretrain, load_network_list)
            ## load states
            state_path = self.out_space_storage + self.model_storage + '{}.state'.format(self.model_path)
            if load_state:
                logger.info('Loading training state')
                if os.path.exists(state_path):
                    self.resume_training(state_path, load_network_list)
                else:
                    logger.info('Did not find state [{:s}] ...'.format(state_path))

        # self.netG.module.clock = 1.0

    def feed_data(self, batch):
        # if self.train_opt['using_self_defined_dataset'] == 1.0:
        #     # {'GT': img_GT, 'GT_path': GT_path}
        #     data, label = batch
        #     self.real_H = data['GT'].cuda()
        #     self.jpeg_real_H = data['JPEG'].cuda()
        #     self.QF_GT = data['QF'].type(torch.FloatTensor).cuda()
        #     self.real_H_path = data['GT_path']
        #     # self.label_GT = label.type(torch.FloatTensor).cuda().unsqueeze(1) # regression
        #     self.label_GT = label.cuda()
        # else:
        img, label, canny_image = batch
        self.real_H = img.cuda()
        self.canny_image = canny_image.cuda()

        # self.ref_L = data['LQ'].cuda()  # LQ
        # self.real_H = data['GT'].cuda()  # GT

    def gaussian_batch(self, dims):
        return torch.clamp(torch.randn(tuple(dims)).cuda(), 0, 1)


    def optimize_parameters(self, step, latest_values=None, train=True, eval_dir=None):
        self.netG.train()
        self.discriminator.train()
        self.global_step = self.global_step + 1
        logs, debug_logs = [], []
        self.tamper_rate = 0.2

        self.real_H = torch.clamp(self.real_H, 0, 1)
        batch_size = self.real_H.shape[0]
        masks_GT = torch.zeros(batch_size, 1, self.real_H.shape[2], self.real_H.shape[3]).cuda()
        for imgs in range(batch_size):
            masks_origin, self.tamper_rate = self.generate_stroke_mask(
                [self.real_H.shape[2], self.real_H.shape[3]])
            masks_origin = masks_origin.cuda()
            masks_GT[imgs, :, :, :] = masks_origin

        mask_tamper = masks_GT.repeat(1, 3, 1, 1)
        save_interval = 3000

        with torch.enable_grad():
            is_real_train = True
            ######################### iMUGE ###########################################################
            # previous: 510
            if self.global_step > -1 and self.previous_images is not None and self.previous_previous_images is not None:
                # We use previous image as important content in the original image to prevent the networks from refering to the image prior
                # Then, we use previous previous image to tamper the image in the same location
                # l_atk, l_qf, pQF = self.train_JPEG_generator()

                if np.random.rand() < 0.15:
                    modified_input = (self.real_H * (1 - mask_tamper) + self.previous_images * mask_tamper).clone().detach()
                    modified_input = torch.clamp(modified_input, 0, 1)
                else:
                    modified_input = self.real_H

                forward_stuff = self.netG(x=torch.cat((modified_input, self.canny_image), dim=1))

                # forward_image = self.Quantization(forward_image)
                forward_image, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                forward_image = torch.clamp(forward_image, 0, 1)
                forward_null = torch.clamp(forward_null, 0, 1)

                ######## crop
                min_rate, max_rate = 0.7, 0.9
                masks_GT = torch.ones_like(self.canny_image)

                self.height_ratio = min_rate + (max_rate - min_rate) * np.random.rand()
                self.width_ratio = min_rate + (max_rate - min_rate) * np.random.rand()

                self.height_ratio = min(self.height_ratio, self.width_ratio + 0.2)
                self.width_ratio = min(self.width_ratio, self.height_ratio + 0.2)
                # image, cover_image = image_and_cover

                h_start, h_end, w_start, w_end = self.crop.get_random_rectangle_inside(forward_image.shape,
                                                                                       self.height_ratio,
                                                                                       self.width_ratio)
                masks_GT[:, :, h_start: h_end, w_start: w_end] = 0
                masks = masks_GT.repeat(1, 3, 1, 1)

                # actually #
                cropped = forward_image[:, :, h_start: h_end, w_start: w_end]
                # zero_images[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]

                scaled_cropped = Functional.interpolate(
                    cropped,
                    size=[forward_image.shape[2], forward_image.shape[3]],
                    mode='bicubic')
                scaled_cropped = torch.clamp(scaled_cropped, 0, 1)

                ####### Attack ###############
                attacked_forward = scaled_cropped
                attacked_forward = torch.clamp(attacked_forward, 0, 1)
                attack_full_name = ""

                # mix-up jpeg layer, remeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeember to clamp after each attack! Or gradient explosion will occur!!!
                # if self.global_step % 4 < 2:
                # attacked_image = torch.zeros((4,forward_image.shape[1],forward_image.shape[2],forward_image.shape[3]),dtype=torch.float32).cuda()
                way_attack = 6
                num_attacks = way_attack * batch_size

                ####### ATTACK IMAGE GENERATION
                attacked_image_0 = self.resize(attacked_forward)
                attacked_image_0 = torch.clamp(attacked_image_0, 0, 1)
                QF_idx0, QF_idx1 = np.random.randint(0, len(self.jpeg_simulate)), np.random.randint(0, len(self.jpeg_simulate))
                attack_layer = self.jpeg_simulate[QF_idx0][0]
                attacked_forward_0 = attack_layer(attacked_forward)
                attacked_image_1 = attacked_forward_0 #beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_1 = torch.clamp(attacked_image_1, 0, 1)

                attacked_image_2 = self.median_blur(attacked_forward)
                attacked_image_2 = torch.clamp(attacked_image_2, 0, 1)

                attacked_image_3 = self.gaussian_blur(attacked_forward)
                attacked_image_3 = torch.clamp(attacked_image_3, 0, 1)

                attacked_image_6 = self.gaussian(attacked_forward)
                attacked_image_6 = torch.clamp(attacked_image_6, 0, 1)

                # attacked_image_7 = self.identity(attacked_forward)
                # attacked_image_7 = torch.clamp(attacked_image_7, 0, 1)
                attack_layer = self.jpeg_simulate[QF_idx1][0]
                attacked_forward_0 = attack_layer(attacked_forward)
                attacked_image_7 = attacked_forward_0  # beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_7 = torch.clamp(attacked_image_7, 0, 1)

                attacked_image = torch.cat((attacked_image_0, attacked_image_1, attacked_image_2, attacked_image_3,
                                            attacked_image_6, attacked_image_7), dim=0)
                attacked_image = self.Quantization(attacked_image)

                ### GROUNDTRUTH IMAGE GENERATION
                attacked_image_0 = self.resize(modified_input)
                attacked_image_0 = torch.clamp(attacked_image_0, 0, 1)
                attack_layer = self.jpeg_simulate[QF_idx0][0]
                attacked_forward_0 = attack_layer(modified_input)
                attacked_image_1 = attacked_forward_0  # beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_1 = torch.clamp(attacked_image_1, 0, 1)

                attacked_image_2 = self.median_blur(modified_input)
                attacked_image_2 = torch.clamp(attacked_image_2, 0, 1)

                attacked_image_3 = self.gaussian_blur(modified_input)
                attacked_image_3 = torch.clamp(attacked_image_3, 0, 1)

                attacked_image_6 = self.gaussian(modified_input)
                attacked_image_6 = torch.clamp(attacked_image_6, 0, 1)

                # attacked_image_7 = self.identity(modified_input)
                # attacked_image_7 = torch.clamp(attacked_image_7, 0, 1)
                attack_layer = self.jpeg_simulate[QF_idx1][0]
                attacked_forward_0 = attack_layer(modified_input)
                attacked_image_7 = attacked_forward_0  # beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_7 = torch.clamp(attacked_image_7, 0, 1)

                modified_expand = torch.cat((attacked_image_0, attacked_image_1, attacked_image_2, attacked_image_3,
                                            attacked_image_6, attacked_image_7), dim=0).detach()

                masks_expand = masks.repeat(way_attack, 1, 1, 1)
                canny_expanded = self.canny_image.repeat(way_attack, 1, 1, 1)
                masks_GT_expand = masks_GT.repeat(way_attack, 1, 1, 1)
                # modified_expand = modified_input.repeat(way_attack, 1, 1, 1)
                forward_expand = forward_image.repeat(way_attack, 1, 1, 1)

                ######### scale back #########
                # ideally it should be #
                self.ideal_crop_pad_image = forward_expand * (1 - masks_expand)

                scaled_back_padded = torch.zeros_like(self.ideal_crop_pad_image)

                scaled_back = Functional.interpolate(
                    attacked_image,
                    size=[h_end - h_start, w_end - w_start],
                    mode='bicubic')
                scaled_back = torch.clamp(scaled_back, 0, 1)
                scaled_back_padded[:, :, h_start: h_end, w_start: w_end] = scaled_back
                dual_reshape_diff = (scaled_back_padded - self.ideal_crop_pad_image).clone().detach()

                self.real_crop_pad_image = self.ideal_crop_pad_image + dual_reshape_diff

                ######### begin localization
                apex = (h_start / forward_image.shape[2], h_end / forward_image.shape[2], w_start / forward_image.shape[3],
                        w_end / forward_image.shape[3])

                diffused_image = attacked_image.clone().detach()
                # Extract patch and label.
                cropmask, location = self.discriminator(diffused_image)
                # location = torch.clamp(location, 0, 1)
                labels_tensor = torch.zeros_like(location).cuda()
                labels_tensor[:, 0] = apex[0]  # int(32 * apex[0]) / 32
                labels_tensor[:, 1] = apex[1]  # int(32 * apex[1]) / 32
                labels_tensor[:, 2] = apex[2]  # int(32 * apex[2]) / 32
                labels_tensor[:, 3] = apex[3]  # int(32 * apex[3]) / 32
                # location = torch.clamp(location, 0, 1)
                patch_loss_value = self.l1_loss(location, labels_tensor)
                mask_loss_value = self.l1_loss(cropmask, masks_GT_expand)

                crop_loss = 1 * patch_loss_value + mask_loss_value  # * 0.01
                self.optimizer_discriminator.zero_grad()
                crop_loss.backward()
                # gradient clipping
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.train_opt['gradient_clipping'])
                self.optimizer_discriminator.step()
                self.optimizer_discriminator.zero_grad()
                # del cropmask
                # del location

                cropmask, location = self.discriminator(attacked_image)
                # location = torch.clamp(location, 0, 1)
                patch_loss_value = self.l1_loss(location, labels_tensor)
                mask_loss_value = self.l1_loss(cropmask, masks_GT_expand)
                CE = 1 * patch_loss_value + mask_loss_value  # * 0.01
                logs.append(('CE', crop_loss.item()))

                h_start, h_end, w_start, w_end = location[:, 0], location[:, 1], location[:, 2], location[:, 3]
                predicted_crop_region = torch.ones_like(masks_GT_expand)
                for idxx in range(location.shape[0]):
                    predicted_crop_region[idxx, :,
                    int(h_start[idxx] * self.width_height): int(h_end[idxx] * self.width_height),
                    int(w_start[idxx] * self.width_height): int(w_end[idxx] * self.width_height)] = 0
                apex_item = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}" \
                    .format(apex[0], apex[1], apex[2], apex[3])
                croppred_item = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}" \
                    .format(h_start[0].item(), h_end[0].item(), w_start[0].item(), w_end[0].item())
                logs.append(('AGT', apex_item))
                logs.append(('A', croppred_item))
                ######### end localization

                canny_input = torch.zeros_like(canny_expanded).cuda()
                for icn in range(batch_size):
                    img_GT = self.tensor_to_image(self.real_crop_pad_image[icn, :, :, :])
                    img_gray = rgb2gray(img_GT)
                    sigma = 2  # random.randint(1, 4)
                    cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                    cannied = self.image_to_tensor(cannied).cuda()
                    canny_input[icn, :, :, :] = cannied

                reversed_stuff, reverse_feature = self.netG(
                    torch.cat((self.real_crop_pad_image, canny_input), dim=1), rev=True)
                reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
                reversed_ch1 = torch.clamp(reversed_ch1, 0, 1)
                reversed_ch2 = torch.clamp(reversed_ch2, 0, 1)
                reversed_image = reversed_ch1
                reversed_canny = reversed_ch2

                distance = self.l1_loss  # if np.random.rand()>0.7 else self.l2_loss
                l_forward = distance(forward_image, modified_input)
                l_forward += distance(forward_null, self.canny_image)

                l_backward = distance(reversed_image, modified_expand)
                # l_backward = distance(reversed_image, forward_image.clone().detach())
                l_back_canny = distance(reversed_canny, canny_expanded)
                l_backward += l_back_canny
                # l_back_canny_local = (distance(reversed_canny * masks_GT_expand, canny_expanded * masks_GT_expand)) / torch.mean(
                #     masks_GT_expand)
                # l_backward += 1 * l_back_canny_local
                l_backward_l1_local = (distance(reversed_image * masks_expand,
                                                modified_expand * masks_expand)) / torch.mean(masks)
                # l_backward_l1_local = (distance(reversed_image * masks, forward_image.clone().detach() * masks)) / torch.mean(masks)

                psnr_forward = self.psnr(self.postprocess(modified_input), self.postprocess(forward_image)).item()
                psnr_backward = self.psnr(self.postprocess(modified_expand), self.postprocess(reversed_image)).item()
                # psnr_comp = self.psnr(self.postprocess(forward_image), self.postprocess(reversed_image)).item()
                logs.append(('PF', psnr_forward))
                logs.append(('PB', psnr_backward))
                # logs.append(('Pad', psnr_comp))

                loss = 0
                loss += (8 if psnr_forward < 31 else 7) * l_forward
                loss += (l_backward + 1 * l_backward_l1_local)
                loss += 0.001 * CE

                #################
                ### FREQUENY LOSS
                ## fft does not support halftensor
                forward_fft_loss = fft_L1_loss_color(fake_image=forward_image, real_image=modified_input)
                backward_fft_loss = fft_L1_loss_color(fake_image=reversed_image, real_image=modified_expand)

                loss += 1 * forward_fft_loss
                loss += 1 * backward_fft_loss
                logs.append(('F_FFT', forward_fft_loss.item()))
                logs.append(('B_FFT', backward_fft_loss.item()))
                ##################
                logs.append(('loss', loss.item()))
                logs.append(('lF', l_forward.item()))
                logs.append(('lB', l_backward.item()))
                logs.append(('lBedge', l_back_canny.item()))
                logs.append(('local', l_backward_l1_local.item()))
                logs.append(('mask', torch.mean(masks).item()))

                l_percept_fw_ssim = - self.ssim_loss(forward_image, modified_input)
                l_percept_bk_ssim = - self.ssim_loss(reversed_image, modified_expand)
                # # l_percept_canny_ssim = - self.ssim_loss(reversed_canny, canny_expanded)
                loss += 0.05 * l_percept_fw_ssim
                loss += 0.01 * (l_percept_bk_ssim)

                self.optimizer_G.zero_grad()
                loss.backward()
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
                self.optimizer_G.step()

                SSFW = (-l_percept_fw_ssim).item()
                SSBK = (-l_percept_bk_ssim).item()
                logs.append(('SF', SSFW))
                logs.append(('SB', SSBK))
                # logs.append(('pred', dg_pred.detach().cpu().numpy()))
                # logs.append(('gt', attack_rate_tensor.detach().cpu().numpy()))
                logs.append(('FW', psnr_forward))
                logs.append(('BK', psnr_backward))
                logs.append(('L-RealTime', l_backward_l1_local.item()))

                if step % 200 == 10 and self.rank <= 0:
                    images = stitch_images(
                        self.postprocess(modified_expand),
                        self.postprocess(canny_expanded),
                        self.postprocess(forward_expand),
                        self.postprocess(10 * torch.abs(modified_expand - forward_expand)),
                        # self.postprocess(predicted_marked),
                        self.postprocess(self.real_crop_pad_image),
                        # self.postprocess(self.real_crop_pad_image),
                        self.postprocess(masks_GT_expand),
                        self.postprocess(cropmask),
                        self.postprocess(10 * torch.abs(masks_GT_expand - cropmask)),
                        self.postprocess(reversed_image),
                        self.postprocess(10 * torch.abs(modified_expand - reversed_image)),
                        self.postprocess(reversed_canny),
                        self.postprocess(10 * torch.abs(canny_expanded - reversed_canny)),

                        img_per_row=1
                    )

                    name = self.out_space_storage + '/images/' + str(self.global_step).zfill(5) + ".png"
                    print('\nsaving sample ' + name)
                    images.save(name)

                # ####### Save independent images #############
                save_independent = False
                if save_independent:
                    name = self.out_space_storage + '/ori_0114/' + str(self.global_step).zfill(5) + "_" + str(
                        self.rank) + ".png"
                    for image_no in range(self.real_H.shape[0]):
                        camera_ready = self.real_H[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = self.out_space_storage + '/immu_0114/' + str(self.global_step).zfill(5) + "_" + str(
                        self.rank) + ".png"
                    for image_no in range(forward_image.shape[0]):
                        camera_ready = forward_image[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

        ######## Finally ####################
        if step % save_interval == 10 and self.rank <= 0:
            logger.info('Saving models and training states.')
            self.save(self.global_step, network_list=self.network_list)
        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H.clone().detach()
        return logs, debug_logs

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

    def evaluate(self, data_origin=None, data_immunize=None, data_tampered=None, data_tampersource=None,
                 data_mask=None):
        # self.netG.eval()
        # self.localizer.eval()
        with torch.no_grad():
            logs, debug_logs = [], []
            image_list_origin = None if data_origin is None else self.get_paths_from_images(data_origin)
            image_list_immunize = None if data_immunize is None else self.get_paths_from_images(data_immunize)
            image_list_tamper = None if data_tampered is None else self.get_paths_from_images(data_tampered)
            image_list_tampersource = None if data_tampersource is None else self.get_paths_from_images(
                data_tampersource)
            image_list_mask = None if data_mask is None else self.get_paths_from_images(data_mask)

            for idx in range(len(image_list_origin)):

                p, q, r = image_list_origin[idx]
                ori_path = os.path.join(p, q, r)
                img_GT = self.load_image(ori_path)
                print("Ori: {} {}".format(ori_path, img_GT.shape))
                self.real_H = self.img_random_crop(img_GT, 512, 512).cuda().unsqueeze(0)
                self.real_H = torch.clamp(self.real_H, 0, 1)
                img_gray = rgb2gray(img_GT)
                sigma = 2  # random.randint(1, 4)
                cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                self.canny_image = self.image_to_tensor(cannied).cuda().unsqueeze(0)

                p, q, r = image_list_immunize[idx]
                immu_path = os.path.join(p, q, r)
                img_GT = self.load_image(immu_path)
                print("Imu: {} {}".format(immu_path, img_GT.shape))
                self.immunize = self.img_random_crop(img_GT, 512, 512).cuda().unsqueeze(0)
                self.immunize = torch.clamp(self.immunize, 0, 1)
                p, q, r = image_list_tamper[idx]
                attack_path = os.path.join(p, q, r)
                img_GT = self.load_image(attack_path)
                print("Atk: {} {}".format(attack_path, img_GT.shape))
                self.attacked_image = self.img_random_crop(img_GT, 512, 512).cuda().unsqueeze(0)
                self.attacked_image = torch.clamp(self.attacked_image, 0, 1)
                p, q, r = image_list_tampersource[idx]
                another_path = os.path.join(p, q, r)
                img_GT = self.load_image(another_path)
                print("Another: {} {}".format(another_path, img_GT.shape))
                self.another_image = self.img_random_crop(img_GT, 512, 512).cuda().unsqueeze(0)
                self.another_image = torch.clamp(self.another_image, 0, 1)
                p, q, r = image_list_mask[idx]
                mask_path = os.path.join(p, q, r)
                img_GT = self.load_image(mask_path, grayscale=True)
                print("Mask: {} {}".format(mask_path, img_GT.shape))
                self.mask = self.img_random_crop(img_GT, 512, 512, grayscale=True).cuda().unsqueeze(0)
                self.mask = torch.clamp(self.mask, 0, 1)

                if self.immunize is None:
                    ##### generates immunized images ########
                    modified_input = self.real_H
                    # print(self.canny_image.shape)
                    forward_stuff = self.netG(x=torch.cat((modified_input, self.canny_image), dim=1))

                    # forward_image = self.Quantization(forward_image)
                    self.immunize, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                    self.immunize = torch.clamp(self.immunize, 0, 1)
                    forward_null = torch.clamp(forward_null, 0, 1)

                ####### Tamper ###############
                if self.attacked_image is None:

                    attacked_forward = self.immunize * (1 - self.mask) + self.another_image * self.mask

                    way_attack = 8
                    num_attacks = way_attack * batch_size

                    attacked_image_0 = self.resize(attacked_forward)
                    attacked_image_0 = torch.clamp(attacked_image_0, 0, 1)

                    attack_layer = self.combined_jpeg_weak
                    attack_layer_1 = self.combined_jpeg_weak
                    attacked_forward_0 = attack_layer(attacked_forward)
                    attacked_forward_1 = attack_layer_1(attacked_forward)
                    beta = np.random.rand()
                    attacked_image_1 = beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                    attacked_image_1 = torch.clamp(attacked_image_1, 0, 1)

                    attacked_image_2 = self.median_blur(attacked_forward)
                    attacked_image_2 = torch.clamp(attacked_image_2, 0, 1)

                    attacked_image_3 = self.gaussian_blur(attacked_forward)
                    attacked_image_3 = torch.clamp(attacked_image_3, 0, 1)

                    attacked_image_4 = self.dropout(attacked_forward * (1 - masks),
                                                    modified_input) + self.previous_previous_images * masks
                    attacked_image_4 = torch.clamp(attacked_image_4, 0, 1)

                    attack_layer = self.combined_jpeg_strong
                    attack_layer_1 = self.combined_jpeg_strong
                    attacked_forward_0 = attack_layer(attacked_forward)
                    attacked_forward_1 = attack_layer_1(attacked_forward)
                    beta = np.random.rand()
                    attacked_image_5 = beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                    attacked_image_5 = torch.clamp(attacked_image_5, 0, 1)

                    attacked_image_6 = self.gaussian(attacked_forward)
                    attacked_image_6 = torch.clamp(attacked_image_6, 0, 1)

                    attacked_image_7 = self.identity(attacked_forward)
                    attacked_image_7 = torch.clamp(attacked_image_7, 0, 1)

                    self.attacked_image = torch.cat((attacked_image_0, attacked_image_1, attacked_image_2,
                                                     attacked_image_3, attacked_image_4, attacked_image_5,
                                                     attacked_image_6, attacked_image_7), dim=0)
                    self.attacked_image = self.Quantization(self.attacked_image)
                    self.mask = self.masks.repeat(way_attack, 1, 1, 1)
                else:
                    self.attacked_image = self.identity(self.attacked_image)

                self.diffused_image = self.attacked_image.clone().detach()

                self.rectified_image = self.attacked_image * (1 - self.mask)
                self.rectified_image = torch.clamp(self.rectified_image, 0, 1)

                canny_input = torch.zeros(self.attacked_image.shape[0], 1, self.real_H.shape[2],
                                          self.real_H.shape[3]).cuda()
                for icn in range(1):
                    img_GT = self.tensor_to_image(self.rectified_image[icn, :, :, :])
                    img_gray = rgb2gray(img_GT)
                    sigma = 2  # random.randint(1, 4)
                    cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                    cannied = self.image_to_tensor(cannied).cuda()
                    canny_input[icn, :, :, :] = cannied

                reversed_stuff, reverse_feature = self.netG(
                    torch.cat((self.rectified_image, canny_input), dim=1), rev=True)
                reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
                reversed_ch1 = torch.clamp(reversed_ch1, 0, 1)
                reversed_ch2 = torch.clamp(reversed_ch2, 0, 1)
                self.reversed_image = reversed_ch1
                self.reversed_canny = reversed_ch2

                self.predicted_mask, _ = self.localizer(self.diffused_image)

                psnr_forward = self.psnr(self.postprocess(self.real_H), self.postprocess(self.immunize)).item()
                psnr_backward = self.psnr(self.postprocess(self.real_H),
                                          self.postprocess(self.reversed_image)).item()

                logs.append(('PF', psnr_forward))
                logs.append(('PB', psnr_backward))

                # l_percept_fw_ssim = - self.ssim_loss(forward_image, modified_input)
                # l_percept_bk_ssim = - self.ssim_loss(reversed_image, modified_expand)
                #
                # SSFW = (-l_percept_fw_ssim).item()
                # SSBK = (-l_percept_bk_ssim).item()
                # logs.append(('SF', SSFW))
                # logs.append(('SB', SSBK))

                logs.append(('FW', psnr_forward))
                logs.append(('BK', psnr_backward))

                # ####### Save independent images #############
                name = self.out_space_storage + '/results/jpeg/recovered_image/' + r + ".png"
                for image_no in range(self.reversed_image.shape[0]):
                    camera_ready = self.reversed_image[image_no].unsqueeze(0)
                    torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                 name, nrow=1, padding=0,
                                                 normalize=False)
                print("Saved:{}".format(name))

                name = self.out_space_storage + '/results/jpeg/predicted_masks/' + r + ".png"
                for image_no in range(self.predicted_mask.shape[0]):
                    camera_ready = self.predicted_mask[image_no].unsqueeze(0)
                    torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                 name, nrow=1, padding=0,
                                                 normalize=False)
                print("Saved:{}".format(name))

    def print_individual_image(self, cropped_GT, name):
        for image_no in range(cropped_GT.shape[0]):
            camera_ready = cropped_GT[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         name, nrow=1, padding=0, normalize=False)

    def load_image(self, path, readimg=False, Height=512, Width=512, grayscale=False):
        import data.util as util
        GT_path = path

        img_GT = util.read_img(GT_path)

        # change color space if necessary
        # img_GT = util.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]
        if grayscale:
            img_GT = rgb2gray(img_GT)

        img_GT = cv2.resize(np.copy(img_GT), (Width, Height), interpolation=cv2.INTER_LINEAR)
        return img_GT

    def img_random_crop(self, img_GT, Height=512, Width=512, grayscale=False):
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
            img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
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

    def test(self):
        # Lshape = self.ref_L.shape
        #
        # input_dim = Lshape[1]
        # self.input = self.real_H
        #
        # zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]
        #
        # gaussian_scale = 1
        # if self.test_opt and self.test_opt['gaussian_scale'] != None:
        #     gaussian_scale = self.test_opt['gaussian_scale']
        #
        # self.netG.eval()
        # with torch.no_grad():
        #     self.forw_L = self.netG(x=self.input)[:, :3, :, :]
        #     self.forw_L = self.Quantization(self.forw_L)
        #     y_forw = torch.cat((self.forw_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
        #     self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

        self.netG.train()

    # def downscale(self, HR_img):
    #     self.netG.eval()
    #     with torch.no_grad():
    #         LR_img = self.netG(x=HR_img)[:, :3, :, :]
    #         LR_img = self.Quantization(LR_img)
    #     self.netG.train()
    #
    #     return LR_img

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def image_to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(np.asarray(img)).float()
        return img_t

    # def upscale(self, LR_img, scale, gaussian_scale=1):
    #     Lshape = LR_img.shape
    #     zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
    #     y_ = torch.cat((LR_img, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
    #
    #     self.netG.eval()
    #     with torch.no_grad():
    #         HR_img = self.netG(x=y_, rev=True)[:, :3, :, :]
    #     self.netG.train()
    #
    #     return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    # def print_network(self):
    #     s, n = self.get_network_description(self.netG)
    #     if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
    #         net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
    #                                          self.netG.module.__class__.__name__)
    #     else:
    #         net_struc_str = '{}'.format(self.netG.__class__.__name__)
    #     if self.rank <= 0:
    #         logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
    #         logger.info(s)

    def reload(self, pretrain, network_list=['netG', 'localizer']):
        if 'netG' in network_list:
            load_path_G = pretrain + "_netG.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.netG, strict=False)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'KD_JPEG' in network_list:
            load_path_G = pretrain + "_KD_JPEG.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.KD_JPEG_net, strict=False)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'discriminator_mask' in network_list:
            load_path_G = pretrain + "_discriminator_mask.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.discriminator_mask, strict=False)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'discriminator' in network_list:
            load_path_G = pretrain + "_discriminator.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.discriminator, strict=True)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'qf_predict' in network_list:
            load_path_G = pretrain + "_qf_predict.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.qf_predict_network, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'localizer' in network_list:
            load_path_G = pretrain + "_localizer.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.localizer, strict=False)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'generator' in network_list:
            load_path_G = pretrain + "_generator.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.generator, strict=True)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

    def save(self, iter_label, folder='models', network_list=['netG', 'discriminator']):
        if 'netG' in network_list:
            self.save_network(self.netG, 'netG', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/' + self.task_name + '_' + str(self.gpu_id) + '/')
        if 'localizer' in network_list:
            self.save_network(self.localizer, 'localizer', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/' + self.task_name + '_' + str(self.gpu_id) + '/')
        if 'KD_JPEG' in network_list:
            self.save_network(self.KD_JPEG_net, 'KD_JPEG', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/' + self.task_name + '_' + str(self.gpu_id) + '/')
        if 'discriminator_mask' in network_list:
            self.save_network(self.discriminator_mask, 'discriminator_mask', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/' + self.task_name + '_' + str(self.gpu_id) + '/')
        if 'discriminator' in network_list:
            self.save_network(self.discriminator, 'discriminator', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/' + self.task_name + '_' + str(self.gpu_id) + '/')
        if 'qf_predict' in network_list:
            self.save_network(self.qf_predict_network, 'qf_predict', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/' + self.task_name + '_' + str(self.gpu_id) + '/')
        if 'generator' in network_list:
            self.save_network(self.generator, 'generator', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/' + self.task_name + '_' + str(self.gpu_id) + '/')
        # if 'netG' in network_list:
        self.save_training_state(epoch=0, iter_step=iter_label if self.rank == 0 else 0,
                                 model_path=self.out_space_storage + f'/{folder}/' + self.task_name + '_' + str(self.gpu_id) + '/',
                                 network_list=network_list)

    def generate_stroke_mask(self, im_size, parts=5, parts_square=2, maxVertex=8, maxLength=64, maxBrushWidth=32,
                             maxAngle=360, percent_range=(0.0, 0.4)):
        maxLength = int(im_size[0] / 4)
        maxBrushWidth = int(im_size[0] / 4)
        mask = np.zeros((im_size[0], im_size[1]), dtype=np.float32)
        lower_bound_percent = percent_range[0] + (percent_range[1] - percent_range[0]) * np.random.rand()

        # part = np.random.randint(2, parts + 1)

        # percent = 0
        while True:
            mask = mask + self.np_free_form_mask(mask, maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0],
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

    def np_free_form_mask(self, mask_re, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
        mask = np.zeros_like(mask_re)
        numVertex = np.random.randint(1, maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(8, maxLength + 1)
            brushWidth = np.random.randint(8, maxBrushWidth + 1) // 2 * 2
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
        """
        Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
        This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
        (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
        a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
        that we crop from top/bottom with equal probability.
        The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
        :param image: The image we want to crop
        :param height_ratio_range: The range of remaining height ratio
        :param width_ratio_range:  The range of remaining width ratio.
        :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
        """
        # image_height = image.shape[2]
        # image_width = image.shape[3]

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
        """
        Return a random number
        :param min:
        :param max:
        :return:
        """
        return np.random.rand() * (max - min) + min
