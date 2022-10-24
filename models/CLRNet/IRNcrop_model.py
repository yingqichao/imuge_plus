import logging
from collections import OrderedDict
from PIL import Image
import torchvision.transforms.functional as F
import torchvision
import torch.nn as nn
from skimage.color import rgb2gray
import models.lr_scheduler as lr_scheduler
from models.base_model import BaseModel
from models.modules.loss import ReconstructionLoss, CWLoss
from models.modules.Quantization import Quantization
import torch.distributed as dist
from utils.JPEG import DiffJPEG
from losses.loss import AdversarialLoss, PerceptualLoss, StyleLoss
import cv2
from utils.metrics import PSNR
from models.invertible_net import Inveritible_Decolorization_PAMI, ResBlock
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
from models.networks import DG_discriminator, Discriminator, UNetDiscriminator
from models.conditional_jpeg_generator import domain_generalization_predictor
from losses.loss import ExclusionLoss

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
import lpips


class IRNcropModel(BaseModel):
    def __init__(self, opt):
        super(IRNcropModel, self).__init__(opt)
        lr_D = 2e-5  # 2*train_opt['lr_G']
        lr_later = 1e-4
        ########### CONSTANTS ###############
        self.TASK_IMUGEV2 = "ImugeV2"
        self.TASK_TEST = "Test"
        self.TASK_CropLocalize = "CropLocalize"
        self.TASK_RHI3 = "RHI3"
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.lpips_vgg = lpips.LPIPS(net="vgg").cuda()
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.real_H, self.real_H_path, self.previous_images, self.previous_previous_images = None, None, None, None
        self.task_name = self.train_opt['task_name']
        print("Task Name: {}".format(self.task_name))
        self.global_step = 0
        self.new_task = self.train_opt['new_task']

        ############## Metrics and attacks #############
        self.tanh = nn.Tanh().cuda()
        self.psnr = PSNR(255.0).cuda()
        self.exclusion_loss = ExclusionLoss().type(torch.cuda.FloatTensor).cuda()
        self.ssim_loss = pytorch_ssim.SSIM().cuda()
        self.jpeg90 = Jpeg(90).cuda()
        self.jpeg80 = Jpeg(80).cuda()
        self.jpeg70 = Jpeg(70).cuda()
        self.jpeg60 = Jpeg(60).cuda()
        self.jpeg50 = Jpeg(50).cuda()
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
        self.Quantization = Quantization().cuda()
        self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw']).cuda()
        self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back']).cuda()
        self.criterion_adv = CWLoss().cuda()  # loss for fooling target model
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.width_height = opt['datasets']['train']['GT_size']
        self.init_gaussian = None
        self.adversarial_loss = AdversarialLoss(type="nsgan").cuda()

        ############## Nets ################################
        self.generator = domain_generalization_predictor().cuda()

        self.localizer = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,
                                           use_spectral_norm=True, dim=16).cuda()

        # self.localizer = DistributedDataParallel(self.localizer, device_ids=[torch.cuda.current_device()])
        # self.discriminator = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,use_spectral_norm=True, use_SRM=False).cuda()
        self.discriminator = DG_discriminator(in_channels=256, use_SRM=True).cuda()
        # self.discriminator = DistributedDataParallel(self.discriminator,device_ids=[torch.cuda.current_device()])

        self.netG = Inveritible_Decolorization_PAMI(dims_in=[[4, 64, 64]], block_num=[2, 2, 2],
                                                    subnet_constructor=ResBlock).cuda()
        # self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        # self.generator = Discriminator(in_channels=3, use_sigmoid=True).cuda()
        # self.generator = DistributedDataParallel(self.generator, device_ids=[torch.cuda.current_device()])
        self.discriminator_mask = Discriminator(in_channels=3, use_SRM=True).cuda()
        self.dis_adv_cov = Discriminator(in_channels=1, use_SRM=False).cuda()
        # self.discriminator_mask = UNetDiscriminator(use_sigmoid=True, in_channels=3, residual_blocks=2, out_channels=1,use_spectral_norm=True, use_SRM=False).cuda()
        # self.discriminator_mask = DistributedDataParallel(self.discriminator_mask, device_ids=[torch.cuda.current_device()])

        ########### For Crop localization ############
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

        ########## Load pre-trained ##################
        # self.load()

        load_state = True
        if load_state:
            pretrain = "/media/qichaoying/65F33762C14D581B/20220102_NOWAY/15010"

            load_path_G = pretrain + "_domain.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.generator, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            load_path_G = pretrain + "_netG.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            # load_path_G = pretrain + "_discriminator.pth"
            # if load_path_G is not None:
            #     logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
            #     if os.path.exists(load_path_G):
            #         self.load_network(load_path_G, self.discriminator, self.opt['path']['strict_load'])
            #     else:
            #         logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            # load_path_G = pretrain + "_discriminator_mask.pth"
            # if load_path_G is not None:
            #     logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
            #     if os.path.exists(load_path_G):
            #         self.load_network(load_path_G, self.discriminator_mask, self.opt['path']['strict_load'])
            #     else:
            #         logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            # load_path_G = pretrain + "_dis_adv_cov.pth"
            # if load_path_G is not None:
            #     logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
            #     if os.path.exists(load_path_G):
            #         self.load_network(load_path_G, self.dis_adv_cov, self.opt['path']['strict_load'])
            #     else:
            #         logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

            load_path_G = pretrain + "_localizer.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.localizer, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

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

        # for domain generator
        optim_params = []
        for k, v in self.generator.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_generator = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
                                                    weight_decay=wd_G,
                                                    betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_generator)

        # for mask discriminator
        optim_params = []
        for k, v in self.discriminator_mask.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_discriminator_mask = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
                                                             weight_decay=wd_G,
                                                             betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_discriminator_mask)

        # for mask discriminator
        optim_params = []
        for k, v in self.dis_adv_cov.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_dis_adv_cov = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
                                                      weight_decay=wd_G,
                                                      betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_dis_adv_cov)

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

        # localizer
        optim_params = []
        for k, v in self.localizer.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_localizer = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
                                                    weight_decay=wd_G,
                                                    betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_localizer)

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
        if label is not None:
            self.label = label.cuda()
        if canny_image is not None:
            self.canny_image = canny_image.cuda()

        # self.ref_L = data['LQ'].cuda()  # LQ
        # self.real_H = data['GT'].cuda()  # GT

    def gaussian_batch(self, dims):
        return torch.clamp(torch.randn(tuple(dims)).cuda(), 0, 1)

    def loss_forward(self, label_GT, label_pred, out, y, z):
        is_targeted = False
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = z.reshape([out.shape[0], -1])
        l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z ** 2) / z.shape[0]
        loss_adv = None
        if label_GT is not None:
            loss_adv = 2 * self.criterion_adv(label_pred, label_GT, is_targeted)

        return l_forw_fit, l_forw_ce, loss_adv

    def loss_backward(self, label_pred, label_GT, GT_ref, reversed_image):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(GT_ref, reversed_image)

        # loss_label = self.criterion(label_pred, label_GT)
        loss_label = None
        return l_back_rec, loss_label

    def loss_forward_and_backward_imuge(self, fake_outputs, cover_images, masks=None, use_l1=True, use_vgg=False,
                                        use_percept=False):
        gen_loss = 0
        if use_l1:
            gen_l1_loss = self.l1_loss(fake_outputs, cover_images)
        else:
            gen_l1_loss = self.l2_loss(fake_outputs, cover_images)
        gen_loss += gen_l1_loss
        gen_l1_local_loss = None
        if masks is not None:
            if use_l1:
                gen_l1_local_loss = self.l1_loss(fake_outputs * masks,
                                                 cover_images * masks) / torch.mean(masks)
            else:
                gen_l1_local_loss = self.l2_loss(fake_outputs * masks,
                                                 cover_images * masks) / torch.mean(masks)
            # gen_loss += 2 * gen_l1_local_loss

        if use_percept and cover_images.shape[1] == 3:
            # generator perceptual loss

            gen_content_loss = self.perceptual_loss(fake_outputs, cover_images)
            gen_loss += 0.01 * gen_content_loss

        if use_vgg and cover_images.shape[1] == 3:
            l_forward_ssim = - self.ssim_loss(fake_outputs, cover_images)
            gen_loss += 0.01 * l_forward_ssim

        # generator style loss
        # if masks is not None:
        #     gen_style_loss = self.style_loss(fake_outputs, cover_images)
        #     gen_style_loss = gen_style_loss * 250
        #     gen_loss += gen_style_loss

        return gen_l1_local_loss, gen_l1_loss, gen_loss

    def symm_pad(self, im, padding):
        h, w = im.shape[-2:]
        left, right, top, bottom = padding

        x_idx = np.arange(-left, w + right)
        y_idx = np.arange(-top, h + bottom)

        x_pad = self.reflect(x_idx, -0.5, w - 0.5)
        y_pad = self.reflect(y_idx, -0.5, h - 0.5)
        xx, yy = np.meshgrid(x_pad, y_pad)
        return im[..., yy, xx]

    def reflect(self, x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2 * rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod + double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    def optimize_parameters(self, step, latest_values=None, train=True, eval_dir=None):

        self.global_step = self.global_step + 1
        logs, debug_logs = [], []

        self.real_H = torch.clamp(self.real_H, 0, 1)
        batch_size = self.real_H.shape[0]

        masks, tamper_rate = self.generate_stroke_mask(
            [self.real_H.shape[2], self.real_H.shape[3]])
        masks = masks.cuda()
        masks_GT = masks.expand(batch_size, 1, -1, -1)
        masks = masks.expand(batch_size, 3, -1, -1)
        save_interval = 3000

        with torch.enable_grad():
            is_real_train = True
            ######################### iMUGE ###########################################################

            if self.previous_images is not None and self.previous_previous_images is not None:

                modified_input = self.real_H

                watermark = torch.zeros((batch_size,1,self.real_H.shape[2],self.real_H.shape[3]),dtype=torch.float32).cuda()
                for imgs in range(batch_size):
                    img_GT = self.tensor_to_image(self.previous_images[imgs, :, :, :])
                    img_gray = rgb2gray(img_GT)
                    img_gray = img_gray.astype(np.float)
                    watermark[imgs,:,:,:] = self.image_to_tensor(img_gray).cuda()

                forward_stuff = self.netG(x=torch.cat((modified_input, watermark), dim=1))

                # forward_image = self.Quantization(forward_image)
                forward_image, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                forward_image = torch.clamp(forward_image, 0, 1)
                forward_null = torch.clamp(forward_null, 0, 1)

                ####### Tamper ######################################################################################
                attacked_forward = forward_image.clone() * (1 - masks) + self.previous_previous_images * masks
                attacked_forward = torch.clamp(attacked_forward, 0, 1)
                attack_full_name = ""

                # mix-up jpeg layer, remeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeember to clamp after each attack! Or gradient explosion will occur!!!
                num_attacks = 4 * batch_size
                watermark_expanded = torch.zeros((num_attacks, 1, self.real_H.shape[2], self.real_H.shape[3]),
                                                 dtype=torch.float32).cuda()
                for imgs in range(batch_size):
                    watermark_expanded[imgs * 4:imgs * 4 + 4, :, :, :] = watermark[imgs, :, :, :].expand(4, -1, -1, -1)

                masks_GT_expand = torch.zeros((num_attacks, 1, self.real_H.shape[2], self.real_H.shape[3]),
                                              dtype=torch.float32).cuda()
                for imgs in range(batch_size):
                    masks_GT_expand[imgs * 4:imgs * 4 + 4, :, :, :] = masks_GT[imgs, :, :, :].expand(4, -1, -1,-1)
                modified_expand = torch.zeros((num_attacks, 3, self.real_H.shape[2], self.real_H.shape[3]),
                                              dtype=torch.float32).cuda()
                for imgs in range(batch_size):
                    modified_expand[imgs * 4:imgs * 4 + 4, :, :, :] = modified_input[imgs, :, :, :].expand(4, -1, -1,-1)
                forward_expand = torch.zeros((num_attacks, 3, self.real_H.shape[2], self.real_H.shape[3]),
                                             dtype=torch.float32).cuda()
                for imgs in range(batch_size):
                    forward_expand[imgs * 4:imgs * 4 + 4, :, :, :] = forward_image[imgs, :, :, :].expand(4, -1, -1, -1)

                attacked_image = torch.zeros((num_attacks, 3, self.real_H.shape[2], self.real_H.shape[3]),dtype=torch.float32).cuda()

                attacked_image_0 = self.resize(attacked_forward)
                attacked_image_0 = torch.clamp(attacked_image_0, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4:imgs * 4 + 1, :, :, :] = attacked_image_0[imgs:imgs + 1, :, :, :]

                attack_layer = self.combined_jpeg_weak
                attack_layer_1 = self.combined_jpeg_weak
                attacked_forward_0 = attack_layer(attacked_forward)
                # attacked_forward_0 = torch.clamp(attacked_forward_0, 0, 1)
                attacked_forward_1 = attack_layer_1(attacked_forward)
                # attacked_forward_1 = torch.clamp(attacked_forward_1, 0, 1)
                beta = np.random.rand()
                attacked_image_1 = beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_1 = torch.clamp(attacked_image_1, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4 + 1:imgs * 4 + 2, :, :, :] = attacked_image_1[imgs:imgs + 1, :, :, :]

                attacked_image_2 = self.median_blur(attacked_forward)
                attacked_image_2 = torch.clamp(attacked_image_2, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4 + 2:imgs * 4 + 3, :, :, :] = attacked_image_2[imgs:imgs + 1, :, :, :]

                attacked_image_3 = self.gaussian_blur(attacked_forward)
                attacked_image_3 = torch.clamp(attacked_image_3, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4 + 3:imgs * 4 + 4, :, :, :] = attacked_image_3[imgs:imgs + 1, :, :, :]

                # attacked_image_4 = self.dropout(attacked_forward, modified_input)
                # attacked_image_4 = torch.clamp(attacked_image_4, 0, 1)
                #
                # attack_layer = self.combined_jpeg_strong
                # attack_layer_1 = self.combined_jpeg_strong
                # attacked_forward_0 = attack_layer(attacked_forward)
                # # attacked_forward_0 = torch.clamp(attacked_forward_0, 0, 1)
                # attacked_forward_1 = attack_layer_1(attacked_forward)
                # # attacked_forward_1 = torch.clamp(attacked_forward_1, 0, 1)
                # beta = np.random.rand()
                # attacked_image_5 = beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                # attacked_image_5 = torch.clamp(attacked_image_5, 0, 1)
                #
                # attacked_image_6 = self.gaussian(attacked_forward)
                # attacked_image_6 = torch.clamp(attacked_image_6, 0, 1)
                #
                # attacked_image_7 = self.identity(attacked_forward)
                # attacked_image_7 = torch.clamp(attacked_image_7, 0, 1)

                attack_full_name += "Mixed"


                attacked_image = self.Quantization(attacked_image)

                diffused_image = attacked_image.clone().detach()

                logs.append(('Kind', attack_full_name))

                ######## Train Localizer for recover ##########

                localizer_input = diffused_image
                gen_input_fake = localizer_input
                gen_fake, _ = self.localizer(gen_input_fake)
                CE = self.bce_loss(gen_fake, masks_GT_expand)
                self.localizer.train()
                self.optimizer_localizer.zero_grad()
                CE.backward()
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.localizer.parameters(), self.train_opt['gradient_clipping'])
                self.optimizer_localizer.step()
                logs.append(('CE', CE.item()))

                gen_input_fake = diffused_image
                self.optimizer_localizer.zero_grad()
                gen_fake, _ = self.localizer(gen_input_fake)
                CE = self.bce_loss(gen_fake, masks_GT_expand)

                ############################ end tamper ######################################################################

                attacked_forward = forward_image
                attacked_forward = torch.clamp(attacked_forward, 0, 1)
                attack_full_name = ""

                # mix-up jpeg layer, remeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeember to clamp after each attack! Or gradient explosion will occur!!!
                num_attacks = 4 * batch_size

                attacked_image = torch.zeros((num_attacks, 3, self.real_H.shape[2], self.real_H.shape[3]),
                                             dtype=torch.float32).cuda()

                attacked_image_0 = self.resize(attacked_forward)
                attacked_image_0 = torch.clamp(attacked_image_0, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4:imgs * 4 + 1, :, :, :] = attacked_image_0[imgs:imgs + 1, :, :, :]

                attack_layer = self.combined_jpeg_weak
                attack_layer_1 = self.combined_jpeg_weak
                attacked_forward_0 = attack_layer(attacked_forward)
                # attacked_forward_0 = torch.clamp(attacked_forward_0, 0, 1)
                attacked_forward_1 = attack_layer_1(attacked_forward)
                # attacked_forward_1 = torch.clamp(attacked_forward_1, 0, 1)
                beta = np.random.rand()
                attacked_image_1 = beta * attacked_forward_0 + (1 - beta) * attacked_forward_1
                attacked_image_1 = torch.clamp(attacked_image_1, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4 + 1:imgs * 4 + 2, :, :, :] = attacked_image_1[imgs:imgs + 1, :, :, :]

                attacked_image_2 = self.median_blur(attacked_forward)
                attacked_image_2 = torch.clamp(attacked_image_2, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4 + 2:imgs * 4 + 3, :, :, :] = attacked_image_2[imgs:imgs + 1, :, :, :]

                attacked_image_3 = self.gaussian_blur(attacked_forward)
                attacked_image_3 = torch.clamp(attacked_image_3, 0, 1)
                for imgs in range(batch_size):
                    attacked_image[imgs * 4 + 3:imgs * 4 + 4, :, :, :] = attacked_image_3[imgs:imgs + 1, :, :, :]

                attacked_image = self.Quantization(attacked_image)

                tampered_attacked_image, apex = self.crop(attacked_image,min_rate=0.5,max_rate=0.8)
                watermark_GT, _ = self.crop(watermark_expanded,apex=apex)
                reverse_GT, _ = self.crop(modified_expand, apex=apex)
                rectified_crop_padded_image = torch.clamp(tampered_attacked_image, 0, 1)

                reversed_stuff, reverse_feature = self.netG(
                    torch.cat((rectified_crop_padded_image, 0.5+torch.zeros_like(watermark_expanded)), dim=1), rev=True)
                reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
                reversed_ch1 = torch.clamp(reversed_ch1, 0, 1)
                reversed_ch2 = torch.clamp(reversed_ch2, 0, 1)
                reversed_image = reversed_ch1
                reversed_canny = reversed_ch2


                lr = self.get_current_learning_rate()
                logs.append(('lr', lr))


                distance = self.l1_loss  # if np.random.rand()>0.7 else self.l2_loss
                l_forward = distance(forward_image, modified_input)
                l_null = distance(forward_null, 0.5+torch.zeros_like(forward_null))
                l_forward += 10 * l_null

                logs.append(('NULL', l_null.item()))
                l_backward = distance(reversed_image, reverse_GT)
                # l_backward = distance(reversed_image, forward_image.clone().detach())
                l_back_canny = distance(reversed_canny, watermark_GT)
                l_backward += l_back_canny

                gen_content_loss = self.perceptual_loss(forward_image, modified_input)
                l_forward += 0.01 * gen_content_loss
                gen_content_loss_back = self.perceptual_loss(reversed_canny.expand(-1,3,-1,-1), watermark_GT.expand(-1,3,-1,-1))
                l_backward += 0.01 * gen_content_loss_back

                logs.append(('lF', l_forward.item()))
                logs.append(('lB', l_backward.item()))
                logs.append(('lBedge', l_back_canny.item()))
                alpha_forw, alpha_back, gamma, delta = 1.0, 1, 1e-2, 0.01

                psnr_forward = self.psnr(self.postprocess(modified_input), self.postprocess(forward_image)).item()
                psnr_backward = self.psnr(self.postprocess(reversed_canny), self.postprocess(watermark_GT)).item()
                # psnr_comp = self.psnr(self.postprocess(forward_image), self.postprocess(reversed_image)).item()
                logs.append(('PF', psnr_forward))
                logs.append(('PB', psnr_backward))
                # logs.append(('Pad', psnr_comp))

                # nullify
                # l_null = (self.l2_loss(forward_null,0.5+torch.zeros_like(self.canny_image).cuda()))
                # logs.append(('NL', l_null.item()))

                loss = 0
                loss += (1.5 if psnr_forward < 32 else alpha_forw) * l_forward
                loss += (1.25 * alpha_back if psnr_forward - psnr_backward > 1 else alpha_back) * l_backward
                loss += gamma * CE
                    #        + alpha_back * l_bac
                #       kward_l1_local
                # loss += delta * FW_GAN
                # loss += delta * (REV_GAN)
                # loss += delta * REVedge_GAN
                # loss += 1.0 * l_null
                # loss += 1 * l_dg

                # print(l_forward)
                # print(l_backward)
                # print(l_backward_l1_local)
                # print(REV_GAN)
                # print(CE)
                # if psnr_forward<28:

                #     # generator perceptual loss
                # l_percept_fw_vgg = self.perceptual_loss(forward_image, modified_input)
                # l_percept_bk_vgg = self.perceptual_loss(reversed_image, modified_input)

                l_percept_fw_ssim = - self.ssim_loss(forward_image, modified_input)
                l_percept_bk_ssim = - self.ssim_loss(reversed_image, reverse_GT)
                l_percept_canny_ssim = - self.ssim_loss(reversed_canny, watermark_expanded)
                loss += delta * l_percept_fw_ssim
                loss += delta * (l_percept_bk_ssim)
                loss += delta * (l_percept_canny_ssim)
                # loss += delta * l_dg

                # # l_percept_fw_lpips = torch.mean(self.lpips_vgg.forward(forward_image, modified_input))
                # # l_percept_bk_lpips = torch.mean(self.lpips_vgg.forward(reversed_image, modified_input))
                # l_percept_fw = (l_percept_fw_vgg + l_percept_fw_ssim )/2
                # l_percept_bk = (l_percept_bk_vgg + l_percept_bk_ssim )/2
                # logs.append(('lpF', l_percept_fw.item()))
                # logs.append(('lpB', l_percept_bk.item()))
                # logs.append(('lssimF', l_percept_fw_ssim.item()))
                # # logs.append(('lpipF', l_percept_fw_lpips.item()))
                # logs.append(('lvggF', l_percept_fw_vgg.item()))
                # logs.append(('lssimB', l_percept_bk_ssim.item()))
                # # logs.append(('lpipF', l_percept_fw_lpips.item()))
                # logs.append(('lvggB', l_percept_bk_vgg.item()))
                # loss +=  1.5 * l_percept_fw
                # loss += 0.1 * CE
                # loss += 0.1 * FW_GAN
                # if psnr_forward>20:
                #     loss += 0.1 * (REV_GAN)
                #
                #     loss += 1 * l_percept_bk
                #     loss += 2 * l_backward_l1_local

                # l_forward_ssim = - self.ssim_loss(fake_outputs, cover_images)
                # gen_loss += 0.01 * l_forward_ssim

                self.netG.train()
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

                if step % 500 == 10:
                    images = stitch_images(
                        self.postprocess(modified_expand),
                        self.postprocess(watermark_expanded),
                        self.postprocess(forward_expand),
                        self.postprocess(10 * torch.abs(modified_expand - forward_expand)),
                        # self.postprocess(predicted_marked),
                        self.postprocess(diffused_image),
                        # self.postprocess(rectified_crop_padded_image),
                        self.postprocess(masks_GT_expand),
                        self.postprocess(gen_fake),
                        self.postprocess(10 * torch.abs(masks_GT_expand - gen_fake)),
                        self.postprocess(reversed_image),
                        self.postprocess(10 * torch.abs(reverse_GT - reversed_image)),
                        self.postprocess(reversed_canny),
                        self.postprocess(10 * torch.abs(watermark_expanded - reversed_canny)),

                        # self.postprocess(gen_fake_1),
                        # self.postprocess(gen_fake_mask_1),
                        # self.postprocess(gen_fake_mask_2),

                        img_per_row=1
                    )

                    out_space_storage = '/media/qichaoying/65F33762C14D581B/20220102_NOWAY'

                    name = out_space_storage + '/images/' + str(step).zfill(5) + ".png"
                    print('\nsaving sample ' + name)
                    images.save(name)
                # # Clean-up.
                # del patches
                # del true_locations
                # del global_images
                # del true_rectangles

                ############### set log ######################
                # with torch.no_grad():
                #
                #     if step % 500 == 10:
                #         self.netG.eval()
                #         self.localizer.eval()
                #         self.discriminator.eval()
                #         self.discriminator_mask.eval()
                #         masks, tamper_rate = self.generate_stroke_mask(
                #             [self.real_H.shape[2], self.real_H.shape[3]])
                #         masks = masks.cuda()
                #         masks_GT = masks.expand(batch_size, 1, -1, -1)
                #         masks = masks.expand(batch_size, 3, -1, -1)
                #         # if dist.get_rank() == 0:
                #         #     modified_input_1 = self.real_H * (1 - masks) + self.previous_images * masks
                #         #     modified_input_1 = torch.clamp(modified_input_1, 0, 1)
                #         # else:
                #         modified_input_1 = self.real_H
                #         # modified_input = self.real_H * (1 - masks) + self.previous_images * masks
                #         forward_stuff = self.netG(x=torch.cat((modified_input_1,self.canny_image),dim=1))
                #         forward_image_1 = forward_stuff[:, :3, :, :]
                #         forward_image_1 = torch.clamp(forward_image_1, 0, 1)
                #         # forward_image_1 = self.Quantization(torch.clamp(forward_image_1, 0, 1))
                #
                #         ####### Tamper ###############
                #         if np.random.rand() < 0.5:
                #             mixup1 = self.combined_jpeg_weak
                #             mixup2 = self.combined_jpeg_strong
                #         else:
                #             mixup1 = self.resize
                #             mixup2 = self.gaussian_blur
                #
                #         attacked_forward_1 = mixup1(forward_image_1)
                #         attacked_forward_2 = mixup2(forward_image_1)
                #         attacked_forward_1 = self.Quantization(torch.clamp(attacked_forward_1, 0, 1))
                #         attacked_forward_2 = self.Quantization(torch.clamp(attacked_forward_2, 0, 1))
                #
                #         diffused_image_1 = attacked_forward_1 * (
                #                     1 - masks) + self.previous_previous_images * masks
                #         diffused_image_1 = torch.clamp(diffused_image_1, 0, 1)
                #         diffused_image_2 = attacked_forward_2 * (
                #                     1 - masks) + self.previous_previous_images * masks
                #         diffused_image_2 = torch.clamp(diffused_image_2, 0, 1)
                #
                #         predicted_masks_1,_ = self.localizer(diffused_image_1)
                #         predicted_masks_1 = torch.clamp(predicted_masks_1, 0, 1)
                #         predicted_masks_2,_ = self.localizer(diffused_image_2)
                #         predicted_masks_2 = torch.clamp(predicted_masks_2, 0, 1)
                #
                #         tampered_attacked_image_1 = attacked_forward_1 * (1 - masks)
                #         tampered_attacked_image_2 = attacked_forward_2 * (1 - masks)
                #         # Out recovery
                #         # _, rectified_crop_padded_image = self.crop.cropped_out(tampered_attacked_image, min_rate=0.6, max_rate=0.9)
                #         rectified_crop_padded_image_1 = tampered_attacked_image_1
                #         rectified_crop_padded_image_2 = tampered_attacked_image_2
                #
                #         canny_input_1 = torch.zeros_like(self.canny_image).cuda()
                #         for icn in range(batch_size):
                #             img_GT = self.tensor_to_image(rectified_crop_padded_image[icn, :, :, :])
                #             img_gray = rgb2gray(img_GT)
                #             sigma = 2  # random.randint(1, 4)
                #             cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                #             cannied = self.image_to_tensor(cannied).cuda()
                #             canny_input_1[icn, :, :, :] = cannied
                #
                #         reversed_stuff_1,_ = self.netG(torch.cat((rectified_crop_padded_image_1,canny_input_1),dim=1), rev=True)
                #         reversed_ch1, reversed_ch2 = reversed_stuff_1[:, :3, :, :], reversed_stuff_1[:, 3:, :, :]
                #         reversed_image_1_ch1 = torch.clamp(reversed_ch1, 0, 1)
                #         reversed_image_1_ch2 = torch.clamp(reversed_ch2, 0, 1)
                #
                #         canny_input_2 = torch.zeros_like(self.canny_image).cuda()
                #         for icn in range(batch_size):
                #             img_GT = self.tensor_to_image(rectified_crop_padded_image[icn, :, :, :])
                #             img_gray = rgb2gray(img_GT)
                #             sigma = 2  # random.randint(1, 4)
                #             cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                #             cannied = self.image_to_tensor(cannied).cuda()
                #             canny_input_2[icn, :, :, :] = cannied
                #
                #         reversed_stuff_2,_ = self.netG(torch.cat((rectified_crop_padded_image_2,canny_input_2),dim=1), rev=True)
                #         reversed_ch1, reversed_ch2 = reversed_stuff_2[:, :3, :, :], reversed_stuff_2[:, 3:, :, :]
                #         reversed_image_2_ch1 = torch.clamp(reversed_ch1, 0, 1)
                #         reversed_image_2_ch2 = torch.clamp(reversed_ch2, 0, 1)
                #         # reversed_image_2 = reversed_image_2_ch1 * (1 - masks) + reversed_image_2_ch2 * masks
                #         # gen_fake_1, _ = self.discriminator(forward_image_1)
                #         # gen_fake_mask_1, _ = self.discriminator_mask(reversed_stuff_1)
                #         # gen_fake_mask_2, _ = self.discriminator_mask(reversed_stuff_2)
                #         images = stitch_images(
                #             self.postprocess(self.real_H),
                #             self.postprocess(self.canny_image),
                #             self.postprocess(forward_image_1),
                #             self.postprocess(10 * torch.abs(modified_input_1 - forward_image_1)),
                #             # self.postprocess(predicted_marked),
                #             self.postprocess(diffused_image_1),
                #             # self.postprocess(rectified_crop_padded_image),
                #             self.postprocess(masks_GT),
                #             self.postprocess(predicted_masks_1),
                #             self.postprocess(10 * torch.abs(masks_GT - predicted_masks_1)),
                #             self.postprocess(reversed_image_1_ch1),
                #             self.postprocess(reversed_image_1_ch2),
                #             self.postprocess(10 * torch.abs(modified_input - reversed_image_1_ch1)),
                #
                #             self.postprocess(predicted_masks_2),
                #             self.postprocess(10 * torch.abs(masks_GT - predicted_masks_2)),
                #             self.postprocess(reversed_image_2_ch1),
                #             self.postprocess(reversed_image_2_ch2),
                #             self.postprocess(10 * torch.abs(modified_input - reversed_image_2_ch1)),
                #
                #             # self.postprocess(gen_fake_1),
                #             # self.postprocess(gen_fake_mask_1),
                #             # self.postprocess(gen_fake_mask_2),
                #
                #             img_per_row=1
                #         )
                #
                #         out_space_storage = '/media/qichaoying/65F33762C14D581B/20220102_NOWAY/'
                #
                #         name = os.path.join('/media/qichaoying/65F33762C14D581B/20220102_NOWAY/images/', str(step).zfill(5) + "_"  + mixup1.name + mixup2.name + ".png")
                #         print('\nsaving sample ' + name)
                #         images.save(name)
                #
                #         ######## Save independent images #############
                #         name = os.path.join('/media/qichaoying/65F33762C14D581B/20220102_NOWAY/immunized_image/', str(step).zfill(5) + "_"  + attack_full_name + "_")
                #         for image_no in range(forward_image.shape[0]):
                #             camera_ready = forward_image[image_no].unsqueeze(0)
                #             torchvision.utils.save_image((camera_ready * 255).round() / 255,
                #                                          name + str(image_no) + ".png", nrow=1, padding=0,
                #                                          normalize=False)

        ######## Finally ####################
        if step % save_interval == 10:
            logger.info('Saving models and training states.')
            self.save(self.global_step)
        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H.clone().detach()
        return logs, debug_logs

    def evaluate(self):
        pass
        #
        # if eval_dir is not None:
        #     # eval
        #     ############## Evaluate ######################
        #     eval_data, water_data = eval_dir['eval_data'], eval_dir['water_data']
        #     val_path, water_path, save_path = eval_dir['val_path'], eval_dir['water_path'], eval_dir[
        #         'save_path']
        #     tamper_data, mask_data = eval_dir['tamper_data'], eval_dir['mask_data']
        #     source_tamper_path, predicted_mask_tamper_path, gt_mask_tamper_path = eval_dir[
        #                                                                               'source_tamper_path'], \
        #                                                                           eval_dir[
        #                                                                               'predicted_mask_tamper_path'], \
        #                                                                           eval_dir[
        #                                                                               'gt_mask_tamper_path']
        #     val_path = os.path.join(val_path, eval_data)
        #     water_path = os.path.join(water_path, water_data)
        #     source_tamper_path = os.path.join(source_tamper_path, tamper_data)
        #     gt_mask_tamper_path = os.path.join(gt_mask_tamper_path, mask_data)
        #
        #     # print("Tamper: {}  {}".format(tamper_data, mask_data))
        #
        #     tensor_c = self.load_image(path=val_path, grayscale=False)
        #     watermark_c = self.load_image(path=water_path, grayscale=True)
        #     source_tamper = self.load_image(path=source_tamper_path, grayscale=False)
        #     mask_tamper = self.load_image(path=gt_mask_tamper_path, grayscale=True)
        #
        #     forward_image = self.netG(x=tensor_c)
        #
        #     forward_image = self.Quantization(forward_image)
        #     forward_image = torch.clamp(forward_image, 0, 1)
        #     layer = self.jpeg90 if dist.get_rank() == 0 else self.jpeg70
        #     compressed_image = layer(forward_image)
        #     compressed_image = torch.clamp(compressed_image, 0, 1)
        #     tamper = compressed_image * (1 - mask_tamper) + source_tamper * mask_tamper
        #     tamper = torch.clamp(tamper, 0, 1)
        #     predicted_masks = self.localizer(tamper)
        #     rectify = tamper * (1 - mask_tamper)
        #
        #     reversed = self.netG(x=rectify, rev=True)
        #     reversed = torch.clamp(reversed, 0, 1)
        #
        #     name = os.path.join(
        #         save_path,
        #         os.path.splitext(eval_data)[0] + "_" + str(dist.get_rank()) + ("_realworld.png"))
        #     self.print_individual_image(reversed, name)
        #
        #     name = os.path.join(
        #         save_path,
        #         os.path.splitext(eval_data)[0] + str(dist.get_rank()) + "_diff.png")
        #     self.print_individual_image(10 * torch.abs(reversed - tensor_c), name)
        #
        #     name = os.path.join(
        #         save_path,
        #         os.path.splitext(eval_data)[0] + str(dist.get_rank()) + ("_pred.png"))
        #     self.print_individual_image(predicted_masks, name)
        #
        #     print("Saved: {}".format(name))
        #
        #     # name = os.path.join(
        #     #     save_path,
        #     #     os.path.splitext(eval_data)[0] + "_" + str(dist.get_rank()) + "_original.png")
        #     # self.print_individual_image(forward_image, name)
        #     # print("Saved: {}".format(name))

    def print_individual_image(self, cropped_GT, name):
        for image_no in range(cropped_GT.shape[0]):
            camera_ready = cropped_GT[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         name, nrow=1, padding=0, normalize=False)

    def load_image(self, path, grayscale):
        image_c = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1] if not grayscale else cv2.imread(path,
                                                                                                 cv2.IMREAD_GRAYSCALE)
        image_c = cv2.resize(image_c, dsize=(self.width_height, self.width_height), interpolation=cv2.INTER_LINEAR)
        img = image_c.copy().astype(np.float32)
        img /= 255.0
        if not grayscale:
            img = img.transpose(2, 0, 1)
        tensor_c = torch.from_numpy(img).unsqueeze(0).cuda()
        if grayscale:
            tensor_c = tensor_c.unsqueeze(0)

        return tensor_c

    def perform_attack(self, step, forward_image, masks, disable_hard=False):
        rand_step = step + np.random.randint(0, 10)
        if not disable_hard and rand_step % 10 <= 1:
            attack_layer = self.resize
        elif not disable_hard and rand_step % 10 <= 5:
            attack_layer = self.combined_jpeg_strong
        else:
            attack_layer = self.combined_jpeg_weak

        attacked_image = attack_layer(forward_image)
        if self.previous_images is not None:
            attacked_previous_images = attack_layer(self.previous_images).clone().detach()
            attacked_previous_images = torch.clamp(attacked_previous_images, 0, 1)

        attacked_clamp = torch.clamp(attacked_image, 0, 1)
        attacked_image = attacked_clamp

        diffused_image = attacked_image * (1 - masks) + (
            attacked_previous_images if self.previous_images is not None else torch.zeros_like(self.real_H)) * masks

        tampered_attacked_image = attacked_image * (1 - masks)
        # tampered_attacked_image = torch.clamp(tampered_attacked_image, 0, 1)

        return diffused_image, tampered_attacked_image, attack_layer, attacked_image

    def discrim_optimize(self, real_image, forward_image, discriminator, optimizer):
        # discriminator loss
        dis_input_real = real_image.clone().detach()
        dis_input_fake = forward_image.clone().detach()
        dis_real, dis_real_feat = discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, dis_fake_feat = discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        return dis_loss, dis_real_feat, dis_fake_feat

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

    def load(self):
        if self.opt['train']['load'] > 0.0:
            load_path_A = self.opt['path']['pretrain_model'] + "_A.pth"
            if load_path_A is not None:
                if self.opt['train']['load'] == 2.0:
                    load_path_A = '../experiments/pretrained_models/A_latest.pth'
                logger.info('Loading model for Additional Generator [{:s}] ...'.format(load_path_A))
                if os.path.exists(load_path_A):
                    self.load_network(load_path_A, self.generator_additional, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for A [{:s}] ...'.format(load_path_A))
            if self.task_name == self.TASK_TEST:
                load_path_A = self.opt['path']['pretrain_model'] + "_A_zxy.pth"
                if load_path_A is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_A = '../experiments/pretrained_models/A_zxy_latest.pth'
                    logger.info('Loading model for A [{:s}] ...'.format(load_path_A))
                    if os.path.exists(load_path_A):
                        self.load_network(load_path_A, self.attack_net, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for A [{:s}] ...'.format(load_path_A))
            elif self.task_name == self.TASK_IMUGEV2:
                load_path_G = self.opt['path']['pretrain_model'] + "_apex_zxy.pth"
                if load_path_G is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_G = '../experiments/pretrained_models/apex_zxy_latest.pth'
                    logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                    if os.path.exists(load_path_G):
                        self.load_network(load_path_G, self.CropPred_net, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))

                load_path_A = self.opt['path']['pretrain_model'] + "_A_zxy.pth"
                if load_path_A is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_A = '../experiments/pretrained_models/A_zxy_latest.pth'
                    logger.info('Loading model for A [{:s}] ...'.format(load_path_A))
                    if os.path.exists(load_path_A):
                        self.load_network(load_path_A, self.attack_net, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for A [{:s}] ...'.format(load_path_A))

                load_path_D = self.opt['path']['pretrain_model'] + "_D_zxy.pth"
                if load_path_D is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_D = '../experiments/pretrained_models/D_zxy_latest.pth'
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                    if os.path.exists(load_path_D):
                        self.load_network(load_path_D, self.discriminator, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))

                load_path_D = self.opt['path']['pretrain_model'] + "_D_mask_zxy.pth"
                if load_path_D is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_D = '../experiments/pretrained_models/D_mask_zxy_latest.pth'
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                    if os.path.exists(load_path_D):
                        self.load_network(load_path_D, self.discriminator_mask, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))

                load_path_G = self.opt['path']['pretrain_model'] + "_G.pth"
                if load_path_G is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_G = '../experiments/pretrained_models/G_latest.pth'
                    logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                    if os.path.exists(load_path_G):
                        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))

                load_path_L = self.opt['path']['pretrain_model'] + "_L.pth"
                if load_path_L is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_L = '../experiments/pretrained_models/L_latest.pth'
                    logger.info('Loading model for L [{:s}] ...'.format(load_path_L))
                    if os.path.exists(load_path_L):
                        self.load_network(load_path_L, self.localizer, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for L [{:s}] ...'.format(load_path_L))

                load_path_G = self.opt['path']['pretrain_model'] + "_G_zxy.pth"
                if load_path_G is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_G = '../experiments/pretrained_models/G_zxy_latest.pth'
                    logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                    if os.path.exists(load_path_G):
                        self.load_network(load_path_G, self.generator, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))

                load_path_D = self.opt['path']['pretrain_model'] + "_dis_adv_cov.pth"
                if load_path_D is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_D = '../experiments/pretrained_models/dis_adv_cov_latest.pth'
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                    if os.path.exists(load_path_D):
                        self.load_network(load_path_D, self.dis_adv_cov, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))

                load_path_D = self.opt['path']['pretrain_model'] + "_dis_adv_fw.pth"
                if load_path_D is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_D = '../experiments/pretrained_models/dis_adv_fw_latest.pth'
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                    if os.path.exists(load_path_D):
                        self.load_network(load_path_D, self.dis_adv_fw, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))

            elif self.task_name == self.TASK_CropLocalize:
                #### netG localizer attack_net generator discriminator discriminator_mask CropPred_net dis_adv_fw dis_adv_cov

                load_path_L = self.opt['path']['pretrain_model'] + "_L_zxy.pth"
                if load_path_L is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_L = '../experiments/pretrained_models/L_zxy_latest.pth'
                    logger.info('Loading model for L [{:s}] ...'.format(load_path_L))
                    if os.path.exists(load_path_L):
                        self.load_network(load_path_L, self.localizer, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for L [{:s}] ...'.format(load_path_L))

                load_path_A = self.opt['path']['pretrain_model'] + "_A_zxy.pth"
                if load_path_A is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_A = '../experiments/pretrained_models/A_zxy_latest.pth'
                    logger.info('Loading model for A [{:s}] ...'.format(load_path_A))
                    if os.path.exists(load_path_A):
                        self.load_network(load_path_A, self.attack_net, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for A [{:s}] ...'.format(load_path_A))

                load_path_G = self.opt['path']['pretrain_model'] + "_G_zxy.pth"
                if load_path_G is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_G = '../experiments/pretrained_models/G_zxy_latest.pth'
                    logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                    if os.path.exists(load_path_G):
                        self.load_network(load_path_G, self.generator, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))

                load_path_D = self.opt['path']['pretrain_model'] + "_D_zxy.pth"
                if load_path_D is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_D = '../experiments/pretrained_models/D_zxy_latest.pth'
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                    if os.path.exists(load_path_D):
                        self.load_network(load_path_D, self.discriminator, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))

                load_path_D = self.opt['path']['pretrain_model'] + "_D_mask_zxy.pth"
                if load_path_D is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_D = '../experiments/pretrained_models/D_mask_zxy_latest.pth'
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                    if os.path.exists(load_path_D):
                        self.load_network(load_path_D, self.discriminator_mask, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))

                load_path_G = self.opt['path']['pretrain_model'] + "_G.pth"
                if load_path_G is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_G = '../experiments/pretrained_models/G_latest.pth'
                    logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                    if os.path.exists(load_path_G):
                        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))

                load_path_G = self.opt['path']['pretrain_model'] + "_apex_zxy.pth"
                if load_path_G is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_G = '../experiments/pretrained_models/apex_zxy_latest.pth'
                    logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                    if os.path.exists(load_path_G):
                        self.load_network(load_path_G, self.CropPred_net, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))

                load_path_D = self.opt['path']['pretrain_model'] + "_dis_adv_cov.pth"
                if load_path_D is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_D = '../experiments/pretrained_models/dis_adv_cov_latest.pth'
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                    if os.path.exists(load_path_D):
                        self.load_network(load_path_D, self.dis_adv_cov, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))

                load_path_D = self.opt['path']['pretrain_model'] + "_dis_adv_fw.pth"
                if load_path_D is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_D = '../experiments/pretrained_models/dis_adv_fw_latest.pth'
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                    if os.path.exists(load_path_D):
                        self.load_network(load_path_D, self.dis_adv_fw, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))

            else:
                load_path_D = self.opt['path']['pretrain_model'] + "_D_mask_zxy.pth"
                if load_path_D is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_D = '../experiments/pretrained_models/D_mask_zxy_latest.pth'
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                    if os.path.exists(load_path_D):
                        self.load_network(load_path_D, self.discriminator_mask, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for D [{:s}] ...'.format(load_path_D))

                load_path_G = self.opt['path']['pretrain_model'] + "_G.pth"
                if load_path_G is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_G = '../experiments/pretrained_models/G_latest.pth'
                    logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                    if os.path.exists(load_path_G):
                        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))

    def save(self, iter_label):
        out_space_storage = '/media/qichaoying/65F33762C14D581B/20220102_NOWAY/'
        self.save_network(self.netG, 'netG', iter_label, model_path=out_space_storage)
        self.save_network(self.localizer, 'localizer', iter_label, model_path=out_space_storage)
        self.save_network(self.discriminator, 'discriminator', iter_label, model_path=out_space_storage)
        self.save_network(self.discriminator_mask, 'discriminator_mask', iter_label, model_path=out_space_storage)
        self.save_network(self.dis_adv_cov, 'dis_adv_cov', iter_label, model_path=out_space_storage)
        self.save_network(self.generator, 'domain', iter_label, model_path=out_space_storage)

    def generate_stroke_mask(self, im_size, parts=5, parts_square=2, maxVertex=4, maxLength=64, maxBrushWidth=32,
                             maxAngle=360, percent_range=(0.2, 0.3)):
        maxLength = int(im_size[0] / 5)
        maxBrushWidth = int(im_size[0] / 5)
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
