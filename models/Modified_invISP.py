import logging
from collections import OrderedDict
import copy
import torch
import torchvision.transforms.functional_tensor as F_t
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as Functional
from noise_layers.salt_pepper_noise import SaltPepper
import torchvision.transforms.functional_pil as F_pil
from skimage.feature import canny
import torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.nn.parallel import DataParallel, DistributedDataParallel
from skimage.color import rgb2gray
from skimage.metrics._structural_similarity import structural_similarity
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, CWLoss
from models.modules.Quantization import Quantization, diff_round
import torch.distributed as dist
from utils.JPEG import DiffJPEG
from torchvision import models
from loss import AdversarialLoss, PerceptualLoss, StyleLoss
import cv2
from mbrs_models.Encoder_MP import Encoder_MP
from metrics import PSNR, EdgeAccuracy
from .invertible_net import Inveritible_Decolorization_PAMI, ResBlock, DenseBlock, Haar_UNet
from .crop_localize_net import CropLocalizeNet
from .conditional_jpeg_generator import FBCNN, MantraNet, QF_predictor
from utils import Progbar, create_dir, stitch_images, imsave
import os
import pytorch_ssim
from noise_layers import *
from noise_layers.dropout import Dropout
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.middle_filter import MiddleBlur
from noise_layers.resize import Resize
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.crop import Crop
from models.networks import EdgeGenerator, DG_discriminator, InpaintGenerator, Discriminator, NormalGenerator, UNetDiscriminator, \
    JPEGGenerator, Localizer
from MVSS.inference import get_opt
from MVSS.models.mvssnet import get_mvss
from MVSS.models.resfcn import ResFCN
from mbrs_models.Decoder import Decoder, Decoder_MLP
# import matlab.engine
from mbrs_models.baluja_networks import HidingNetwork, RevealNetwork
from pycocotools.coco import COCO
from models.conditional_jpeg_generator import domain_generalization_predictor
from loss import ExclusionLoss

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
import data
# import lpips
from losses.fourier_loss import fft_L1_loss_color, fft_L1_loss_mask, decide_circle
from torchstat import stat
from MantraNet.mantranet import pre_trained_model

class Modified_invISP(BaseModel):
    def __init__(self, opt,args):

        super(Modified_invISP, self).__init__(opt)
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        ########### CONSTANTS ###############
        self.TASK_IMUGEV2 = "ImugeV2"
        self.TASK_TEST = "Test"
        self.TASK_CropLocalize = "CropLocalize"
        self.TASK_RHI3 = "RHI3"

        self.rank = torch.distributed.get_rank()

        self.gpu_id = self.opt['gpu_ids'][0]
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.args = args
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.real_H, self.real_H_path, self.previous_images, self.previous_previous_images = None, None, None, None
        self.previous_canny = None
        self.task_name = self.opt['datasets']['train']['name'] #self.train_opt['task_name']
        print("Task Name: {}".format(self.task_name))
        self.global_step = 0
        self.new_task = self.train_opt['new_task']


        ####################################################################################################
        # todo: losses and attack layers
        # todo: JPEG attack rescaling deblurring
        ####################################################################################################
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
        self.width_height = opt['datasets']['train']['GT_size']
        self.jpeg_simulate = [
            [DiffJPEG(50, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(55, height=self.width_height, width=self.width_height).cuda(), ]
            ,[DiffJPEG(60, height=self.width_height, width=self.width_height).cuda(), ]
            ,[DiffJPEG(65, height=self.width_height, width=self.width_height).cuda(), ]
            ,[DiffJPEG(70, height=self.width_height, width=self.width_height).cuda(), ]
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
        # self.perceptual_loss = PerceptualLoss().cuda()
        # self.style_loss = StyleLoss().cuda()
        self.Quantization = diff_round
        # self.Quantization = Quantization().cuda()
        # self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw']).cuda()
        # self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back']).cuda()
        # self.criterion_adv = CWLoss().cuda()  # loss for fooling target model
        self.CE_loss = nn.CrossEntropyLoss().cuda()
        self.width_height = opt['datasets']['train']['GT_size']
        self.init_gaussian = None
        # self.adversarial_loss = AdversarialLoss(type="nsgan").cuda()

        ####################################################################################################
        # todo: MAIN Networks
        # todo: including ...
        ####################################################################################################
        self.network_list = []
        if self.args.mode in {1}:
            ####################################################################################################
            # todo: MAIN Task
            # todo: networks include ...
            ####################################################################################################
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
            self.discriminator_mask = UNetDiscriminator(in_channels=4, residual_blocks=1) #Localizer().cuda()
            self.discriminator_mask = self.discriminator_mask.cuda()
            self.discriminator_mask = DistributedDataParallel(self.discriminator_mask,
                                                              device_ids=[torch.cuda.current_device()],
                                                              ) # find_unused_parameters=True

            self.scaler_localizer = torch.cuda.amp.GradScaler()
            self.scaler_G = torch.cuda.amp.GradScaler()
            self.scaler_discriminator_mask = torch.cuda.amp.GradScaler()
            self.scaler_generator = torch.cuda.amp.GradScaler()

            ####################################################################################################
            # todo: Image Signal Processing Pipeline
            # todo: invISP
            ####################################################################################################

        elif self.args.mode in {0}:
            self.network_list = ['netG', 'localizer', 'discriminator_mask']
            print(f"network list:{self.network_list}")
            ####################################################################################################
            # todo: Image Manipulation Detection Network (Downstream task)
            # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
            ####################################################################################################
            print("Building MantraNet...........please wait...")
            self.localizer = pre_trained_model(weight_path='./MantraNetv4.pt').cuda()
            self.localizer = DistributedDataParallel(self.localizer, device_ids=[torch.cuda.current_device()],
                                                find_unused_parameters=True)

            print("Building MVSS...........please wait...")
            model_path = './MVSS/ckpt/mvssnet_casia.pt'
            self.netG = get_mvss(backbone='resnet50',
                                      pretrained_base=True,
                                      nclass=1,
                                      sobel=True,
                                      constrain=True,
                                      n_input=3,
                                      ).cuda()
            checkpoint = torch.load(model_path, map_location='cpu')
            self.netG.load_state_dict(checkpoint, strict=True)
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],
                                                     find_unused_parameters=True)
            print("Building ResFCN...........please wait...")
            self.discriminator_mask = ResFCN().cuda()
            self.discriminator_mask = DistributedDataParallel(self.discriminator_mask,
                                                              device_ids=[torch.cuda.current_device()],
                                                              find_unused_parameters=True)
            ## we found no checkpoint in the official repo currently

            self.scaler_localizer = torch.cuda.amp.GradScaler()
            self.scaler_G = torch.cuda.amp.GradScaler()
            self.scaler_discriminator_mask = torch.cuda.amp.GradScaler()

        else:
            raise NotImplementedError("args.mode value error! please check...")

        ####################################################################################################
        # todo: Optmizers
        # todo: invISP
        ####################################################################################################
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

        if 'netG' in self.network_list:
            self.optimizer_G = self.create_optimizer(self.netG,
                                  lr=train_opt['lr_G'], weight_decay=wd_G)
        if 'discriminator_mask' in self.network_list:
            self.optimizer_discriminator_mask = self.create_optimizer(self.discriminator_mask,
                                  lr=train_opt['lr_G'], weight_decay=wd_G)
        if 'localizer' in self.network_list:
            self.optimizer_localizer = self.create_optimizer(self.localizer,
                                  lr=train_opt['lr_G'], weight_decay=wd_G)
        # if 'KD_JPEG_net' in self.network_list:
        #     self.create_optimizer(self.KD_JPEG_net, self.optimizer_KD_JPEG,
        #                           lr=train_opt['lr_G'], weight_decay=wd_G)
        if 'discriminator' in self.network_list:
            self.optimizer_discriminator = self.create_optimizer(self.discriminator,
                                  lr=train_opt['lr_G'], weight_decay=wd_G)
        # if 'generator' in self.network_list:
        #     self.create_optimizer(self.generator, self.optimizer_generator,
        #                           lr=train_opt['lr_G'], weight_decay=wd_G)
        # if 'qf_predict_network' in self.network_list:
        #     self.create_optimizer(self.qf_predict_network, self.optimizer_qf_predict,
        #                           lr=train_opt['lr_G'], weight_decay=wd_G)

        ####################################################################################################
        # todo: Scheduler
        # todo: invISP
        ####################################################################################################
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=118287))

        ####################################################################################################
        # todo: init constants
        # todo: invISP
        ####################################################################################################
        self.forward_image_buff = None
        self.reloaded_time = 0
        self.basic_weight_fw = 5

        ####################################################################################################
        # todo: Loading Pretrained models
        # todo: invISP
        ####################################################################################################


        # good_models: '/model/Rerun_4/29999'
        self.out_space_storage = '/home/groupshare/ISP_results/tamper_results'
        self.model_storage = f'/model/{self.task_name}/'
        self.model_path = str(42999) # 29999

        load_models = False
        load_state = False
        if load_models:
            self.pretrain = self.out_space_storage + self.model_storage + self.model_path
            self.reload(self.pretrain, self.network_list)
            ## load states
            state_path = self.out_space_storage + self.model_storage + '{}.state'.format(self.model_path)
            if load_state:
                logger.info('Loading training state')
                if os.path.exists(state_path):
                    self.resume_training(state_path, self.network_list)
                else:
                    logger.info('Did not find state [{:s}] ...'.format(state_path))

    def create_optimizer(self, net, lr=1e-4, weight_decay=0):
        ## lr should be train_opt['lr_G'] in default
        optim_params = []
        for k, v in net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        optimizer = torch.optim.AdamW(optim_params, lr=lr,
                                             weight_decay=weight_decay,
                                             betas=(0.9, 0.99))  # train_opt['beta1'], train_opt['beta2']
        self.optimizers.append(optimizer)

        return optimizer

    def clamp_with_grad(self,tensor):
        tensor_clamp = torch.clamp(tensor,0,1)
        return tensor+ (tensor_clamp-tensor).clone().detach()

    def feed_data(self, batch):
        img, label, canny_image = batch
        self.real_H = img.cuda()
        self.canny_image = canny_image.cuda()

    def gaussian_batch(self, dims):
        return self.clamp_with_grad(torch.randn(tuple(dims)).cuda())

    def optimize_parameters(self, step, latest_values=None, train=True, eval_dir=None):
        ####################################################################################################
        # todo: Image Manipulation Detection Network (Downstream task)
        # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
        ####################################################################################################
        self.netG.train()
        self.localizer.train()
        self.discriminator_mask.train()

        logs, debug_logs = [], []

        self.real_H = self.clamp_with_grad(self.real_H)
        batch_size, num_channels, height_width, _ = self.real_H.shape
        lr = self.get_current_learning_rate()
        logs.append(('lr', lr))

        modified_input = self.real_H.clone().detach()
        modified_input = self.clamp_with_grad(modified_input)

        if not (self.previous_images is None or self.previous_previous_images is None):

            with torch.cuda.amp.autocast():

                percent_range = (0.05, 0.25)
                masks, masks_GT = self.mask_generation(percent_range=percent_range,logs=logs)

                attacked_forward = self.tampering(
                    forward_image=modified_input, masks=masks, masks_GT=masks_GT,
                    modified_input=modified_input, percent_range=percent_range, logs=logs)

                if self.global_step % 5 in {0, 1, 2}:
                    quality_idx = np.random.randint(19, 21)
                else:
                    quality_idx = np.random.randint(12, 17)
                attacked_image = self.benign_attacks(attacked_forward=attacked_forward, logs=logs, quality_idx=quality_idx)


                _, pred_mvss = self.netG(attacked_image.detach())
                CE_MVSS = self.bce_with_logit_loss(pred_mvss, masks_GT)

                pred_mantra = self.localizer(attacked_image.detach())
                CE_mantra = self.bce_with_logit_loss(pred_mantra, masks_GT)

                _, pred_resfcn = self.discriminator_mask(attacked_image.detach())
                CE_resfcn = self.bce_with_logit_loss(pred_resfcn, masks_GT)


                logs.append(('CE_MVSS', CE_MVSS.item()))
                logs.append(('CE_mantra', CE_mantra.item()))
                logs.append(('CE_resfcn', CE_resfcn.item()))


            ####################################################################################################
            # todo: STEP
            # todo: invISP
            ####################################################################################################
            self.optimizer_G.zero_grad()
            self.optimizer_localizer.zero_grad()
            self.optimizer_discriminator_mask.zero_grad()
            # loss.backward()
            self.scaler_G.scale(CE_MVSS).backward()
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netG.parameters(), 1)
            # self.optimizer_G.step()
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()

            self.optimizer_localizer.zero_grad()
            # CE_train.backward()
            self.scaler_localizer.scale(CE_mantra).backward()
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.localizer.parameters(), 1)
            # self.optimizer_localizer.step()
            self.scaler_localizer.step(self.optimizer_localizer)
            self.scaler_localizer.update()

            self.optimizer_discriminator_mask.zero_grad()
            # dis_loss.backward()
            self.scaler_discriminator_mask.scale(CE_resfcn).backward()
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
            # self.optimizer_discriminator_mask.step()
            self.scaler_discriminator_mask.step(self.optimizer_discriminator_mask)
            self.scaler_discriminator_mask.update()


            ####################################################################################################
            # todo: observation zone
            # todo: invISP
            ####################################################################################################
            # with torch.no_grad():
            #     REVERSE, _ = self.netG(torch.cat((attacked_real_jpeg * (1 - masks),
            #                    torch.zeros_like(modified_canny).cuda()), dim=1), rev=True)
            #     REVERSE = self.clamp_with_grad(REVERSE)
            #     REVERSE = REVERSE[:, :3, :, :]
            #     l_REV = (self.l1_loss(REVERSE * masks_expand, modified_input * masks_expand))
            #     logs.append(('observe', l_REV.item()))

            ####################################################################################################
            # todo: printing the images
            # todo: invISP
            ####################################################################################################
            anomalies = False #CE_recall.item()>0.5
            if anomalies or self.global_step % 200 == 3 or self.global_step<=10:
                images = stitch_images(
                    self.postprocess(modified_input),

                    self.postprocess(attacked_forward),
                    self.postprocess(attacked_image),
                    self.postprocess(10 * torch.abs(attacked_forward - attacked_image)),
                    self.postprocess(masks_GT),
                    self.postprocess(torch.sigmoid(pred_mvss)),
                    self.postprocess(10 * torch.abs(masks_GT - torch.sigmoid(pred_mvss))),
                    self.postprocess(torch.sigmoid(pred_mantra)),
                    self.postprocess(10 * torch.abs(masks_GT - torch.sigmoid(pred_mantra))),
                    self.postprocess(torch.sigmoid(pred_resfcn)),
                    self.postprocess(10 * torch.abs(masks_GT - torch.sigmoid(pred_resfcn))),
                    img_per_row=1
                )

                name = self.out_space_storage + '/images/'+self.task_name+'_'+str(self.gpu_id)+'/'\
                       +str(self.global_step).zfill(5) + "_ "+str(self.gpu_id) + "_ "+str(self.rank) \
                       +("" if not anomalies else "_anomaly")+ ".png"
                print('\nsaving sample ' + name)
                images.save(name)

        ####################################################################################################
        # todo: updating the training stage
        # todo: invISP
        ####################################################################################################
        ######## Finally ####################
        if self.global_step % 1000== 999 or self.global_step==9:
            if self.rank==0:
                logger.info('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.network_list)
        if self.real_H is not None:
            self.previous_canny = self.canny_image
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H
        self.global_step = self.global_step + 1
        return logs, debug_logs

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

    def tamper_based_augmentation(self,modified_input,modified_canny,masks,masks_GT,logs):
        # tamper-based data augmentation
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        for imgs in range(batch_size):
            if imgs%3!=2:
                modified_input[imgs, :, :, :] = (modified_input[imgs, :, :, :] * (1 - masks[imgs, :, :, :]) + self.previous_images[imgs, :, :, :] * masks[imgs, :, :, :]).clone().detach()
                modified_canny[imgs, :, :, :] = (modified_canny[imgs, :, :, :] * (1 - masks_GT[imgs, :, :, :]) + self.previous_canny[imgs, :, :, :] * masks_GT[imgs, :, :, :]).clone().detach()

        return modified_input, modified_canny


    def mask_generation(self,percent_range,logs):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        masks_GT = torch.zeros(batch_size, 1, self.real_H.shape[2], self.real_H.shape[3]).cuda()
        ## THE RECOVERY STAGE WILL ONLY WORK UNDER LARGE TAMPERING
        ## TO LOCALIZE SMALL TAMPERING, WE ONLY UPDATE LOCALIZER NETWORK

        for imgs in range(batch_size):
            if imgs%3==2:
                ## copy-move will not be too large
                percent_range = (0.00,0.15)
            masks_origin, _ = self.generate_stroke_mask(
                [self.real_H.shape[2], self.real_H.shape[3]], percent_range=percent_range)
            masks_GT[imgs, :, :, :] = masks_origin.cuda()
        masks = masks_GT.repeat(1, 3, 1, 1)

        # masks is just 3-channel-version masks_GT
        return masks, masks_GT

    def forward_image_generation(self,modified_input,modified_canny,logs):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        forward_stuff = self.netG(x=torch.cat((modified_input, modified_canny), dim=1))
        forward_stuff = self.clamp_with_grad(forward_stuff)
        forward_image, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
        psnr_forward = self.psnr(self.postprocess(modified_input), self.postprocess(forward_image)).item()

        return forward_image, forward_null, psnr_forward

    def tampering(self, forward_image, masks, masks_GT, modified_input, percent_range, logs):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        ####### Tamper ###############
        attacked_forward = torch.zeros_like(forward_image)
        for img_idx in range(batch_size):
            way_tamper = img_idx%3


            if way_tamper == 0:
                ####################################################################################################
                # todo: splicing
                # todo: invISP
                ####################################################################################################
                attacked_forward[img_idx:img_idx + 1] = forward_image[img_idx:img_idx + 1] * (1 - masks[img_idx:img_idx + 1]) \
                                                        + self.previous_previous_images[img_idx:img_idx + 1] * masks[img_idx:img_idx + 1]
                # attack_name = "splicing"

            else:
                ####################################################################################################
                # todo: copy-move
                # todo: invISP
                ####################################################################################################
                lower_bound_percent = percent_range[0] + (percent_range[1] - percent_range[0]) * np.random.rand()
                tamper = forward_image[img_idx:img_idx+1].clone().detach()
                x_shift, y_shift, valid, retried, max_valid, mask_buff = 0, 0, 0, 0, 0, None
                while retried<20 and not (valid>lower_bound_percent and (abs(x_shift)>(modified_input.shape[2]/3) or abs(y_shift)>(modified_input.shape[3]/3))):
                    x_shift = int((modified_input.shape[2]) * (np.random.rand() - 0.5))
                    y_shift = int((modified_input.shape[3]) * (np.random.rand() - 0.5))

                    ### two times padding ###
                    mask_buff = torch.zeros((masks[img_idx:img_idx+1].shape[0], masks.shape[1],
                                                masks.shape[2] + abs(2 * x_shift),
                                                masks.shape[3] + abs(2 * y_shift))).cuda()

                    mask_buff[:, :,
                    abs(x_shift) + x_shift:abs(x_shift) + x_shift + modified_input.shape[2],
                    abs(y_shift) + y_shift:abs(y_shift) + y_shift + modified_input.shape[3]] = masks[img_idx:img_idx+1]

                    mask_buff = mask_buff[:, :,
                                        abs(x_shift):abs(x_shift) + modified_input.shape[2],
                                        abs(y_shift):abs(y_shift) + modified_input.shape[3]]

                    valid = torch.mean(mask_buff)
                    retried += 1
                    if valid>=max_valid:
                        max_valid = valid
                        self.mask_shifted = mask_buff
                        self.x_shift, self.y_shift = x_shift, y_shift

                self.tamper_shifted = torch.zeros((modified_input[img_idx:img_idx+1].shape[0], modified_input.shape[1],
                                                   modified_input.shape[2] + abs(2 * self.x_shift),
                                                   modified_input.shape[3] + abs(2 * self.y_shift))).cuda()
                self.tamper_shifted[:, :, abs(self.x_shift) + self.x_shift: abs(self.x_shift) + self.x_shift + modified_input.shape[2],
                abs(self.y_shift) + self.y_shift: abs(self.y_shift) + self.y_shift + modified_input.shape[3]] = tamper


                self.tamper_shifted = self.tamper_shifted[:, :,
                                 abs(self.x_shift): abs(self.x_shift) + modified_input.shape[2],
                                 abs(self.y_shift): abs(self.y_shift) + modified_input.shape[3]]

                masks[img_idx:img_idx+1] = self.mask_shifted.clone().detach()
                masks[img_idx:img_idx+1] = self.clamp_with_grad(masks[img_idx:img_idx+1])
                valid = torch.mean(masks[img_idx:img_idx+1])

                masks_GT[img_idx:img_idx+1] = masks[img_idx:img_idx+1, :1, :, :]
                attacked_forward[img_idx:img_idx+1] = forward_image[img_idx:img_idx+1] * (1 - masks[img_idx:img_idx+1]) + self.tamper_shifted.clone().detach() * masks[img_idx:img_idx+1]
                del self.tamper_shifted
                del self.mask_shifted

        attacked_forward = self.clamp_with_grad(attacked_forward)
        # attacked_forward = self.Quantization(attacked_forward)

        return attacked_forward

    def benign_attacks(self, attacked_forward, quality_idx, logs):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        attacked_real_jpeg = torch.rand_like(attacked_forward).cuda()

        if self.global_step % 5 == 1:
            blurring_layer = self.gaussian_blur
        elif self.global_step % 5 == 2:
            blurring_layer = self.median_blur
        elif self.global_step % 5 == 0:
            blurring_layer = self.resize
        else:
            blurring_layer = self.identity

        quality = int(quality_idx * 5)

        jpeg_layer_after_blurring = self.jpeg_simulate[quality_idx - 10][0] if quality < 100 else self.identity
        attacked_real_jpeg_simulate = self.Quantization(self.clamp_with_grad(jpeg_layer_after_blurring(blurring_layer(attacked_forward))))
        if self.global_step % 5 == 4:
            attacked_image = attacked_real_jpeg_simulate
        else:  # if self.global_step%5==3:
            for idx_atkimg in range(batch_size):
                grid = attacked_forward[idx_atkimg]
                realworld_attack = self.real_world_attacking_on_ndarray(grid, quality)
                attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

            attacked_real_jpeg = attacked_real_jpeg.clone().detach()
            attacked_image = attacked_real_jpeg_simulate + (attacked_real_jpeg - attacked_real_jpeg_simulate).clone().detach()

        # error_scratch = attacked_real_jpeg - attacked_forward
        # l_scratch = self.l1_loss(error_scratch, torch.zeros_like(error_scratch).cuda())
        # logs.append(('SCRATCH', l_scratch.item()))
        return attacked_image

    def benign_attacks_without_simulation(self, forward_image, quality_idx, logs):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        attacked_real_jpeg = torch.rand_like(forward_image).cuda()

        quality = int(quality_idx * 5)

        for idx_atkimg in range(batch_size):
            grid = forward_image[idx_atkimg]
            realworld_attack = self.real_world_attacking_on_ndarray(grid, quality)
            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        return attacked_real_jpeg

    def real_world_attacking_on_ndarray(self,grid, qf_after_blur, index=None):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        if index is None:
            index = self.global_step % 5
        if index == 0:
            grid = self.resize(grid.unsqueeze(0))[0]
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        if index == 1:
            realworld_attack = cv2.GaussianBlur(ndarr, (5, 5), 0)
        elif index == 2:
            realworld_attack = cv2.medianBlur(ndarr, 5)
        else:
            realworld_attack = ndarr
        if qf_after_blur!=100:
            _, realworld_attack = cv2.imencode('.jpeg', realworld_attack, (int(cv2.IMWRITE_JPEG_QUALITY), qf_after_blur))
            realworld_attack = cv2.imdecode(realworld_attack, cv2.IMREAD_UNCHANGED)
        # realworld_attack = data.util.channel_convert(realworld_attack.shape[2], 'RGB', [realworld_attack])[0]
        # realworld_attack = cv2.resize(copy.deepcopy(realworld_attack), (height_width, height_width),
        #                               interpolation=cv2.INTER_LINEAR)
        realworld_attack = realworld_attack.astype(np.float32) / 255.
        realworld_attack = torch.from_numpy(
            np.ascontiguousarray(np.transpose(realworld_attack, (2, 0, 1)))).float()
        realworld_attack = realworld_attack.unsqueeze(0).cuda()
        return realworld_attack

    def localization_loss(self,model, attacked_image,forward_image,masks_GT,modified_input,attacked_forward,logs):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        ### LOCALIZATION
        gen_attacked_train = model(attacked_image)
        # gen_non_attacked = self.localizer(forward_image)
        # gen_not_protected = self.localizer(modified_input)
        # gen_attacked_non_compress = self.localizer(attacked_forward)
        CE_recall = self.bce_with_logit_loss(gen_attacked_train, masks_GT)
        # CE_precision = self.bce_with_logit_loss(gen_non_attacked, torch.zeros_like(masks_GT))
        # CE_valid = self.bce_with_logit_loss(gen_not_protected, torch.ones_like(masks_GT))
        # CE_non_compress = self.bce_with_logit_loss(gen_attacked_non_compress, masks_GT)
        CE = 0
        # CE += 0.2*CE_precision
        CE += CE_recall
        # CE += 0.2*CE_valid
        # CE += (CE_precision + CE_valid + CE_non_compress) / 3

        # return CE, CE_recall, CE_non_compress, CE_precision, CE_valid, gen_attacked_train
        return CE, CE_recall, None, CE_recall, CE_recall, gen_attacked_train, gen_attacked_train, gen_attacked_train

    def recovery_image_generation(self,attacked_image,masks,modified_canny, logs):
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

        # reversed_image = modified_expand*(1-masks_expand)+reversed_image*masks_expand
        # reversed_canny = canny_expanded*(1-masks_expand[:,:1])+reversed_canny*masks_expand[:,:1]
        # reversed_image_ideal = modified_input*(1-masks)+reversed_image_ideal*masks
        # reversed_canny_ideal = modified_canny * (1 - masks) + reversed_canny_ideal * masks

        return reversed_image, reversed_canny

    def GAN_loss(self,model, reversed_image, reversed_canny, modified_input, logs):
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

    def GAN_training(self,model, modified_input,modified_canny,reversed_image,reversed_canny,masks_GT,logs):
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

                ####################################################################################################
                # todo: loss for student
                # todo: invISP
                ####################################################################################################
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

                ####################################################################################################
                # todo: loss for teacher
                # todo: invISP
                ####################################################################################################
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

                ####################################################################################################
                # todo: LOSSES for QF predictor
                # todo: invISP
                ####################################################################################################
                loss_qf = 0
                loss_qf += self.CE_loss(QF_real, label)
                logs.append(('qf_ce', loss_qf.item()))

            ####################################################################################################
            # todo: STEP
            # todo: invISP
            ####################################################################################################
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
            ####################################################################################################
            # todo: Print images
            # todo: invISP
            ####################################################################################################
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

                name = self.out_space_storage + '/jpeg_images/' + self.task_name + '_' + str(self.gpu_id) + '/' \
                       + str(self.global_step).zfill(5) + "_ " + str(self.gpu_id) + "_ " + str(self.rank) \
                       + ("" if not anomalies else "_anomaly") + ".png"
                print('\nsaving sample ' + name)
                images.save(name)

        ####################################################################################################
        # todo: updating training stage
        # todo: invISP
        ####################################################################################################
        if self.global_step % 1000 == 999 or self.global_step==9:
            if self.rank == 0:
                logger.info('Saving models and training states.')
                self.save(self.global_step, folder='jpeg_model', network_list=self.network_list)
        if self.real_H is not None:
            self.previous_canny = self.canny_image
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H
        self.global_step = self.global_step + 1
        return logs, debug_logs



    def evaluate(self,data_origin=None,data_immunize=None,data_tampered=None,data_tampersource=None,data_mask=None):
        self.netG.eval()
        self.localizer.eval()
        with torch.no_grad():
            psnr_forward_sum, psnr_backward_sum = [0,0,0,0,0],  [0,0,0,0,0]
            ssim_forward_sum, ssim_backward_sum =  [0,0,0,0,0],  [0,0,0,0,0]
            F1_sum =  [0,0,0,0,0]
            valid_images = [0,0,0,0,0]
            logs, debug_logs = [], []
            image_list_origin = None if data_origin is None else self.get_paths_from_images(data_origin)
            image_list_immunize = None if data_immunize is None else self.get_paths_from_images(data_immunize)
            image_list_tamper = None if data_tampered is None else self.get_paths_from_images(data_tampered)
            image_list_tampersource = None if data_tampersource is None else self.get_paths_from_images(data_tampersource)
            image_list_mask = None if data_mask is None else self.get_paths_from_images(data_mask)

            for idx in range(len(image_list_origin)):

                p, q, r = image_list_origin[idx]
                ori_path = os.path.join(p, q, r)
                img_GT = self.load_image(ori_path)
                print("Ori: {} {}".format(ori_path, img_GT.shape))
                self.real_H = self.img_random_crop(img_GT, 608, 608).cuda().unsqueeze(0)
                self.real_H = self.clamp_with_grad(self.real_H)
                img_gray = rgb2gray(img_GT)
                sigma = 2  # random.randint(1, 4)
                cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                self.canny_image = self.image_to_tensor(cannied).cuda().unsqueeze(0)

                p, q, r = image_list_immunize[idx]
                immu_path = os.path.join(p, q, r)
                img_GT = self.load_image(immu_path)
                print("Imu: {} {}".format(immu_path, img_GT.shape))
                self.immunize = self.img_random_crop(img_GT, 608, 608).cuda().unsqueeze(0)
                self.immunize = self.clamp_with_grad(self.immunize)
                p, q, r = image_list_tamper[idx]
                attack_path = os.path.join(p, q, r)
                img_GT = self.load_image(attack_path)
                print("Atk: {} {}".format(attack_path, img_GT.shape))
                self.attacked_image = self.img_random_crop(img_GT, 608, 608).cuda().unsqueeze(0)
                self.attacked_image = self.clamp_with_grad(self.attacked_image)
                p, q, r = image_list_tampersource[idx]
                another_path = os.path.join(p, q, r)
                img_GT = self.load_image(another_path)
                print("Another: {} {}".format(another_path, img_GT.shape))
                self.another_image = self.img_random_crop(img_GT, 608, 608).cuda().unsqueeze(0)
                self.another_image = self.clamp_with_grad(self.another_image)
                p, q, r = image_list_mask[idx]
                mask_path = os.path.join(p, q, r)
                img_GT = self.load_image(mask_path, grayscale=True)
                print("Mask: {} {}".format(mask_path, img_GT.shape))
                self.mask = self.img_random_crop(img_GT, 608, 608, grayscale=True).cuda().unsqueeze(0)
                self.mask = self.clamp_with_grad(self.mask)
                self.mask = self.mask.repeat(1,3,1,1)

                ### skip images with too large tamper masks
                masked_rate = torch.mean(self.mask)
                redo_gen_mask = masked_rate>0.5
                    # print("Masked rate exceed maximum: {}".format(masked_rate))
                    # continue

                catogory = min(4,int(masked_rate*20))
                valid_images[catogory] += 1
                is_copy_move = False
                if True: #self.immunize is None:
                    ##### re-generates immunized images ########
                    modified_input = self.real_H
                    # print(self.canny_image.shape)
                    forward_stuff = self.netG(x=torch.cat((modified_input, self.canny_image), dim=1))
                    self.immunize, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                    self.immunize = self.clamp_with_grad(self.immunize)
                    self.immunize = self.Quantization(self.immunize)
                    forward_null = self.clamp_with_grad(forward_null)

                ####### Tamper ###############
                if True: #self.attacked_image is None:

                    self.attacked_image = self.immunize * (1-self.mask) + self.another_image * self.mask
                    self.attacked_image = self.clamp_with_grad(self.attacked_image)

                index = np.random.randint(0,5)
                self.attacked_image = self.real_world_attacking_on_ndarray(self.attacked_image[0],
                                                                           qf_after_blur=100 if index<3 else 70,
                                                                           index=index)
                self.reverse_GT = self.real_world_attacking_on_ndarray(self.real_H[0],
                                                                           qf_after_blur=100 if index < 3 else 70,
                                                                           index=index)

                # self.attacked_image = self.clamp_with_grad(self.attacked_image)
                # self.attacked_image = self.Quantization(self.attacked_image)

                self.diffused_image = self.attacked_image.clone().detach()
                self.predicted_mask = torch.sigmoid(self.localizer(self.diffused_image))


                self.predicted_mask = torch.where(self.predicted_mask > 0.5, 1.0, 0.0)
                self.predicted_mask = self.Erode_Dilate(self.predicted_mask)

                F1, TP = self.F1score(self.predicted_mask, self.mask, thresh=0.5)
                F1_sum[catogory] += F1

                self.predicted_mask = self.predicted_mask.repeat(1, 3, 1, 1)

                self.rectified_image = self.attacked_image * (1 - self.predicted_mask)
                self.rectified_image = self.clamp_with_grad(self.rectified_image)


                canny_input = (torch.zeros_like(self.canny_image).cuda())

                reversed_stuff, reverse_feature = self.netG(
                    torch.cat((self.rectified_image, canny_input), dim=1), rev=True)
                reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
                reversed_ch1 = self.clamp_with_grad(reversed_ch1)
                reversed_ch2 = self.clamp_with_grad(reversed_ch2)
                self.reversed_image = reversed_ch1
                self.reversed_canny = reversed_ch2

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

                # ####### Save independent images #############
                save_images = True
                if save_images:
                    eval_kind = self.opt['eval_kind'] #'copy-move/results/RESIZE'
                    eval_attack = self.opt['eval_attack']
                    main_folder = os.path.join(self.out_space_storage,'results', self.opt['dataset_name'], eval_kind)
                    sub_folder = os.path.join(main_folder,eval_attack)
                    if not os.path.exists(main_folder): os.mkdir(main_folder)
                    if not os.path.exists(sub_folder): os .mkdir(sub_folder)
                    if not os.path.exists(sub_folder+ '/recovered_image'): os.mkdir(sub_folder+ '/recovered_image')
                    if not os.path.exists(sub_folder + '/predicted_masks'): os.mkdir(sub_folder + '/predicted_masks')

                    name = sub_folder + '/recovered_image/' + r
                    for image_no in range(self.reversed_image.shape[0]):
                        camera_ready = self.reversed_image[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = sub_folder + '/predicted_masks/' + r

                    for image_no in range(self.predicted_mask.shape[0]):
                        camera_ready = self.predicted_mask[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

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

    def load_image(self, path, readimg=False, Height=608, Width=608,grayscale=False):
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
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
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

    def reload(self,pretrain, network_list=['netG','localizer']):
        if 'netG' in network_list:
            load_path_G = pretrain + "_netG.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.netG, strict=True)
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
        self.save_training_state(epoch=0, iter_step=iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/',
                                 network_list=network_list)

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