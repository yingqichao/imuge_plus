import logging
from collections import OrderedDict
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as Functional
from noise_layers.salt_pepper_noise import SaltPepper
import torchvision.transforms.functional_pil as F_pil
import torchvision.transforms.functional_tensor as F_t
import torchvision
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from skimage.color import rgb2gray
from skimage.metrics._structural_similarity import structural_similarity
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, CWLoss
from models.modules.Quantization import Quantization
import torch.distributed as dist
from utils.JPEG import DiffJPEG
from torchvision import models
from loss import AdversarialLoss, PerceptualLoss, StyleLoss
import cv2
from mbrs_models.Encoder_MP import Encoder_MP
from metrics import PSNR, EdgeAccuracy
from .invertible_net import Inveritible_Decolorization, ResBlock, DenseBlock
from .crop_localize_net import CropLocalizeNet
from .conditional_jpeg_generator import FBCNN, QF_predictor, MantraNet
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
from models.networks import EdgeGenerator, InpaintGenerator, Discriminator, NormalGenerator, UNetDiscriminator, JPEGGenerator
from mbrs_models.Decoder import Decoder, Decoder_MLP
import matlab.engine
from mbrs_models.baluja_networks import HidingNetwork, RevealNetwork
from pycocotools.coco import COCO

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

"""
    ImugeV2 includes: netG localizer
    
    zxy     includes: generator attack_net localizer 


"""

class IRNModel(BaseModel):
    def __init__(self, opt):
        super(IRNModel, self).__init__(opt)
        lr_D = 2e-5 #2*train_opt['lr_G']
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
        self.identity=Identity().cuda()
        self.combined_jpeg_weak = Combined([JpegMask(80),Jpeg(80),JpegMask(90),Jpeg(90),JpegMask(70),Jpeg(70),JpegMask(60),Jpeg(60),JpegMask(50),Jpeg(50)]).cuda()
        self.combined_jpeg_strong = Combined([JpegMask(80),Jpeg(80),JpegMask(90),Jpeg(90),JpegMask(70),Jpeg(70),JpegMask(60),Jpeg(60),JpegMask(50),Jpeg(50)]).cuda()
        self.combined_diffjpeg = Combined([DiffJPEG(90),DiffJPEG(80),DiffJPEG(60),DiffJPEG(70)]).cuda()

        self.bce_loss = nn.BCELoss().cuda()
        self.l1_loss = nn.SmoothL1Loss().cuda() # reduction="sum"
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
        self.SRMConv2D = nn.Conv2d(3, 9, 5, 1, padding=0, bias=False)
        self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
        self.SRMConv2D = self.SRMConv2D.cuda()
        ##SRM filters (fixed)

        for param in self.SRMConv2D.parameters():
            param.requires_grad = False

        self.BayarConv2D = nn.Conv2d(3, 3, 5, 1, padding=0, bias=False).cuda()
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        self.bayar_mask[2, 2] = 0

        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        self.bayar_final[2, 2] = -1

      
        self.localizer = UNetDiscriminator(in_channels=3, residual_blocks=2, out_channels=1,use_spectral_norm=True).cuda() #
        self.CropPred_net = QF_predictor(in_nc=3, classes=4,
                                             crop_pred=True).cuda()  
        self.CropPred_net = DistributedDataParallel(self.CropPred_net,
                                                    device_ids=[torch.cuda.current_device()])
        self.discriminator = Discriminator(in_channels=3, use_sigmoid=True).cuda()
        self.discriminator = DistributedDataParallel(self.discriminator, device_ids=[torch.cuda.current_device()])
        self.localizer = DistributedDataParallel(self.localizer, device_ids=[torch.cuda.current_device()])
        self.netG = Inveritible_Decolorization(dims_in=[[3, 64, 64]], block_num=[4,4,4],subnet_constructor=ResBlock).cuda()
        self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        self.generator = Discriminator(in_channels=3, use_sigmoid=True).cuda()
        self.generator = DistributedDataParallel(self.generator, device_ids=[torch.cuda.current_device()])

        
        self.discriminator_mask = Discriminator(in_channels=15, use_sigmoid=True).cuda()
        self.discriminator_mask = DistributedDataParallel(self.discriminator_mask,
                                                          device_ids=[torch.cuda.current_device()])


        ########### For Crop localization ############
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        # backbone_patch = 'resnet18'
        # self.num_patches = 4 + 1 # train itself 4 and the generator 1
        # self._grid_size_y = 4.0
        # self._grid_size_x = 4.0
        # self.patch_dim = int(self.width_height / self._grid_size_y)
        # patch_embedding_size = 64
        # self.location_classes = int(self._grid_size_x * self._grid_size_y)
        #
        # self.location_losses = []
        # for _ in range(self.location_classes):
        #     self.location_losses.append(torch.nn.CrossEntropyLoss())

        optim_params = []
        for k, v in self.CropPred_net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_CropPred = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                   weight_decay=wd_G,
                                                   betas=(train_opt['beta1'], train_opt['beta2']))

        ########## Attack Net ##########

        # self.attack_net = FBCNN().cuda() #UNetDiscriminator(use_SRM=False, with_attn=True, additional_conv=False).cuda() #rec_FBCNN().cuda() #rec_FBCNN().cuda()
        # self.attack_net = DistributedDataParallel(self.attack_net,
        #                                             device_ids=[torch.cuda.current_device()])
        # self.dis_adv_cov = UNetDiscriminator(use_SRM=False, with_attn=False, additional_conv=False).cuda()  # rec_FBCNN().cuda() #rec_FBCNN().cuda()
        # self.dis_adv_cov = DistributedDataParallel(self.dis_adv_cov,
        #                                           device_ids=[torch.cuda.current_device()])
        # self.dis_adv_fw = QF_predictor(in_nc=3, classes=6).cuda() #MantraNet().cuda()
        # self.dis_adv_fw = DistributedDataParallel(self.dis_adv_fw,
        #                                           device_ids=[torch.cuda.current_device()])
        # 
        # # self.generator_additional = RevealNetwork().cuda()
        # self.generator_additional = UNetDiscriminator(use_SRM=False, with_attn=False,
        #                                               additional_conv=False).cuda()
        # self.generator_additional = DistributedDataParallel(self.generator_additional,
        #                                                     device_ids=[torch.cuda.current_device()])
        # optim_params = []
        # for k, v in self.generator_additional.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         if self.rank <= 0:
        #             logger.warning('Params [{:s}] will not optimize.'.format(k))
        # self.optimizer_gen_add = torch.optim.Adam(optim_params, lr=lr_later,  # train_opt['lr_G'],
        #                                            weight_decay=wd_G,
        #                                            betas=(train_opt['beta1'], train_opt['beta2']))
        # self.optimizers.append(self.optimizer_gen_add)
        # 
        # # for attacker-Student
        # optim_params = []
        # for k, v in self.attack_net.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         if self.rank <= 0:
        #             logger.warning('Params [{:s}] will not optimize.'.format(k))
        # self.optimizer_attacker = torch.optim.Adam(optim_params, lr=lr_D, #train_opt['lr_G'],
        #                                            weight_decay=wd_G,
        #                                            betas=(train_opt['beta1'], train_opt['beta2']))
        # self.optimizers.append(self.optimizer_attacker)
        # 
        # # Teacher
        # optim_params = []
        # for k, v in self.dis_adv_cov.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         if self.rank <= 0:
        #             logger.warning('Params [{:s}] will not optimize.'.format(k))
        # self.optimizer_dis_adv_cov = torch.optim.Adam(optim_params, lr=lr_D,  # train_opt['lr_G'],
        #                                              weight_decay=wd_G,
        #                                              betas=(train_opt['beta1'], train_opt['beta2']))

        # QF predictor 
        optim_params = []
        for k, v in self.dis_adv_fw.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_dis_adv_fw = torch.optim.Adam(optim_params, lr=lr_D, #train_opt['lr_G'],
                                                     weight_decay=wd_G,
                                                     betas=(train_opt['beta1'], train_opt['beta2']))
        if self.task_name == self.TASK_TEST:
            # reload
            self.attack_net = RevealNetwork().cuda()  # rec_FBCNN().cuda() #rec_FBCNN().cuda()
            self.attack_net = DistributedDataParallel(self.attack_net,
                                                      device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
            # for attacker-Student
            optim_params = []
            for k, v in self.attack_net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_attacker = torch.optim.Adam(optim_params, lr=lr_D,  # train_opt['lr_G'],
                                                       weight_decay=wd_G,
                                                       betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_attacker)


        ############ End #############

        # seed is fixed.
            # self.tensor_GT = torch.rand((1,self.width_height,self.width_height),dtype=torch.float32,requires_grad=True).cuda()
        self.tensor_GT = None

        # else:
        #     self.classification_net = models.googlenet(pretrained=True).cuda()
        #     self.classification_net_1 = models.googlenet(pretrained=True).cuda()


        ########## Load pre-trained ##################
        self.load()
        self.log_dict = OrderedDict()
        self.networks_train()

        ########## optimizers ##################
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

        if self.task_name == self.TASK_IMUGEV2 or self.task_name == self.TASK_TEST:
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

            # for mask discriminator
            optim_params = []
            for k, v in self.discriminator_mask.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_discriminator_mask = torch.optim.Adam(optim_params, lr=lr_later,
                                                                 weight_decay=wd_G,
                                                                 betas=(train_opt['beta1'], train_opt['beta2']))

            # for generator
            optim_params = []
            for k, v in self.generator.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_generator = torch.optim.Adam(optim_params, lr=lr_later,
                                                        weight_decay=wd_G,
                                                        betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_generator)

            # for discriminator
            optim_params = []
            for k, v in self.discriminator.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_discriminator = torch.optim.Adam(optim_params, lr=lr_later,)
                                                            # weight_decay=wd_G,
                                                            # betas=(train_opt['beta1'], train_opt['beta2']))
            # localizer
            optim_params = []
            for k, v in self.localizer.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_localizer = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],)
                                                        # weight_decay=wd_G,
                                                        # betas=(train_opt['beta1'], train_opt['beta2']))

        elif self.task_name == self.TASK_RHI3:
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

            optim_params = []
            for k, v in self.discriminator_mask.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_discriminator_mask = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                                 weight_decay=wd_G,
                                                                 betas=(train_opt['beta1'], train_opt['beta2']))


        elif self.task_name == self.TASK_CropLocalize:
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

            optim_params = []
            for k, v in self.localizer.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_localizer = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                        weight_decay=wd_G,
                                                        betas=(train_opt['beta1'], train_opt['beta2']))

            # for attacker
            optim_params = []
            for k, v in self.attack_net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_attacker = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                        weight_decay=wd_G,
                                                        betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_attacker)

            # for generator
            optim_params = []
            for k, v in self.generator.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_generator = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                        weight_decay=wd_G,
                                                        betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_generator)

            # for discriminator
            optim_params = []
            for k, v in self.discriminator.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_discriminator = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                        weight_decay=wd_G,
                                                        betas=(train_opt['beta1'], train_opt['beta2']))

            optim_params = []
            for k, v in self.discriminator_mask.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_discriminator_mask = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                            weight_decay=wd_G,
                                                            betas=(train_opt['beta1'], train_opt['beta2']))

            optim_params = []
            for k, v in self.dis_adv_cov.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_dis_adv_cov = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                                 weight_decay=wd_G,
                                                                 betas=(train_opt['beta1'], train_opt['beta2']))

            optim_params = []
            for k, v in self.dis_adv_fw.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_dis_adv_fw = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                                 weight_decay=wd_G,
                                                                 betas=(train_opt['beta1'], train_opt['beta2']))

            optim_params = []
            for k, v in self.CropPred_net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_CropPred = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                                 weight_decay=wd_G,
                                                                 betas=(train_opt['beta1'], train_opt['beta2']))

        ############## schedulers #########################
        # if train_opt['lr_scheme'] == 'MultiStepLR':
        #     for optimizer in self.optimizers:
        #         self.schedulers.append(
        #             lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
        #                                              restarts=train_opt['restarts'],
        #                                              weights=train_opt['restart_weights'],
        #                                              gamma=train_opt['lr_gamma'],
        #                                              clear_state=train_opt['clear_state']))
        # elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
        #     for optimizer in self.optimizers:
        #         self.schedulers.append(
        #             lr_scheduler.CosineAnnealingLR_Restart(
        #                 optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
        #                 restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        # else:
        #     raise NotImplementedError('MultiStepLR learning rate scheme is enough.')


    def feed_data(self, batch):
        # if self.train_opt['using_self_defined_dataset'] == 1.0:
            # {'GT': img_GT, 'GT_path': GT_path}
        data, label, canny = batch
        self.real_H = data.cuda()
        # self.jpeg_real_H = data['JPEG'].cuda()
        # self.QF_GT = data['QF'].type(torch.FloatTensor).cuda().unsqueeze(1)
        # self.real_H_path = data['GT_path']
        # self.label_GT = label.type(torch.FloatTensor).cuda().unsqueeze(1) # regression
        self.canny_image = canny.cuda()
        # else:
        #     img, label = batch
        #     self.real_H = img.cuda()
        #     self.label_GT = label.cuda()

        # self.ref_L = data['LQ'].cuda()  # LQ
        # self.real_H = data['GT'].cuda()  # GT

    def gaussian_batch(self, dims):
        return torch.clamp(torch.randn(tuple(dims)).cuda(),0,1)

    def loss_forward(self, label_GT, label_pred, out, y, z):
        is_targeted = False
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = z.reshape([out.shape[0], -1])
        l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z**2) / z.shape[0]
        loss_adv = None
        if label_GT is not None:
            loss_adv = 2 * self.criterion_adv(label_pred, label_GT, is_targeted)

        return l_forw_fit, l_forw_ce, loss_adv

    def loss_backward(self, label_pred, label_GT, GT_ref, reversed_image):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(GT_ref, reversed_image)

        # loss_label = self.criterion(label_pred, label_GT)
        loss_label = None
        return l_back_rec, loss_label

    def loss_forward_and_backward_imuge(self, fake_outputs, cover_images, masks=None, use_l1=True, use_vgg=False, use_percept=False):
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
                                                cover_images * masks)  / torch.mean(masks)
            else:
                gen_l1_local_loss = self.l2_loss(fake_outputs * masks,
                                                         cover_images * masks) / torch.mean(masks)
            # gen_loss += 2 * gen_l1_local_loss

        if use_percept and cover_images.shape[1] == 3:
            # generator perceptual loss

            gen_content_loss = self.perceptual_loss(fake_outputs, cover_images)
            gen_loss += 0.1* gen_content_loss

        if use_vgg and cover_images.shape[1] == 3:
            l_forward_ssim = - self.ssim_loss(fake_outputs, cover_images)
            gen_loss += 0.1*l_forward_ssim

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

        self.global_step = self.global_step+1
        logs, debug_logs = [], []
        #### init
        l_train_denoise = 0
        l_forward_ab, l_back_ab, l_crop, l_bce, l_bce_ab, l_forward, gen_gan_loss, gen_mask_gan_loss, dis_mask_loss, l_apex_loss = \
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
        psnr_forward, psnr_pattern, psnr_back, psnr_ablation, psnr_forward_ab, psnr_back_ab = 10, 10, 10, 10, 10, 10
        l_back_image, l_back_zero = 0.2, 0.2
        gen_loss, gen_loss_forward, dis_fake = 0.2, 0.2, 1.0
        attack_layer = "Identity"

        if self.real_H is not None:
            self.real_H = torch.clamp(self.real_H, 0, 1)
            batch_size = self.real_H.shape[0]

            masks, tamper_rate = self.generate_stroke_mask(
                [self.real_H.shape[2], self.real_H.shape[3]])
            masks = masks.cuda()
            masks_GT = masks.expand(batch_size, 1, -1, -1)
            masks = masks.expand(batch_size, 3, -1, -1)
        save_interval = 10000


        if self.task_name == self.TASK_IMUGEV2 or self.task_name == self.TASK_TEST:
            ######  Training of IMUGE  #######
            # modified_input, apex, scaled_cropped_attacked, reversed_image, rectified_crop_padded_image, forward_image, cropped_padded_image = \
            #                                                                                                             None,None,None,None,None,None,None
            # predicted_crop_region, rectified_groundtruth, rectified_plane = None,None,None
            train_main_stream, train_jpeg, new_task = True, True, True
            with torch.enable_grad():
                if eval_dir is not None:

                    ############## Evaluate ######################
                    eval_data, water_data = eval_dir['eval_data'], eval_dir['water_data']
                    val_path, water_path, save_path = eval_dir['val_path'], eval_dir['water_path'], eval_dir[
                        'save_path']
                    tamper_data, mask_data = eval_dir['tamper_data'], eval_dir['mask_data']
                    source_tamper_path, predicted_mask_tamper_path, gt_mask_tamper_path = eval_dir[
                                                                                              'source_tamper_path'], \
                                                                                          eval_dir[
                                                                                              'predicted_mask_tamper_path'], \
                                                                                          eval_dir[
                                                                                              'gt_mask_tamper_path']
                    val_path = os.path.join(val_path, eval_data)
                    water_path = os.path.join(water_path, water_data)
                    source_tamper_path = os.path.join(source_tamper_path, tamper_data)
                    gt_mask_tamper_path = os.path.join(gt_mask_tamper_path, mask_data)

                    # print("Tamper: {}  {}".format(tamper_data, mask_data))

                    tensor_c = self.load_image(path=val_path, grayscale=False)
                    watermark_c = self.load_image(path=water_path, grayscale=True)
                    source_tamper = self.load_image(path=source_tamper_path, grayscale=False)
                    mask_tamper = self.load_image(path=gt_mask_tamper_path, grayscale=True)

                    forward_image = self.netG(x=tensor_c)

                    forward_image = self.Quantization(forward_image)
                    forward_image = torch.clamp(forward_image, 0, 1)
                    layer = self.jpeg90 if dist.get_rank() == 0 else self.jpeg70
                    compressed_image = layer(forward_image)
                    compressed_image = torch.clamp(compressed_image, 0, 1)
                    tamper = compressed_image * (1 - mask_tamper) + source_tamper * mask_tamper
                    tamper = torch.clamp(tamper, 0, 1)
                    predicted_masks = self.localizer(tamper)
                    rectify = tamper * (1 - mask_tamper)

                    reversed = self.netG(x=rectify, rev=True)
                    reversed = torch.clamp(reversed, 0, 1)

                    name = os.path.join(
                        save_path,
                        os.path.splitext(eval_data)[0] + "_" + str(dist.get_rank()) + ("_realworld.png"))
                    self.print_individual_image(reversed, name)

                    name = os.path.join(
                        save_path,
                        os.path.splitext(eval_data)[0] + str(dist.get_rank()) + "_diff.png")
                    self.print_individual_image(10 * torch.abs(reversed - tensor_c), name)

                    name = os.path.join(
                        save_path,
                        os.path.splitext(eval_data)[0] + str(dist.get_rank()) + ("_pred.png"))
                    self.print_individual_image(predicted_masks, name)

                    print("Saved: {}".format(name))

                else:
                    ###
                    train_main_stream, train_jpeg, new_task = False, False, False
                ######### Train #########################
                # is_real_train = True
                new_task = self.new_task>0
                # train_jpeg = True
                train_main_stream = False
                ######### Train JPEG #########################
                if train_jpeg:
                    self.BayarConv2D.weight.data *= self.bayar_mask
                    self.BayarConv2D.weight.data *= torch.pow(
                        self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1),-1)
                    self.BayarConv2D.weight.data += self.bayar_final

                    realjpeg_bayar = self.BayarConv2D(self.symm_pad(self.jpeg_real_H, (2, 2, 2, 2)))
                    realjpeg_srm = self.SRMConv2D(self.symm_pad(self.jpeg_real_H, (2, 2, 2, 2)))
                    # print("diff shape {}".format((self.jpeg_real_H-self.real_H).shape))
                    # print("bayar shape {}".format((realjpeg_bayar).shape))
                    # print("srm shape {}".format((realjpeg_srm).shape))
                    realjpeg_higher_band = torch.cat([self.jpeg_real_H-self.real_H, realjpeg_bayar, realjpeg_srm], axis=1)
                    ##### reconstruct JPEG #######

                    QF_r2 = self.dis_adv_fw(self.real_H)
                    # QF_r2 = torch.clamp(QF_r2, 0, 1)
                    plaintext_label = 5
                    l_qf_r_2 = self.criterion(QF_r2, torch.tensor(plaintext_label).long().cuda().expand_as(self.label_GT))

                    QF_r3 = self.dis_adv_fw(self.jpeg_real_H)
                    # QF_simul_JPEG = torch.clamp(QF_simul_JPEG, 0, 1)
                    # print("QF_R size:{}".format(QF_r))
                    l_qf_r_3 = self.criterion(QF_r3, self.label_GT)
                    l_qf_r = (l_qf_r_3 + l_qf_r_2) / 2
                    l_qf_r.backward()
                    # gradient clipping
                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.dis_adv_fw.parameters(), self.train_opt['gradient_clipping'])
                        # nn.utils.clip_grad_norm_(self.attack_net.parameters(), self.train_opt['gradient_clipping'])
                    self.optimizer_dis_adv_fw.step()
                    # self.optimizer_attacker.step()
                    self.optimizer_dis_adv_fw.zero_grad()
                    ############## train jpeg #################################################

                    generated_jpeg, gen_feat_qf = self.attack_net(
                        torch.clamp(self.real_H + 0.0 * self.gaussian(self.real_H), 0, 1), self.QF_GT)
                    generated_jpeg = torch.clamp(generated_jpeg, 0, 1)

                    generated_bayar = self.BayarConv2D(self.symm_pad(generated_jpeg, (2, 2, 2, 2)))
                    generated_srm = self.SRMConv2D(self.symm_pad(generated_jpeg, (2, 2, 2, 2)))
                    generated_higher_band = torch.cat([generated_jpeg-self.real_H,generated_bayar, generated_srm], axis=1)

                    # l_jpegloss = self.l1_loss(generated_higher_band, realjpeg_higher_band.clone().detach())
                    l_jpegloss = self.l1_loss(generated_jpeg,self.jpeg_real_H)

                    QF_simul_JPEG = self.dis_adv_fw(generated_jpeg)
                    l_QF_simul_JPEG = self.criterion(QF_simul_JPEG, self.label_GT)
                    l_QF_simul_feat = self.criterion(gen_feat_qf, self.label_GT)

                    # l_jpeg_feat_loss = 0
                    # for feat_level in range(len(raw_feats)):
                    #     l_jpeg_feat_loss += self.l1_loss(raw_feats[feat_level], jpg_feats[feat_level].detach())
                    # l_jpeg_feat_loss /= len(raw_feats)

                    l_train_jpeg = l_jpegloss + 0.001 * l_QF_simul_JPEG + 0.01*l_QF_simul_feat#+ 0.1 * jpeg_adv_loss #+ 0.01 * l_QF_simul_JPEG + 0.1*l_jpeg_feat_loss
                    l_train_jpeg.backward()
                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.attack_net.parameters(), self.train_opt['gradient_clipping'])
                    self.optimizer_attacker.step()
                    self.networks_zerograd()

                    psnr_JPEG = self.psnr(self.postprocess(generated_jpeg), self.postprocess(self.jpeg_real_H))
                    # psnr_JPEG1 = self.psnr(self.postprocess(reconstructed_jpeg), self.postprocess(self.jpeg_real_H))

                    logs.append(('psnJPG', psnr_JPEG.item()))
                    # logs.append(('psnJPGRecon', psnr_JPEG1.item()))
                    logs.append(('lQFclass', l_qf_r.item()))
                    # logs.append(('lQFrecon', l_QF_recon_JPEG.item()))
                    # logs.append(('L1reconJPEG', l_jpegloss_recon.item()))
                    # logs.append(('Adv', jpeg_adv_loss.item()))
                    logs.append(('lQFsimul', l_QF_simul_JPEG.item()))
                    logs.append(('lQFsimulFeat', l_QF_simul_feat.item()))
                    # logs.append(('lJPEGfeat', l_jpeg_feat_loss.item()))
                    logs.append(('L1simulJPEG', l_jpegloss.item()))
                    if step % 500 == 499:
                        images = stitch_images(
                            self.postprocess(self.real_H),
                            self.postprocess(self.jpeg_real_H),
                            self.postprocess(realjpeg_bayar),
                            self.postprocess(10 * torch.abs(self.real_H - self.jpeg_real_H)),

                            self.postprocess(generated_jpeg),
                            self.postprocess(generated_bayar),
                            self.postprocess(10 * torch.abs(generated_jpeg - self.real_H)),
                            self.postprocess(10 * torch.abs(generated_jpeg - self.jpeg_real_H)),
                            img_per_row=1
                        )

                        name = os.path.join('./images/out_muging/AttackSimulation/', str(step).zfill(5) + "_" + str(
                            dist.get_rank()) + ".png")
                        print('\nsaving sample ' + name)
                        images.save(name)
                    #
                    # QF1_item = "{0:.2f} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5:.2f}" \
                    #     .format(QF_simul_JPEG[0, 0].item(), QF_simul_JPEG[0, 1].item(), QF_simul_JPEG[0, 2].item(), QF_simul_JPEG[0, 3].item(),
                    #             QF_simul_JPEG[0, 4].item(), QF_simul_JPEG[0, 5].item())
                    # QF1GT_item = (self.label_GT[0].item())


                if train_main_stream and new_task and self.previous_images is not None and self.previous_previous_images is not None:
                ####### CVPR ##################

                    if np.random.rand() < 0.15:
                        modified_input = (self.real_H * (1 - masks) + self.previous_images * masks).clone().detach()
                        modified_input = torch.clamp(modified_input, 0, 1)
                    else:
                        modified_input = self.real_H
                    forward_image = self.netG(x=modified_input)

                    forward_image = self.Quantization(forward_image)
                    forward_image = torch.clamp(forward_image, 0, 1)

                    attack_full_name = ""

                    # mix-up jpeg layer, remeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeember to clamp after each attack! Or gradient explosion will occur!!!
                    # if attack_num < 2 and np.random.rand() < 0.33:
                    #     attack_num += 1
                    #     cropped_forward = self.resize(cropped_forward)
                    #     cropped_forward = torch.clamp(cropped_forward, 0, 1)
                    #     attack_full_name += self.resize.name
                    # if attack_num < 2 and np.random.rand() < 0.33:
                    #     attack_num += 1
                    #     cropped_forward = self.gaussian_blur(cropped_forward)
                    #     cropped_forward = torch.clamp(cropped_forward, 0, 1)
                    #     attack_full_name += self.gaussian_blur.name


                    ###### Begin Crop function #######################
                    # scaled_cropped_attacked, cropped_padded_image, masks, apex, just_crop_image = self.crop.cropped_out(
                    #     cropped_forward, min_rate=0.6, max_rate=0.9)
                    min_rate , max_rate = 0.6, 0.9
                    masks = torch.ones_like(forward_image)
                    if min_rate:
                        self.height_ratio = min_rate + (max_rate - min_rate) * np.random.rand()
                        self.width_ratio = min_rate + (max_rate - min_rate) * np.random.rand()
                    else:
                        self.height_ratio = 0.3 + 0.7 * np.random.rand()
                        self.width_ratio = 0.3 + 0.7 * np.random.rand()
                    self.height_ratio = min(self.height_ratio, self.width_ratio + 0.3)
                    self.width_ratio = min(self.width_ratio, self.height_ratio + 0.3)
                    # image, cover_image = image_and_cover

                    h_start, h_end, w_start, w_end = self.crop.get_random_rectangle_inside(forward_image.shape, self.height_ratio,self.width_ratio)

                    cropped_fragmented_images = forward_image[:, :, h_start: h_end, w_start: w_end]
                    # zero_images[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
                    masks[:, :, h_start: h_end, w_start: w_end] = 0
                    cropped_padded_image = forward_image * (1 - masks)
                    scaled_cropped_images = Functional.interpolate(
                        cropped_fragmented_images,
                        size=[forward_image.shape[2], forward_image.shape[3]],
                        mode='bicubic')
                    scaled_cropped_images = torch.clamp(scaled_cropped_images, 0, 1)

                    ####### JPEG Simulator ###################
                    if self.global_step % 10 < 10:  # if np.random.rand() < 0.0:
                        attack_layer = self.combined_jpeg_weak
                        attack_layer_1 = self.combined_jpeg_strong
                        cropped_forward_0 = attack_layer(scaled_cropped_images)
                        cropped_forward_0 = torch.clamp(cropped_forward_0, 0, 1)
                        cropped_forward_1 = attack_layer_1(scaled_cropped_images)
                        cropped_forward_1 = torch.clamp(cropped_forward_1, 0, 1)
                        beta = np.random.rand()
                        jpeg_image = beta * cropped_forward_0 + (1 - beta) * cropped_forward_1
                        attack_full_name += attack_layer.name
                    elif self.global_step % 10 < 10:
                        jpeg_image, _ = self.dis_adv_cov(scaled_cropped_images, self.QF_GT)
                        attack_full_name += "JPEGGenerator"
                    else:  # self.global_step % 2 == 1:
                        jpeg_image, _ = self.attack_net(scaled_cropped_images, self.QF_GT)
                        attack_full_name += "JPEGSimulator"
                    jpeg_image = torch.clamp(jpeg_image, 0, 1)
                    ASL_jpeg_diff = (jpeg_image - scaled_cropped_images).clone().detach()

                    ######### test whether we can train the network with the following attacks ############
                    ### The answer is no. 2021.10.13 #########
                    denoise_input = scaled_cropped_images  # .clone().detach()
                    gaussian_image = self.gaussian(denoise_input)
                    gaussian_image = torch.clamp(gaussian_image, 0, 1)
                    ASL_gaussian_diff = (gaussian_image - scaled_cropped_images).clone().detach()
                    salt_image = self.salt_pepper(denoise_input)
                    salt_image = torch.clamp(salt_image, 0, 1)
                    ASL_salt_diff = (salt_image - scaled_cropped_images).clone().detach()

                    blur_image = self.gaussian_blur(denoise_input)
                    blur_image = torch.clamp(blur_image, 0, 1)
                    ASL_blur_diff = (blur_image - scaled_cropped_images).clone().detach()
                    median_image = self.median_blur(denoise_input)
                    median_image = torch.clamp(median_image, 0, 1)
                    ASL_median_diff = (median_image - scaled_cropped_images).clone().detach()

                    # brightness_factor,contrast_factor, saturation_factor, hue_factor = 1.5,1.5,1.5,0.25
                    # bright_image = F_t.adjust_brightness(forward_image, brightness_factor)
                    # ASL_brightness_diff = (bright_image - forward_image).clone().detach()
                    # contrast_image = F_t.adjust_contrast(forward_image, contrast_factor)
                    # ASL_contrast_diff = (contrast_image - forward_image).clone().detach()
                    # satuarate_image = F_t.adjust_saturation(forward_image, saturation_factor)
                    # ASL_satuarate_diff = (satuarate_image - forward_image).clone().detach()
                    # hue_image = F_t.adjust_hue(forward_image, hue_factor)
                    # ASL_hue_diff = (hue_image - forward_image).clone().detach()

                    ########## ASL simulation ###########
                    if self.global_step % 9 < 2:
                        # cropped_forward = blur_image
                        scaled_cropped_attacked = ASL_blur_diff + scaled_cropped_images
                        cropped_blur_forward = scaled_cropped_attacked
                        attack_full_name = "blur"
                    elif self.global_step % 9 < 4:
                        # scaled_cropped_attacked = median_image
                        scaled_cropped_attacked = ASL_median_diff + scaled_cropped_images
                        cropped_median_forward = scaled_cropped_attacked
                        attack_full_name = "median"
                    elif self.global_step % 9 < 5:
                        scaled_cropped_attacked = scaled_cropped_images
                        attack_full_name = "identity"
                        # if np.random.rand()<0.5:
                        #     scaled_cropped_attacked = ASL_salt_diff+forward_image
                        #     cropped_salt_forward = scaled_cropped_attacked
                        #     attack_full_name = "salt"
                        # else:
                        #     scaled_cropped_attacked = ASL_gaussian_diff + forward_image
                        #     cropped_gaussian_forward = scaled_cropped_attacked
                        #     attack_full_name = "gaussian"
                    else:  # if self.global_step % 9 < 7:
                        # scaled_cropped_attacked = jpeg_image
                        scaled_cropped_attacked = ASL_jpeg_diff + scaled_cropped_images
                        cropped_jpeg_forward = scaled_cropped_attacked
                        attack_full_name = "jpeg"

                    ###### dual reshape loss ############
                    scaled_back = Functional.interpolate(
                        scaled_cropped_attacked,
                        size=[h_end - h_start, w_end - w_start],
                        mode='bicubic')
                    scaled_back = torch.clamp(scaled_back, 0, 1)
                    scaled_back_padded = torch.zeros_like(forward_image)
                    scaled_back_padded[:, :, h_start: h_end, w_start: w_end] = scaled_back
                    dual_reshape_diff = (scaled_back_padded - cropped_padded_image).clone().detach()

                    cropped_padded_image = cropped_padded_image + dual_reshape_diff

                    apex = (h_start/forward_image.shape[2], h_end/forward_image.shape[2], w_start/forward_image.shape[3], w_end/forward_image.shape[3])

                    ###### End Crop function #######################

                    ##### unseen ##########
                    # cropped_forward = ASL_jpeg_diff+forward_image
                    # cropped_jpeg_forward = cropped_forward
                    # attack_full_name = "jpeg"
                    ### salt
                    # cropped_forward = ASL_salt_diff+forward_image
                    # cropped_salt_forward = cropped_forward
                    # attack_full_name = "salt"
                    ### Gaussian
                    # cropped_forward = ASL_gaussian_diff+forward_image
                    # cropped_gaussian_forward = cropped_forward
                    # attack_full_name = "gaussian"
                    ### CollorJitter
                    # brightness_factor,contrast_factor, saturation_factor, hue_factor = 1.5,1.5,1.5,0.25
                    # # pil_image = F_t.adjust_brightness(forward_image, brightness_factor)
                    # # pil_image = F_t.adjust_contrast(forward_image, contrast_factor)
                    # # pil_image = F_t.adjust_saturation(forward_image, saturation_factor)
                    # pil_image = F_t.adjust_hue(forward_image, hue_factor)
                    # ASL_brightness = (pil_image - forward_image).clone().detach()
                    # cropped_forward = ASL_brightness + forward_image
                    # cropped_brightness_forward = cropped_forward
                    # attack_full_name = "hue_factor"

                    ########## Denoise ##############

                    # united_input_1 = torch.cat((cropped_salt_forward.clone().detach(), cropped_gaussian_forward.clone().detach()), dim=0)
                    # united_input_2 = torch.cat((cropped_median_forward.clone().detach(), cropped_blur_forward.clone().detach()), dim=0)
                    # # GT_forward, FW_feats = self.generator_additional(forward_image_detach,None)
                    # GT_forward, _, r2_1, l2_1, rjpg2, ljpg2 = self.generator_additional(united_input_1,
                    #                                                                            cropped_forward.clone().detach(),
                    #                                                                            double=0.0)
                    # denoise_forward, DN_feats, r2_1, l2_1, rjpg2, ljpg2 = self.generator_additional(united_input_2,
                    #                                                                                 cropped_forward.clone().detach(),
                    #                                                                                 double=1.0)
                    # #
                    # GT_forward_concat = torch.cat((GT_forward,denoise_forward),dim=0)
                    # GT_forward_concat = torch.clamp(GT_forward_concat, 0, 1)
                    # forward_image_detach = forward_image.clone().detach().expand_as(GT_forward_concat)
                    # _, _, l_recon = self.loss_forward_and_backward_imuge(
                    #     GT_forward_concat, forward_image_detach, masks=None, use_l1=True, use_percept=True, use_vgg=False)
                    #
                    # # jpeg_forward_concat = torch.cat((FW_feats, DN_feats), dim=1)
                    # # jpeg_forward_concat = torch.clamp(jpeg_forward_concat, 0, 1)
                    # # forward_image_detach = forward_image.clone().detach().expand_as(jpeg_forward_concat)
                    # # _, _, l_jpegloss_recon = self.loss_forward_and_backward_imuge(
                    # #     jpeg_forward_concat, forward_image_detach, masks=None, use_l1=True, use_percept=True,
                    # #     use_vgg=False)
                    # #
                    # reversed_image = self.netG(GT_forward_concat, rev=True)
                    # reversed_image = torch.clamp(reversed_image, 0, 1)
                    # _, _, l_jpeg_backward = self.loss_forward_and_backward_imuge(
                    #     reversed_image, modified_input.expand_as(forward_image_detach), masks=masks, use_l1=True, use_percept=False, use_vgg=True)
                    #
                    # # reversed_image_2 = self.netG(jpeg_forward_concat, rev=True)
                    # # reversed_image_2 = torch.clamp(reversed_image_2, 0, 1)
                    # # _, _, l_backward_2 = self.loss_forward_and_backward_imuge(
                    # #     reversed_image_2, modified_input, masks=masks, use_l1=True, use_percept=False, use_vgg=True)
                    #
                    # l_train_jpeg = 0.5*l_jpeg_backward + l_recon #+ l_jpegloss_recon + l_backward_2
                    #
                    # l_train_jpeg.backward()
                    # if self.train_opt['gradient_clipping']:
                    #     nn.utils.clip_grad_norm_(self.generator_additional.parameters(), self.train_opt['gradient_clipping'])
                    # self.optimizer_gen_add.step()
                    # self.networks_zerograd()

                    ######## Crop adjust ###################
                    # scaled_cropped_attacked, cropped_padded_image, masks, apex, just_crop_image = self.crop.cropped_out(
                    #     cropped_forward, min_rate=0.6, max_rate=0.9)
                    crop_rate = torch.mean(masks)
                    scaled_cropped_attacked = self.Quantization(scaled_cropped_attacked)
                    scaled_cropped_attacked = torch.clamp(scaled_cropped_attacked, 0, 1)
                    # just_crop_image = self.Quantization(just_crop_image)
                    # just_crop_image = torch.clamp(just_crop_image, 0, 1)
                    cropped_padded_image = self.Quantization(cropped_padded_image)
                    cropped_padded_image = torch.clamp(cropped_padded_image, 0, 1)

                    # ################################################
                    # # ########distillation-based image pre-processor
                    # ################################################
                    # forward_image_detach = forward_image.clone().detach()
                    # GT_forward, FW_feats = self.generator_additional(forward_image_detach,None)
                    # GT_forward = torch.clamp(GT_forward, 0, 1)
                    # # l_recon = self.l1_loss(GT_forward, forward_image_detach)
                    # _, _, l_recon = self.loss_forward_and_backward_imuge(
                    #     GT_forward, forward_image_detach, masks=None, use_l1=True, use_percept=False, use_vgg=True)
                    #
                    # scaled_cropped_attacked, rectified_crop_padded_image, masks, apex = self.crop.cropped_out(
                    #     GT_forward, min_rate=0.6, max_rate=0.9)
                    # scaled_cropped_attacked = self.Quantization(scaled_cropped_attacked)
                    # scaled_cropped_attacked = torch.clamp(scaled_cropped_attacked, 0, 1)
                    # rectified_crop_padded_image = self.Quantization(rectified_crop_padded_image)
                    # rectified_crop_padded_image = torch.clamp(rectified_crop_padded_image, 0, 1)
                    #
                    # reversed_image_GT = self.netG(rectified_crop_padded_image, rev=True)
                    # reversed_image_GT = torch.clamp(reversed_image_GT, 0, 1)
                    # _, _, l_jpeg_backward = self.loss_forward_and_backward_imuge(
                    #     reversed_image_GT, modified_input, masks=masks, use_l1=True, use_percept=False, use_vgg=True)
                    #
                    # psnr_recover_GT = self.psnr(self.postprocess(modified_input), self.postprocess(reversed_image_GT))
                    # psnr_recon_GT = self.psnr(self.postprocess(GT_forward), self.postprocess(forward_image_detach))
                    #
                    # l_train_jpeg = l_jpeg_backward # l_recon +
                    # l_train_jpeg.backward()
                    # if self.train_opt['gradient_clipping']:
                    #     nn.utils.clip_grad_norm_(self.generator_additional.parameters(), self.train_opt['gradient_clipping'])
                    # self.optimizer_gen_add.step()
                    # self.networks_zerograd()
                    #
                    # denoise_forward = scaled_cropped_attacked #.clone().detach()
                    # denoise_forward, DN_feats = self.generator(scaled_cropped_attacked.clone().detach(), None)
                    # denoise_forward = torch.clamp(denoise_forward, 0, 1)
                    # _, _, l_denoise_recon = self.loss_forward_and_backward_imuge(
                    #     denoise_forward, scaled_cropped_attacked.clone().detach(), masks=None, use_l1=True, use_percept=False, use_vgg=False)
                    # l_train_denoise = l_denoise_recon
                    #
                    # l_denoise_level_loss = 0
                    # for feat_level in range(len(DN_feats)):
                    #     l_denoise_level_loss += self.l1_loss(DN_feats[feat_level], FW_feats[feat_level].detach())
                    # l_denoise_level_loss /= len(DN_feats)
                    #
                    # l_denoise_level_loss.backward(retain_graph=True)
                    # if self.train_opt['gradient_clipping']:
                    #     nn.utils.clip_grad_norm_(self.generator.parameters(),
                    #                              self.train_opt['gradient_clipping'])
                    # self.optimizer_generator.step()
                    # self.networks_zerograd()
                    # ################################################
                    # # update finished, forward by pre-processing
                    # ################################################
                    # scaled_cropped_attacked = self.generator(scaled_cropped_attacked)

                    ############ Crop localization loss ################

                    # Extract patch and label.
                    cropmask, location = self.CropPred_net(scaled_cropped_attacked.clone().detach())
                    location = torch.clamp(location, 0, 1)
                    labels_tensor = torch.zeros_like(location).cuda()
                    labels_tensor[:, 0] = apex[0]  # int(32 * apex[0]) / 32
                    labels_tensor[:, 1] = apex[1]  # int(32 * apex[1]) / 32
                    labels_tensor[:, 2] = apex[2]  # int(32 * apex[2]) / 32
                    labels_tensor[:, 3] = apex[3]  # int(32 * apex[3]) / 32
                    # location = torch.clamp(location, 0, 1)
                    patch_loss_value = self.l1_loss(location, labels_tensor)
                    down_sampled_mask = masks[:,:1,:,:] #Functional.interpolate(masks[:,:1,:,:],size=[32,32],mode='bicubic')
                    mask_loss_value = self.l1_loss(cropmask, down_sampled_mask)

                    crop_loss = patch_loss_value+mask_loss_value  # * 0.01
                    crop_loss.backward()
                    # gradient clipping
                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.CropPred_net.parameters(), self.train_opt['gradient_clipping'])
                    self.optimizer_CropPred.step()
                    self.optimizer_CropPred.zero_grad()

                    cropmask, location = self.CropPred_net(scaled_cropped_attacked)
                    location = torch.clamp(location, 0, 1)
                    patch_loss_value = self.l1_loss(location, labels_tensor)
                    down_sampled_mask = masks[:, :1, :, :]  # Functional.interpolate(masks[:,:1,:,:],size=[32,32],mode='bicubic')
                    mask_loss_value = self.l1_loss(cropmask, down_sampled_mask)

                    crop_loss = 10*patch_loss_value + mask_loss_value  # * 0.01

                    h_start, h_end, w_start, w_end = location[:, 0], location[:, 1], location[:, 2], location[:, 3]
                    predicted_crop_region = torch.ones_like(scaled_cropped_attacked)
                    for idxx in range(batch_size):
                        predicted_crop_region[idxx, :,
                        int(h_start[idxx] * self.width_height): int(h_end[idxx] * self.width_height),
                        int(w_start[idxx] * self.width_height): int(w_end[idxx] * self.width_height)] = 0
                    apex_item = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}" \
                        .format(apex[0], apex[1], apex[2], apex[3])
                    croppred_item = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}" \
                        .format(h_start[0].item(), h_end[0].item(), w_start[0].item(), w_end[0].item())

                    ####### Ideal Crop adjust ##########
                    # rectified_crop_padded_image = torch.zeros_like(self.real_H)
                    # # h_start, h_end, w_start, w_end = apex
                    # for image_idx in range(self.real_H.shape[0]):
                    #     h_start_pre = max(0, int(self.real_H.shape[2] * apex[0]))
                    #     h_end_pre = min(self.real_H.shape[2] - 1, int(self.real_H.shape[2] * apex[1]))
                    #     w_start_pre = max(0, int(self.real_H.shape[3] * apex[2]))
                    #     w_end_pre = min(self.real_H.shape[3] - 1, int(self.real_H.shape[3] * apex[3]))
                    #     # print("{} {} {} {}".format(h_start_pre,h_end_pre,w_start_pre,w_end_pre))
                    #     scaled_cropped_images = Functional.interpolate(
                    #         denoise_forward[image_idx, :, :, :].unsqueeze(0),
                    #         size=[h_end_pre - h_start_pre, w_end_pre - w_start_pre],
                    #         mode='bicubic')
                    #     rectified_crop_padded_image[image_idx, :, h_start_pre:h_end_pre,
                    #     w_start_pre:w_end_pre] = scaled_cropped_images.squeeze(0)
                    ######### Actual Crop adjust ##########
                    slight_augmented_location = location #+(torch.clamp(torch.randn(tuple(location.shape)).cuda(),0,1)*0.1)
                    # rectified_crop_padded_image = torch.zeros_like(self.real_H)
                    # h_start, h_end, w_start, w_end = apex
                    # for image_idx in range(self.real_H.shape[0]):
                    #     h_start_pre = min(self.real_H.shape[2] - 1,
                    #                       int(self.real_H.shape[2] * slight_augmented_location[image_idx, 0]))
                    #     h_end_pre = min(self.real_H.shape[2] - 1,
                    #                     int(self.real_H.shape[2] * slight_augmented_location[image_idx, 1]))
                    #     w_start_pre = min(self.real_H.shape[3] - 1,
                    #                       int(self.real_H.shape[3] * slight_augmented_location[image_idx, 2]))
                    #     w_end_pre = min(self.real_H.shape[3] - 1,
                    #                     int(self.real_H.shape[3] * slight_augmented_location[image_idx, 3]))
                    #     # print("{} {} {} {}".format(h_start_pre,h_end_pre,w_start_pre,w_end_pre))
                    #     scaled_cropped_images = Functional.interpolate(
                    #         just_crop_image[image_idx, :, :, :].unsqueeze(0),
                    #         size=[h_end_pre - h_start_pre, w_end_pre - w_start_pre],
                    #         mode='bicubic').squeeze(0)
                    #     rectified_crop_padded_image[image_idx, :, h_start_pre:h_end_pre,
                    #     w_start_pre:w_end_pre] = scaled_cropped_images
                    # rectified_crop_padded_image.retain_grad()
                    ## rectified_crop_padded_image.requires_grad = True
                    ################ rectify the GT ##################
                    # rectified_crop_padded_image = cropped_padded_image
                    # rectified_mask = masks
                    # rectified_groundtruth = modified_input
                    H_S, H_E, W_S, W_E = slight_augmented_location[:, 0], slight_augmented_location[:, 1], slight_augmented_location[:, 2], slight_augmented_location[:, 3]
                    # H_S, H_E, W_S, W_E = apex[0],apex[1],apex[2],apex[3]
                    h_start_r, h_end_r, w_start_r, w_end_r = apex[0], apex[1], apex[2], apex[3]
                    rectified_groundtruth = torch.zeros_like(modified_input)
                    rectified_plane = torch.zeros_like(masks)
                    rectified_crop_padded_image = torch.zeros_like(modified_input)
                    rectified_mask = torch.zeros_like(masks)
                    for image_idx in range(self.real_H.shape[0]):
                        h_start, h_end, w_start, w_end = H_S[image_idx].item(), H_E[image_idx].item(), W_S[image_idx].item(), W_E[image_idx].item()
                        # h_start, h_end, w_start, w_end = H_S, H_E, W_S, W_E
                        rate_h = (h_end - h_start) / (h_end_r - h_start_r)
                        rate_w = (w_end - w_start) / (w_end_r - w_start_r)
                        # print("rate_h {} rate_w {}".format(rate_h,rate_w))
                        # rect_H_len, rect_W_len = int(self.real_H.shape[2] * rate_h), int(self.real_H.shape[3] * rate_w)
                        H_ENDx = int((h_end + (1 - h_end) * rate_h)*self.real_H.shape[3])
                        W_ENDx = int((w_end + (1 - w_end) * rate_w)*self.real_H.shape[2])
                        H_STARTx = int(H_ENDx - rate_h*self.real_H.shape[3])
                        W_STARTx = int(W_ENDx - rate_w*self.real_H.shape[2])
                        H_END_CLAMP = max(0,H_ENDx-self.real_H.shape[2])
                        H_END = min(H_ENDx,self.real_H.shape[2])
                        H_START_CLAMP = max(0,-H_STARTx)
                        H_START = max(0,H_STARTx)
                        W_END_CLAMP = max(0,W_ENDx-self.real_H.shape[3])
                        W_END = min(W_ENDx,self.real_H.shape[3])
                        W_START_CLAMP = max(0,-W_STARTx)
                        W_START = max(0,W_STARTx)

                        rectified_plane[image_idx, :, H_START:H_END, W_START:W_END] = 1
                        # rect_H_len, rect_W_len = int(self.real_H.shape[2] * H_ENDx) - int(self.real_H.shape[2] * H_STARTx),\
                        #                          int(self.real_H.shape[3] * W_ENDx) - int(self.real_H.shape[3] * W_STARTx)
                        rect_H_len, rect_W_len = H_ENDx - H_STARTx, W_ENDx-W_STARTx
                        rectified_image = Functional.interpolate(modified_input[image_idx, :, :, :].unsqueeze(0),
                            size=[H_ENDx-H_STARTx,
                                  W_ENDx-W_STARTx],
                            mode='bicubic')
                        rectified_image = torch.clamp(rectified_image,0,1)

                        crop_padded_image_resize = Functional.interpolate(cropped_padded_image[image_idx, :, :, :].unsqueeze(0),
                                                                 size=[H_ENDx - H_STARTx,
                                                                       W_ENDx - W_STARTx],
                                                                 mode='bicubic')
                        crop_padded_image_resize = torch.clamp(crop_padded_image_resize, 0, 1)
                        rectified_masks = Functional.interpolate(
                            masks[image_idx, :, :, :].unsqueeze(0),
                            size=[H_ENDx - H_STARTx,
                                  W_ENDx - W_STARTx],
                            mode='bicubic')
                        rectified_masks = torch.clamp(rectified_masks, 0, 1)
                        # print("rect_H_len {} rect_W_len {} H_START:H_END {} W_START:W_END {} H_START_CLAMP:H_END_CLAMP {} W_START_CLAMP:W_END_CLAMP {}"
                        #       .format(rect_H_len, rect_W_len, H_START-H_END, W_START-W_END,H_START_CLAMP-rect_H_len+H_END_CLAMP,W_START_CLAMP-rect_W_len+W_END_CLAMP))
                        rectified_groundtruth[image_idx,:, H_START:H_END, W_START:W_END] = \
                            rectified_image[0,:,H_START_CLAMP:rect_H_len-H_END_CLAMP,
                                                W_START_CLAMP:rect_W_len-W_END_CLAMP]

                        rectified_crop_padded_image[image_idx, :, H_START:H_END, W_START:W_END] = \
                            crop_padded_image_resize[0, :, H_START_CLAMP:rect_H_len - H_END_CLAMP,
                            W_START_CLAMP:rect_W_len - W_END_CLAMP]

                        rectified_mask[image_idx, :, H_START:H_END, W_START:W_END] = \
                            rectified_masks[0, :, H_START_CLAMP:rect_H_len - H_END_CLAMP,
                            W_START_CLAMP:rect_W_len - W_END_CLAMP]

                    ################ END rectify the GT ##################
                    ## discriminator ##
                    dis_input_real = modified_input.clone().detach()
                    dis_input_fake = forward_image.clone().detach()
                    dis_real, _ = self.discriminator(dis_input_real)
                    dis_fake, _ = self.discriminator(dis_input_fake)
                    dis_real_loss = self.adversarial_loss(dis_real, True, True)
                    dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
                    dis_loss = (dis_real_loss + dis_fake_loss) / 2
                    dis_loss.backward()
                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                                 self.train_opt['gradient_clipping'])
                    self.optimizer_discriminator.step()
                    self.optimizer_discriminator.zero_grad()

                    gen_input_fake = forward_image
                    # gen_fake = self.discriminator(gen_input_fake)
                    _, feat_fake = self.discriminator(gen_input_fake)
                    _, feat_real = self.discriminator(dis_input_real)
                    gen_gan_loss = 0
                    for feat_level in range(len(feat_fake)):
                        gen_gan_loss += self.l1_loss(input=feat_fake[feat_level], target=feat_real[feat_level])
                    gen_gan_loss /= len(feat_fake)

                    reversed_image = self.netG(rectified_crop_padded_image, rev=True) #,rectified_crop_padded_image
                    reversed_image = torch.clamp(reversed_image, 0, 1)

                    # ## discriminator for recovery ##
                    dis_input_real = modified_input.clone().detach()
                    dis_input_fake = reversed_image.clone().detach()
                    dis_real, _ = self.generator(dis_input_real)
                    dis_fake, _ = self.generator(dis_input_fake)
                    dis_real_loss = self.adversarial_loss(dis_real, True, True)
                    dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
                    dis_loss = (dis_real_loss + dis_fake_loss) / 2
                    dis_loss.backward()
                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.generator.parameters(),
                                                 self.train_opt['gradient_clipping'])
                    self.optimizer_generator.step()
                    self.optimizer_generator.zero_grad()

                    gen_input_fake = reversed_image
                    gen_fake, gen_fake_feat = self.generator(gen_input_fake)
                    # _, feat_fake = self.generator(gen_input_fake)
                    # _, feat_real = self.generator(dis_input_real)
                    # gen_recovery_dis_loss = 0
                    # for feat_level in range(len(feat_fake)):
                    #     gen_recovery_dis_loss += self.l1_loss(input=feat_fake[feat_level], target=feat_real[feat_level])
                    # gen_recovery_dis_loss /= len(feat_fake)
                    gen_recovery_dis_loss = self.adversarial_loss(gen_fake, True, False)

                    use_l1 = True  # np.random.rand() > 0.5
                    ########### strategic loss ###########
                    _, l_forward_l1, l_forward = self.loss_forward_and_backward_imuge(forward_image, modified_input,
                                                                                      masks=None, use_l1=use_l1, use_percept=False, use_vgg=False)
                    # l_backward_l1_local, l_backward_l1, l_backward = self.loss_forward_and_backward_imuge(
                    #     reversed_image, modified_input, masks=torch.ones_like(reversed_image), use_l1=use_l1, use_percept=False, use_vgg=False)
                    l_backward_l1_local, l_backward_l1, l_backward = self.loss_forward_and_backward_imuge(
                        reversed_image, rectified_groundtruth*rectified_plane, masks=rectified_mask, use_l1=use_l1, use_percept=False, use_vgg=False)

                    alpha_forw, alpha_back, gamma, delta, epsilon = 1.25, 1.0, 0.01, 0.1, 0.01

                    loss = alpha_forw * l_forward + alpha_back * l_backward + gamma * crop_loss + delta * gen_gan_loss + epsilon*gen_recovery_dis_loss
                    # if attack_full_name in {"salt", "gaussian"}:
                    # loss += 0.5*l_denoise_level_loss


                    psnr_forward = self.psnr(self.postprocess(modified_input), self.postprocess(forward_image))
                    psnr_process = self.psnr(self.postprocess(scaled_cropped_attacked.clone().detach()), self.postprocess(scaled_cropped_attacked))
                    psnr_backward = self.psnr(self.postprocess(rectified_groundtruth*rectified_plane), self.postprocess(reversed_image))
                    # psnr_backward_GT = self.psnr(self.postprocess(modified_input),
                    #                              self.postprocess(reversed_image * masks + modified_input * (1 - masks)))


                    loss.backward()
                    # print(rectified_crop_padded_image.requires_grad) #None
                    # print(rectified_crop_padded_image.grad) # None
                    # gradient clipping
                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
                        # if attack_full_name in {"salt", "gaussian"}:
                        nn.utils.clip_grad_norm_(self.generator.parameters(), self.train_opt['gradient_clipping'])
                    self.optimizer_G.step()
                    # if attack_full_name in {"salt","gaussian"}:
                    self.optimizer_generator.step()
                    self.networks_zerograd()

                ####### Validation ##################################################################
                ####### Ideal Crop adjust ##########
                # rectified_crop_padded_image = torch.zeros_like(self.real_H)
                # # h_start, h_end, w_start, w_end = apex
                # for image_idx in range(self.real_H.shape[0]):
                #     h_start_pre = max(0, int(self.real_H.shape[2] * apex[0]))
                #     h_end_pre = min(self.real_H.shape[2] - 1, int(self.real_H.shape[2] * apex[1]))
                #     w_start_pre = max(0, int(self.real_H.shape[3] * apex[2]))
                #     w_end_pre = min(self.real_H.shape[3] - 1, int(self.real_H.shape[3] * apex[3]))
                #     # print("{} {} {} {}".format(h_start_pre,h_end_pre,w_start_pre,w_end_pre))
                #     scaled_cropped_images = Functional.interpolate(
                #         denoise_forward[image_idx, :, :, :].unsqueeze(0),
                #         size=[h_end_pre - h_start_pre, w_end_pre - w_start_pre],
                #         mode='bicubic')
                #     rectified_crop_padded_image[image_idx, :, h_start_pre:h_end_pre,
                #     w_start_pre:w_end_pre] = scaled_cropped_images.squeeze(0)


                    ssim_forw, ssim_back = [], []
                    for ii in range(batch_size):
                        ssim_forw.append(
                            structural_similarity(self.tensor_to_image(modified_input[ii]),
                                                  self.tensor_to_image(forward_image[ii]),
                                                  multichannel=True))
                        ssim_back.append(
                            structural_similarity(self.tensor_to_image(rectified_groundtruth[ii]*rectified_plane[ii]),
                                                  self.tensor_to_image(reversed_image[ii]),
                                                  multichannel=True))

                    logs.append(('PSprocess', psnr_process.item()))
                    # logs.append(('PSDNrecon', psnr_recon_GT.item()))
                    # logs.append(('PSDNrecover', psnr_recover_GT.item()))
                    # logs.append(('lGTrecover', l_train_jpeg.item()))
                    # logs.append(('lFeatloss', l_denoise_level_loss.item()))
                    # logs.append(('lDNrecon', l_train_denoise.item()))
                    logs.append(('Crop', crop_loss.item()))
                    logs.append(('GAN', gen_gan_loss.item()))
                    # logs.append(('GANep', gen_recovery_dis_loss.item()))
                    logs.append(('lL1', l_backward_l1_local.item()))
                    # logs.append(('lQFr', l_qf_r.item()))
                    # logs.append(('lQFJ', l_QF_simul_JPEG.item()))
                    # logs.append(('lQFJRecon', l_QF_recon_JPEG.item()))
                    # logs.append(('lJPG', l_jpegloss.item()))
                    # logs.append(('lRecon', l_jpegloss_recon.item()))
                    # logs.append(('AdvJPG', jpeg_adv_loss.item()))

                    logs.append(('AGT', apex_item))
                    logs.append(('A', croppred_item))
                    logs.append(('psnF', psnr_forward.item()))
                    logs.append(('psnB', psnr_backward.item()))
                    # logs.append(('psnBGT', psnr_backward_GT.item()))
                    # logs.append(('psnJPG', psnr_JPEG.item()))
                    # logs.append(('psnJPGRecon', psnr_JPEG1.item()))
                    logs.append(('ssiF', round(np.mean(ssim_forw), 4)))
                    logs.append(('ssiB', round(np.mean(ssim_back), 4)))

                    logs.append(('fL1', l_forward_l1.item()))
                    logs.append(('bL1', l_backward_l1.item()))
                    # logs.append(('direct', l_direct))
                    logs.append(('FW', round(psnr_forward.item(), 4)))
                    logs.append(('BK', round(psnr_backward.item(), 4)))
                    logs.append(('SSBK', round(np.mean(ssim_back), 4)))
                    logs.append(('RATE', round(crop_rate.item(), 4)))
                    logs.append(('LC', l_backward_l1_local.item()))
                    logs.append(("Kind", attack_full_name))



                # with torch.no_grad():
                    ####### actual Crop adjust ##########
                    # rectified_crop_padded_image_1 = cropped_padded_image
                    rectified_crop_padded_image = torch.zeros_like(self.real_H)
                    # h_start, h_end, w_start, w_end = apex
                    for image_idx in range(self.real_H.shape[0]):
                        h_start_pre = max(0,int(self.real_H.shape[2] * location[image_idx, 0]))
                        h_end_pre = min(self.real_H.shape[2] - 1,
                                        int(self.real_H.shape[2] * location[image_idx, 1]))
                        w_start_pre = max(0,int(self.real_H.shape[3] * location[image_idx, 2]))
                        w_end_pre = min(self.real_H.shape[3] - 1,
                                        int(self.real_H.shape[3] * location[image_idx, 3]))
                        # print("{} {} {} {}".format(h_start_pre,h_end_pre,w_start_pre,w_end_pre))
                        scaled_cropped_images = Functional.interpolate(
                            scaled_cropped_attacked[image_idx, :, :, :].unsqueeze(0),
                            size=[h_end_pre - h_start_pre, w_end_pre - w_start_pre],
                            mode='bicubic')
                        rectified_crop_padded_image[image_idx, :, h_start_pre:h_end_pre,w_start_pre:w_end_pre] = scaled_cropped_images.squeeze(0)

                    # ########## ideal result: we do not need to adjust the mask. See what will happen in this case #############
                    reversed_image_1 = self.netG(rectified_crop_padded_image, rev=True)
                    reversed_image_1 = torch.clamp(reversed_image_1, 0, 1)

                    if step % 100 == 99:
                        images = stitch_images(
                            self.postprocess(modified_input),
                            # self.postprocess(self.jpeg_real_H),
                            # self.postprocess(10 * torch.abs(self.real_H - self.jpeg_real_H)),
                            # self.postprocess(generated_jpeg),
                            # self.postprocess(reconstructed_jpeg),
                            # self.postprocess(10 * torch.abs(generated_jpeg - self.jpeg_real_H)),
                            # self.postprocess(10 * torch.abs(reconstructed_jpeg - self.jpeg_real_H)),
                            # self.postprocess(modified_input),
                            self.postprocess(forward_image),
                            self.postprocess(10 * torch.abs(modified_input - forward_image)),
                            # self.postprocess(cropmask),
                            self.postprocess(scaled_cropped_attacked),

                            self.postprocess(reversed_image),
                            self.postprocess(10 * torch.abs(modified_input - reversed_image)),

                            self.postprocess(masks[:, :1, :, :]),
                            self.postprocess(predicted_crop_region),
                            # self.postprocess(cropped_padded_image),
                            self.postprocess(rectified_crop_padded_image),
                            self.postprocess(reversed_image_1),
                            self.postprocess(10 * torch.abs(modified_input - reversed_image_1)),
                            # self.postprocess(rectified_groundtruth),
                            # self.postprocess(rectified_plane),
                            # self.postprocess(denoise_forward),
                            # self.postprocess(GT_forward),
                            # self.postprocess(reversed_image_GT),

                            img_per_row=1
                        )

                        name = os.path.join('./images/out_muging/Test1/', str(step).zfill(5) + "_" + str(
                            dist.get_rank()) + "_" + attack_full_name + ".png")
                        print('\nsaving sample ' + name)
                        images.save(name)


        elif self.task_name == self.TASK_CropLocalize:
            ######## ICASSP No way to crop ############
            cropped_forward = None
            with torch.enable_grad():
                if eval_dir is not None:

                    ### Evaluate ######################
                    eval_data, water_data = eval_dir['eval_data'], eval_dir['water_data']
                    val_path, water_path, save_path = eval_dir['val_path'], eval_dir['water_path'], eval_dir[
                        'save_path']
                    tamper_data, mask_data = eval_dir['tamper_data'], eval_dir['mask_data']
                    source_tamper_path, predicted_mask_tamper_path, gt_mask_tamper_path = eval_dir[
                                                                                              'source_tamper_path'], \
                                                                                          eval_dir[
                                                                                              'predicted_mask_tamper_path'], \
                                                                                          eval_dir[
                                                                                              'gt_mask_tamper_path']

                    val_path = os.path.join(val_path , eval_data)
                    water_path = os.path.join(water_path , water_data)
                    source_tamper_path = os.path.join(source_tamper_path , tamper_data)
                    gt_mask_tamper_path = os.path.join(gt_mask_tamper_path , mask_data)
                    print("Test: {}  {}".format(val_path,water_path))
                    # print("Tamper: {}  {}".format(tamper_data, mask_data))

                    tensor_c = self.load_image(path=val_path,grayscale=False)
                    watermark_c = self.load_image(path=water_path,grayscale=True)
                    source_tamper = self.load_image(path=source_tamper_path,grayscale=False)
                    mask_tamper = self.load_image(path=gt_mask_tamper_path,grayscale=True)
                    # gt_mask_tamper = self.load_image(path=gt_mask_tamper_path, grayscale=True)

                    TestLocalize=2
                    if TestLocalize==0:
                        input_img = torch.cat((tensor_c, watermark_c), dim=1)
                        ablation_forward = self.netG(x=input_img)
                        forward_image_1 = ablation_forward[:, :3, :, :]
                        forward_empty = ablation_forward[:, 3:, :, :]
                        layer = self.jpeg90 if dist.get_rank()==0 else self.jpeg70
                        # layer = self.gaussian_blur
                        intermediate, apex = layer(forward_image_1)
                                # self.crop(layer(forward_image_1), min_rate=0.6, max_rate=0.9, max_rate=0.8)

                        # intermediate = (1.-mask_tamper)*(intermediate*(1-gt_mask_tamper)+source_tamper*gt_mask_tamper)

                        cropped_forward_1 = torch.clamp(intermediate, 0, 1)
                        cropped_GT, _ = self.crop(watermark_c.clone().detach(), apex=apex)
                        cropped_GT= torch.clamp(cropped_GT, 0, 1)

                        ablation_reverse = self.netG(
                            torch.cat((cropped_forward_1, self.gaussian_batch(forward_empty.shape).cuda()), dim=1),
                            rev=True)
                        reversed_image_1 = ablation_reverse[:, 3:, :, :]

                        name = os.path.join(
                            save_path,
                            os.path.splitext(eval_data)[0] + "_"+str(dist.get_rank())+"_extracted_"+ str(apex) + ".png")
                        self.print_individual_image(reversed_image_1, name)

                        name = os.path.join(
                            save_path,
                            os.path.splitext(eval_data)[0] + "_"+str(dist.get_rank())+ "_intermediate.png")
                        self.print_individual_image(tensor_c, name)

                        name = os.path.join(
                            save_path,
                            os.path.splitext(eval_data)[0] + "_"+str(dist.get_rank())+ "_watermark.png")
                        self.print_individual_image(watermark_c, name)

                        name = os.path.join(
                            save_path,
                            os.path.splitext(eval_data)[0] + "_" + str(dist.get_rank()) + "_cropped_GT.png")
                        self.print_individual_image(cropped_GT,name)


                    elif TestLocalize==1:
                        input_img = torch.cat((tensor_c, watermark_c), dim=1)
                        ablation_forward = self.netG(x=input_img)
                        forward_image_1 = ablation_forward[:, :3, :, :]
                        forward_empty = ablation_forward[:, 3:, :, :]
                        layer = self.identity if dist.get_rank() == 0 else self.jpeg70
                        # intermediate, apex = self.crop(layer(forward_image_1), min_rate=0.6, max_rate=0.9,
                        #                                max_rate=0.95)

                        # cropped = torch.clamp(intermediate, 0, 1)
                        cropped = (1-mask_tamper)*forward_image_1+mask_tamper*source_tamper
                        cropped_no_embedding = (1 - mask_tamper) * tensor_c + mask_tamper * source_tamper
                        cropped = layer(cropped)
                        if layer.name=='Jpeg70':
                            cropped = self.resize(cropped)
                        cropped_no_embedding = layer(cropped_no_embedding)

                        predicted_masks = self.localizer(cropped.clone().detach())
                        predicted_masks_ablation = self.discriminator(cropped_no_embedding.clone().detach())

                        name = os.path.join(
                            save_path,
                            os.path.splitext(eval_data)[0] + "_" + str(
                                self.global_step) + "_diffuse.png")
                        self.print_individual_image(cropped,name)

                        name = os.path.join(
                            save_path,
                            os.path.splitext(eval_data)[0] + "_" + str(self.global_step) + "_"
                            + (layer.name if layer.name!='Jpeg70' else "Mantranet_identity") + ".png")
                        self.print_individual_image(predicted_masks,name)

                        name = os.path.join(
                            save_path,
                            os.path.splitext(eval_data)[0] + "_" + str(self.global_step) + "_" + layer.name + "_ablation.png")
                        self.print_individual_image(predicted_masks_ablation,name)

                    else:
                        tensor_c = self.load_image(
                            path='/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/soldier.png',
                            grayscale=False)
                        # watermark_c = load_image(path='/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/images/images/1.png',grayscale=True)
                        source_tamper = self.load_image(
                            path='/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/images/images/1.png',
                            grayscale=False)
                        mask_tamper = self.load_image(
                            path='/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/1_GT.png',
                            grayscale=True)
                        save_path = '/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real'
                        input_img = torch.cat((tensor_c, watermark_c), dim=1)
                        ablation_forward = self.netG(x=input_img)
                        forward_image_1 = ablation_forward[:, :3, :, :]

                        cropped, cropped_blank = self.crop.cropped_out(tensor_c.clone().detach(), min_rate=0.6, max_rate=0.9)

                        name = os.path.join(
                            save_path,
                            "for_experiment_marked_"+str(self.global_step)+".png")
                        self.print_individual_image(forward_image_1,name)

                        name = os.path.join(
                            save_path,
                            "for_experiment_cropped_" + str(self.global_step) + ".png")
                        self.print_individual_image(cropped,name)

                        name = os.path.join(
                            save_path,
                            "for_experiment_blank_" + str(self.global_step) + ".png")
                        self.print_individual_image(cropped_blank,name)

                    self.networks_train()
                ### End Eval ##################
                ### Train #####################
                else:

                    if self.previous_images is not None:
                        # Each time, the first image from the previous batch is used as the watermark
                        image_c = self.tensor_to_image(self.previous_images[np.random.randint(0, batch_size)])
                        image_c = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)/ 255
                        image_c = cv2.resize(image_c, dsize=(self.width_height, self.width_height),
                                             interpolation=cv2.INTER_LINEAR)
                        tensor_c = torch.from_numpy(image_c.astype(np.float32))
                        self.tensor_GT = tensor_c.expand(batch_size,1,-1,-1).cuda()
                        self.tensor_GT = torch.clamp(self.tensor_GT,0,1)

                        train_enc_dec, train_glow = False, True

                        use_l1 = np.random.rand() > 0.25
                        if train_enc_dec:
                            ############# First Round #########################
                            input_img = torch.cat((self.real_H.detach(), self.tensor_GT), dim=1)
                            forward_image = self.generator(x=input_img)
                            forward_image = self.Quantization(forward_image)
                            forward_quantize = torch.clamp(forward_image, 0, 1)
                            # logs.append(('0', torch.mean(torch.abs(forward_quantize - forward_image)).item()))
                            forward_image = forward_quantize

                            ###### Crop pred ##########
                            # mere_attack_image = self.combined_jpeg_weak(forward_image.clone().detach())
                            # cropped_forward, apex = self.crop(mere_attack_image)
                            # cropped_GT, _ = self.crop(self.tensor_GT, apex=apex)
                            # cropped_pred = self.CropPred_net(cropped_forward)
                            # l_crop = self.l1_loss(input=cropped_pred, target=cropped_GT.clone().detach())
                            # l_crop.backward()
                            # if self.train_opt['gradient_clipping']:
                            #     nn.utils.clip_grad_norm_(self.CropPred_net.parameters(), self.train_opt['gradient_clipping'])
                            # self.optimizer_CropPred.step()
                            # self.optimizer_CropPred.zero_grad()

                            ####### Attacks ############
                            intermediate, apex = self.crop(forward_image)
                            attack_layer = self.combined_jpeg_weak # mix-up jpeg layer
                            attack_layer_1 = self.combined_jpeg_strong
                            cropped_forward_0 = attack_layer(intermediate)
                            cropped_forward_1 = attack_layer_1(intermediate)
                            beta = np.random.rand()
                            cropped_forward = beta * cropped_forward_0 + (1-beta) * cropped_forward_1

                            cropped_forward_clamp = torch.clamp(cropped_forward, 0, 1)
                            # logs.append(('2', torch.mean(torch.abs(cropped_forward_clamp - cropped_forward)).item()))
                            cropped_forward = cropped_forward_clamp

                            cropped_GT, _ = self.crop(self.tensor_GT.clone().detach(), apex=apex)
                            realH_GT, _ = self.crop(self.real_H.clone().detach(), apex=apex)
                            cropped_GT, realH_GT = torch.clamp(cropped_GT,0,1), torch.clamp(realH_GT,0,1)
                            #############################

                            ###### Tamper pred ########
                            diffused_image = cropped_forward * (1 - masks) + attack_layer(self.previous_images) * masks
                            diffused_image = torch.clamp(diffused_image, 0, 1)
                            predicted_masks = self.localizer(diffused_image.clone().detach())
                            l_mask = self.bce_loss(input=predicted_masks, target=masks_GT)
                            l_mask.backward()
                            if self.train_opt['gradient_clipping']:
                                nn.utils.clip_grad_norm_(self.localizer.parameters(), self.train_opt['gradient_clipping'])
                            self.optimizer_localizer.step()
                            self.optimizer_localizer.zero_grad()

                            ### ablation on the localizer
                            diffused_image_1 = forward_image * (1 - masks) + attack_layer(self.previous_images) * masks
                            diffused_image_1 = torch.clamp(diffused_image_1, 0, 1)
                            predicted_masks_1 = self.discriminator(diffused_image_1.clone().detach())
                            l_mask_1 = self.bce_loss(input=predicted_masks_1, target=masks_GT)
                            l_mask_1.backward()
                            if self.train_opt['gradient_clipping']:
                                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.train_opt['gradient_clipping'])
                            self.optimizer_discriminator.step()
                            self.optimizer_discriminator.zero_grad()

                            ###### Second Round #############
                            reversed = self.CropPred_net(cropped_forward.detach())
                            reversed_clamp = torch.clamp(reversed, 0, 1)
                            reversed_image = reversed_clamp

                            _, l_back_image_l1, l_back_image = self.loss_forward_and_backward_imuge(reversed_image,
                                                                                                    cropped_GT.clone().detach(),
                                                                                                    masks=None, use_l1=use_l1)
                            l_back_image.backward()
                            if self.train_opt['gradient_clipping']:
                                nn.utils.clip_grad_norm_(self.CropPred_net.parameters(), self.train_opt['gradient_clipping'])
                            self.optimizer_CropPred.step()
                            self.optimizer_CropPred.zero_grad()

                            reversed = self.CropPred_net(cropped_forward)
                            reversed_clamp = torch.clamp(reversed, 0, 1)
                            reversed_image = reversed_clamp

                            ############# losses
                            _, l_forward_l1, l_forward = self.loss_forward_and_backward_imuge(forward_image, self.real_H.detach(), masks=None, use_l1=use_l1, use_vgg=True)
                            _, l_back_image_l1, l_back_image = self.loss_forward_and_backward_imuge(reversed_image, cropped_GT.clone().detach(),
                                                                                              masks=None, use_l1=use_l1)

                            psnr_forward = self.psnr(self.postprocess(self.real_H), self.postprocess(forward_image)).item()
                            psnr_back = self.psnr(self.postprocess(cropped_GT), self.postprocess(reversed_image)).item()


                            # Total loss
                            alpha = 1.0
                            beta = 0.15
                            l_forward_loss_sum = alpha * (l_forward) + beta * l_back_image #+ gen_loss_forward
                                                 # + l_forw_ce #+ 0.1 * l_bce_loss

                            l_forward_loss_sum.backward()

                            # gradient clipping
                            if self.train_opt['gradient_clipping']:
                                # nn.utils.clip_grad_norm_(self.localizer.parameters(), self.train_opt['gradient_clipping'])
                                nn.utils.clip_grad_norm_(self.generator.parameters(), self.train_opt['gradient_clipping'])
                                # nn.utils.clip_grad_norm_(self.CropPred_net.parameters(), self.train_opt['gradient_clipping'])
                            # self.optimizer_localizer.step()
                            self.optimizer_generator.step()
                            self.networks_zerograd()
                            logs.append(('bce', l_bce))
                            logs.append(('F', l_forward))
                            logs.append(('B', l_back_image))
                            logs.append(('psnr_forward', round(psnr_forward, 4)))
                            logs.append(('psnr_back', round(psnr_back, 4)))
                            logs.append(('FW', round(psnr_forward, 4)))
                            logs.append(('BK', round(psnr_back, 4)))



                        if train_glow:
                            ############## ablation #####################
                            input_img = torch.cat((self.real_H.detach(), self.tensor_GT), dim=1)
                            ablation_forward = self.netG(x=input_img)
                            ablation_forward = torch.clamp(ablation_forward, 0, 1)
                            forward_image_1 = ablation_forward[:,:3,:,:]
                            forward_empty = ablation_forward[:,3:,:,:]
                            if train_enc_dec:
                                intermediate, _ = self.crop(forward_image_1, apex=apex)
                            else:
                                intermediate, apex = self.crop(forward_image_1, min_rate=0.3)
                                cropped_GT, _ = self.crop(self.tensor_GT.clone().detach(), apex=apex)
                                realH_GT, _ = self.crop(self.real_H.clone().detach(), apex=apex)
                                cropped_GT, realH_GT = torch.clamp(cropped_GT, 0, 1), torch.clamp(realH_GT, 0, 1)
                            attack_layer = self.combined_jpeg_weak  # mix-up jpeg layer
                            attack_layer_1 = self.combined_jpeg_strong
                            cropped_forward_0 = attack_layer(intermediate)
                            cropped_forward_1 = attack_layer_1(intermediate)
                            beta = np.random.rand()
                            cropped_forward_1 = beta * cropped_forward_0 + (1 - beta) * cropped_forward_1
                            cropped_forward_1 = torch.clamp(cropped_forward_1, 0, 1)

                            ablation_reverse = self.netG(torch.cat((cropped_forward_1,self.gaussian_batch(forward_empty.shape).cuda()),dim=1), rev=True)
                            reversed_cover = ablation_reverse[:,:3,:,:]
                            reversed_image_1 = ablation_reverse[:,3:,:,:]

                            dis_input_real = self.real_H
                            dis_input_fake = forward_image_1.detach()
                            dis_real, dis_real_feat_forward = self.discriminator_mask(
                                dis_input_real)  # in: (grayscale(1) + edge(1))
                            dis_fake, dis_fake_feat = self.discriminator_mask(
                                dis_input_fake)  # in: (grayscale(1) + edge(1))
                            dis_real_loss = self.adversarial_loss(dis_real, True, True)
                            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
                            dis_loss = (dis_real_loss + dis_fake_loss) / 2
                            dis_loss.backward()
                            if self.train_opt['gradient_clipping']:
                                nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(),
                                                         self.train_opt['gradient_clipping'])
                            self.optimizer_discriminator_mask.step()
                            self.optimizer_discriminator_mask.zero_grad()

                            ########## generator adversarial loss
                            gen_loss_forward = 0
                            gen_input_fake = forward_image_1
                            gen_fake, gen_fake_feat_forward = self.discriminator_mask(
                                gen_input_fake)  # in: (grayscale(1) + edge(1))
                            gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
                            gen_loss_forward += 0.1 * gen_gan_loss

                            _, _, l_forward_ab = self.loss_forward_and_backward_imuge(forward_image_1,
                                                                                              self.real_H.detach(), masks=None,
                                                                                              use_l1=use_l1, use_vgg=False)
                            _, _, l_back_ab = self.loss_forward_and_backward_imuge(reversed_image_1,
                                                                                                    cropped_GT.clone().detach(),
                                                                                                    masks=None, use_l1=use_l1)
                            _, _, l_back_cover_ab = self.loss_forward_and_backward_imuge(reversed_cover,
                                                                                                 realH_GT.clone().detach(),
                                                                                                 masks=None, use_l1=use_l1)
                            l_back_image_empty = self.l2_loss(forward_empty,
                                                              self.gaussian_batch(forward_empty.shape).cuda())

                            psnr_forward_ab = self.psnr(self.postprocess(self.real_H), self.postprocess(forward_image_1)).item()
                            psnr_back_ab = self.psnr(self.postprocess(cropped_GT), self.postprocess(reversed_image_1)).item()

                            l_ablation = 2*l_forward_ab + 1*l_back_ab + 10 * l_back_image_empty + 0.7 * l_back_cover_ab + gen_loss_forward
                            l_ablation.backward()
                            if self.train_opt['gradient_clipping']:
                                # nn.utils.clip_grad_norm_(self.localizer.parameters(), self.train_opt['gradient_clipping'])
                                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
                                # nn.utils.clip_grad_norm_(self.CropPred_net.parameters(), self.train_opt['gradient_clipping'])
                            # self.optimizer_localizer.step()
                            self.optimizer_G.step()
                            # logs.append(('bceAB', l_bce_ab.item()))
                            logs.append(('FAB', l_forward_ab.item()))
                            logs.append(('BAB', l_back_ab.item()))
                            logs.append(('pfAB', psnr_forward_ab))
                            logs.append(('pbAB', psnr_back_ab))
                            logs.append(('FW1', round(psnr_forward_ab, 4)))
                            logs.append(('BK1', round(psnr_back_ab, 4)))
                            logs.append(('GAN', gen_loss_forward.item()))

                    with torch.no_grad():
                        save_image = False
                        if save_image and step % 200 == 100:
                            diff = 5 * torch.abs(self.real_H - forward_image)
                            diff_1 = 5 * torch.abs(self.real_H - forward_image_1)
                            images = stitch_images(
                                self.postprocess(self.real_H),
                                self.postprocess(self.tensor_GT),
                                self.postprocess(forward_image),
                                self.postprocess(diff),
                                self.postprocess(cropped_forward),
                                self.postprocess(reversed_image),
                                self.postprocess(cropped_GT),
                                self.postprocess(diffused_image),
                                self.postprocess(predicted_masks),
                                self.postprocess(masks_GT),
                                self.postprocess(5 * torch.abs(masks_GT - predicted_masks)),
                                self.postprocess(forward_image_1),
                                self.postprocess(diff_1),
                                self.postprocess(reversed_image_1),
                                # self.postprocess(predicted_masks_1),
                                img_per_row=1
                            )

                            name = os.path.join('./images_zxy/', str(step).zfill(5) + "_" + str(
                                dist.get_rank()) + "_" + attack_layer + ".png")
                            print('\nsaving sample ' + name)
                            images.save(name)

                            #### save image independently ####
                            name = os.path.join('./extracted/', str(step).zfill(5) + "_" + str(
                                dist.get_rank()) + "_")
                            for image_no in range(reversed_image_1.shape[0]):
                                camera_ready = reversed_image_1[image_no].unsqueeze(0)
                                torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                             name + str(image_no) + ".png", nrow=1, padding=0,
                                                             normalize=False)

                            name = os.path.join('./water/', str(step).zfill(5) + "_" + str(
                                dist.get_rank()) + "_")
                            for image_no in range(self.tensor_GT.shape[0]):
                                camera_ready = self.tensor_GT[image_no].unsqueeze(0)
                                torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                             name + str(image_no) + ".png",
                                                             nrow=1, padding=0, normalize=False)

                            name = os.path.join('./crop/', str(step).zfill(5) + "_" + str(
                                dist.get_rank()) + "_")
                            for image_no in range(cropped_forward_1.shape[0]):
                                camera_ready = cropped_forward_1[image_no].unsqueeze(0)
                                torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                             name + str(image_no) + ".png",
                                                             nrow=1, padding=0, normalize=False)

                            name = os.path.join('./diff/', str(step).zfill(5) + "_" + str(
                                dist.get_rank()) + "_")
                            for image_no in range(diff_1.shape[0]):
                                camera_ready = diff_1[image_no].unsqueeze(0)
                                torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                             name + str(image_no) + ".png",
                                                             nrow=1, padding=0, normalize=False)

                            name = os.path.join('./cover/', str(step).zfill(5) + "_" + str(
                                dist.get_rank()) + "_")
                            for image_no in range(self.real_H.shape[0]):
                                camera_ready = self.real_H[image_no].unsqueeze(0)
                                torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                             name + str(image_no) + ".png",
                                                             nrow=1, padding=0, normalize=False)

                            name = os.path.join('./marked/', str(step).zfill(5) + "_" + str(
                                dist.get_rank()) + "_")
                            for image_no in range(forward_image_1.shape[0]):
                                camera_ready = forward_image_1[image_no].unsqueeze(0)
                                torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                             name + str(image_no) + ".png",
                                                             nrow=1, padding=0, normalize=False)

        else:
            with torch.enable_grad():
                # We hide 3 images into a image
                if self.previous_images is not None:
                    self.tensor_GT = self.previous_images[:3,:,:,:].view(1,9,self.width_height, self.width_height)

                    self.tensor_GT = self.tensor_GT.expand(batch_size,-1,-1,-1)

                    input_img = torch.cat((self.real_H.detach(), self.tensor_GT), dim=1)
                    ablation_forward = self.netG(x=input_img)
                    ablation_forward = torch.clamp(ablation_forward, 0, 1)
                    forward_image_1 = ablation_forward[:, :3, :, :]
                    forward_empty = ablation_forward[:, 3:, :, :]
                    # mix-up jpeg layer
                    attack_layer = self.combined_jpeg_weak
                    attack_layer_1 = self.combined_jpeg_strong
                    cropped_forward_0 = attack_layer(forward_image_1)
                    cropped_forward_1 = attack_layer_1(forward_image_1)
                    alpha = 0 # np.random.rand()
                    beta = np.random.rand()
                    cropped_forward_1 = alpha*forward_image_1 + (1 - alpha) * (beta * cropped_forward_0 + (1 - beta) * cropped_forward_1)
                    if np.random.rand()<0.3:
                        cropped_forward_1 = self.resize(cropped_forward_1)
                    elif np.random.rand()<0.6:
                        cropped_forward_1 = self.gaussian_blur(cropped_forward_1)
                    # elif np.random.rand()<0.6:
                    #     cropped_forward_1 = self.gaussian(cropped_forward_1)

                    cropped_forward_1 = torch.clamp(cropped_forward_1, 0, 1)
                    # cropped_forward_1 = forward_image_1

                    ablation_reverse = self.netG(
                        torch.cat((cropped_forward_1, torch.zeros_like(forward_empty).cuda()+0.5), dim=1), rev=True)
                    ablation_reverse = torch.clamp(ablation_reverse, 0, 1)
                    reversed_cover = ablation_reverse[:, :3, :, :]
                    reversed_image_1 = ablation_reverse[:, 3:, :, :]

                    dis_input_real = self.real_H
                    dis_input_fake = forward_image_1.detach()
                    dis_real, dis_real_feat_forward = self.discriminator_mask(dis_input_real)  # in: (grayscale(1) + edge(1))
                    dis_fake, dis_fake_feat = self.discriminator_mask(dis_input_fake)  # in: (grayscale(1) + edge(1))
                    dis_real_loss = self.adversarial_loss(dis_real, True, True)
                    dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
                    dis_loss = (dis_real_loss + dis_fake_loss) / 2
                    dis_loss.backward()
                    if self.train_opt['gradient_clipping']:
                        nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(),
                                                 self.train_opt['gradient_clipping'])
                    self.optimizer_discriminator_mask.step()
                    self.optimizer_discriminator_mask.zero_grad()
                    # use_l1 = True
                    gen_loss_forward = 0
                    gen_input_fake = forward_image_1
                    gen_fake, gen_fake_feat_forward = self.discriminator_mask(
                        gen_input_fake)  # in: (grayscale(1) + edge(1))
                    gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
                    gen_loss_forward += 0.1 * gen_gan_loss

                    # _, _, l_forward_ab = self.loss_forward_and_backward_imuge(forward_image_1,
                    #                                                           self.real_H.detach(), masks=None,
                    #                                                           use_l1=use_l1, use_vgg=False)
                    # _, _, l_back_ab = self.loss_forward_and_backward_imuge(reversed_image_1,
                    #                                                        self.tensor_GT.clone().detach(),
                    #                                                        masks=None, use_l1=use_l1)
                    # _, _, l_back_cover_ab = self.loss_forward_and_backward_imuge(reversed_cover,
                    #                                                              realH_GT.clone().detach(),
                    #                                                              masks=None, use_l1=use_l1)
                    l_forward_ab_l1 = self.l1_loss(forward_image_1,self.real_H)
                    l_forward_ab = -self.ssim_loss(forward_image_1, self.real_H)
                    l_back_ab_l1 = self.l1_loss(reversed_image_1,self.tensor_GT)
                    l_back_ab = -self.ssim_loss(reversed_image_1[:,:3,:,:], self.tensor_GT[:,:3,:,:])
                    l_back_ab += -self.ssim_loss(reversed_image_1[:,3:6,:,:], self.tensor_GT[:,3:6,:,:])
                    l_back_ab += -self.ssim_loss(reversed_image_1[:,6:9,:,:], self.tensor_GT[:,6:9,:,:])
                    l_back_cover_ab_l1 = self.l1_loss(reversed_cover,self.real_H)
                    l_back_cover_ab = -self.ssim_loss(reversed_cover, self.real_H)
                    l_back_image_empty = self.l2_loss(forward_empty,
                                                      torch.zeros_like(forward_empty).cuda()+0.5)

                    psnr_forward_ab = self.psnr(self.postprocess(self.real_H), self.postprocess(forward_image_1)).item()
                    psnr_back_ab = self.psnr(self.postprocess(self.tensor_GT[:,:3,:,:]), self.postprocess(reversed_image_1[:,:3,:,:])).item()

                    l_ablation = 1.2 * (l_forward_ab_l1+l_forward_ab) + 1. * (l_back_ab_l1+l_back_ab) \
                                 + 16 * l_back_image_empty + 0.7 * (l_back_cover_ab_l1 + l_back_cover_ab)
                    l_ablation += gen_loss_forward
                    l_ablation.backward()
                    if self.train_opt['gradient_clipping']:
                        # nn.utils.clip_grad_norm_(self.localizer.parameters(), self.train_opt['gradient_clipping'])
                        nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
                        # nn.utils.clip_grad_norm_(self.CropPred_net.parameters(), self.train_opt['gradient_clipping'])
                    # self.optimizer_localizer.step()
                    self.optimizer_G.step()
                    # logs.append(('bceAB', l_bce_ab.item()))
                    logs.append(('FAB', l_forward_ab.item()))
                    logs.append(('BAB', l_back_ab.item()))
                    logs.append(('pfAB', psnr_forward_ab))
                    logs.append(('pbAB', psnr_back_ab))
                    logs.append(('FW1', round(psnr_forward_ab, 4)))
                    logs.append(('BK1', round(psnr_back_ab, 4)))
                    logs.append(('FAB1', l_forward_ab.item()))
                    logs.append(('BAB1', l_back_ab.item()))
                    logs.append(('Kind', attack_layer.name+attack_layer_1.name))
                    logs.append(('GAN', gen_loss_forward.item()))
                    logs.append(('empty', l_back_image_empty.item()))

                    with torch.no_grad():
                        if step % 200 == 100:
                            self.networks_eval()
                            realH_eval, water_eval = self.real_H[:1,:,:,:], self.tensor_GT[:1,:,:,:]
                            input_img = torch.cat((realH_eval.detach(), water_eval.detach()), dim=1)
                            ablation_forward = self.netG(x=input_img)
                            ablation_forward = torch.clamp(ablation_forward, 0, 1)
                            forward_image_1 = ablation_forward[:, :3, :, :]
                            forward_empty = ablation_forward[:, 3:, :, :]
                            diff_1 = 5 * torch.abs(realH_eval - forward_image_1)
                            # mix-up jpeg layer
                            if np.random.rand() < 0.5:
                                mixup1 = self.identity
                                mixup2 = self.combined_jpeg_weak
                            else:
                                mixup1 = self.resize
                                mixup2 = self.gaussian_blur
                            cropped_forward_1 = mixup1(forward_image_1)
                            cropped_forward_1 = torch.clamp(cropped_forward_1, 0, 1)
                            ablation_reverse1 = self.netG(
                                torch.cat((cropped_forward_1, torch.zeros_like(forward_empty).cuda() + 0.5), dim=1),
                                rev=True)
                            ablation_reverse1 = torch.clamp(ablation_reverse1, 0, 1)
                            reversed_cover_1 = ablation_reverse1[:, :3, :, :]
                            reversed_image_1 = ablation_reverse1[:, 3:, :, :]

                            cropped_forward_2 = mixup2(forward_image_1)
                            cropped_forward_2 = torch.clamp(cropped_forward_2, 0, 1)
                            ablation_reverse2 = self.netG(
                                torch.cat((cropped_forward_2, torch.zeros_like(forward_empty).cuda() + 0.5), dim=1),
                                rev=True)
                            ablation_reverse2 = torch.clamp(ablation_reverse2, 0, 1)
                            reversed_cover_2 = ablation_reverse2[:, :3, :, :]
                            reversed_image_2 = ablation_reverse2[:, 3:, :, :]
                            images = stitch_images(
                                self.postprocess(realH_eval),
                                self.postprocess(water_eval[:,:3,:,:]),
                                self.postprocess(water_eval[:, 3:6, :, :]),
                                self.postprocess(water_eval[:, 6:9, :, :]),
                                self.postprocess(forward_image_1),
                                self.postprocess(diff_1),
                                self.postprocess(reversed_image_1[:, :3, :, :]),
                                self.postprocess(reversed_image_1[:, 3:6, :, :]),
                                self.postprocess(reversed_image_1[:, 6:9, :, :]),
                                self.postprocess(reversed_image_2[:, :3, :, :]),
                                self.postprocess(reversed_image_2[:, 3:6, :, :]),
                                self.postprocess(reversed_image_2[:, 6:9, :, :]),
                                img_per_row=1
                            )
                            name = os.path.join('./images_RHI3_eval/', str(step).zfill(5) + "_" + str(
                                dist.get_rank()) + mixup2.name+ ".png")
                            print('\nsaving sample ' + name)
                            images.save(name)
                            self.networks_train()

        ######## Finally ####################
        if step % save_interval == 100:
            if dist.get_rank() <= 0:
                logger.info('Saving models and training states.')
                self.save(step)
        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H.clone().detach()
        return logs, debug_logs


                # ###### adv training ####################
                # forward_image_sum, reversed_image_sum = None, None
                # for img_no in range(batch_size):
                #     cover_image = self.real_H[img_no,:,:,:].unsqueeze(0)
                #     self.classification_net.eval()
                #     self.classification_net_1.eval()
                #     forward_image_1 = self.netG(x=cover_image)
                #     forward_clamp = torch.clamp(forward_image_1, 0, 1)
                #     # debug_logs.append(('1', torch.mean(torch.abs(forward_image - forward_clamp)).item()))
                #     # logs.append(('1', torch.mean(torch.abs(forward_image - forward_clamp)).item()))
                #     forward_image_1 = self.Quantization(forward_clamp)
                #     attacked_image_1, apex = self.crop(forward_image_1, min_rate=0.680, max_rate=0.9)
                #     GT_1, _ = self.crop(cover_image, apex=apex)
                #     attack_layer_1 = self.combined_jpeg_weak #if np.random.rand()<0.6 else self.combined_jpeg_strong
                #     attacked_image_1 = attack_layer_1(attacked_image_1)
                #     attacked_image_1 = torch.clamp(attacked_image_1, 0, 1)
                #
                #     reversed_image_1 = self.netG(attacked_image_1, rev=True)
                #     attacked_original = attack_layer_1(GT_1)
                #     # reversed_image_1 = reversed_image_1 + attacked_original
                #     reversed_image_1 = torch.clamp(reversed_image_1, 0, 1)
                #
                #     ########### discriminator loss
                #     # dis_input_real = self.real_H
                #     # dis_input_fake = reversed_image_1.detach()
                #     # dis_real, dis_real_feat = self.dis_adv_fw(dis_input_real)  # in: (grayscale(1) + edge(1))
                #     # dis_fake, dis_fake_feat = self.dis_adv_fw(dis_input_fake)  # in: (grayscale(1) + edge(1))
                #     # dis_real_loss = self.adversarial_loss(dis_real, True, True)
                #     # dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
                #     # dis_loss = (dis_real_loss + dis_fake_loss) / 2
                #     # dis_loss.backward()
                #     # if self.train_opt['gradient_clipping']:
                #     #     nn.utils.clip_grad_norm_(self.dis_adv_fw.parameters(), self.train_opt['gradient_clipping'])
                #     # self.optimizer_dis_adv_fw.step()
                #     # self.optimizer_dis_adv_fw.zero_grad()

                #     ########## generator adversarial loss
                #     # gen_loss = 0
                #     # gen_input_fake = reversed_image_1
                #     # gen_fake, gen_fake_feat = self.dis_adv_fw(gen_input_fake)
                #     # gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
                #     # gen_loss += 0.1 * gen_gan_loss
                #
                #     gen_loss_forward = 0
                #     gen_input_fake = forward_image_1
                #     gen_fake, gen_fake_feat_forward = self.dis_adv_cov(gen_input_fake)
                #     gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
                #     gen_loss_forward += 0.01 * gen_gan_loss
                #
                #     ########## generator feature matching loss
                #     # gen_fm_loss = 0
                #     # for i in range(len(dis_real_feat)):
                #     #     gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
                #     # gen_loss += 0.1 * gen_fm_loss
                #
                #     # gen_fm_loss_forward = 0
                #     # for i in range(len(dis_real_feat_forward)):
                #     #     gen_fm_loss_forward += self.l1_loss(gen_fake_feat_forward[i], dis_real_feat_forward[i].detach())
                #     # gen_loss_forward += 0.1 * gen_fm_loss_forward
                #     # l_forward_1 = 0
                #     _, l_forward_l1, l_forward_1 = self.loss_forward_and_backward_imuge(forward_image_1, cover_image.detach(),
                #                                                                       masks=None, use_l1=False, use_vgg=False)
                #     # # generator style loss
                #     # gen_style_loss = self.style_loss(forward_image_1, cover_image.detach())
                #     # gen_style_loss = gen_style_loss * 250
                #     # l_forward_1 += gen_style_loss
                #     # l_ssim_1 = - self.ssim_loss(forward_image_1,  cover_image.detach())
                #     # l_forward_1 += l_ssim_1
                #
                #     _, l_back_image_l1, l_back_image_1 = self.loss_forward_and_backward_imuge(reversed_image_1,
                #                                                                             GT_1.clone().detach(),
                #                                                                             masks=None, use_l1=False, use_vgg=False)
                #
                #     net_1 = self.classification_net if step % 2 == 0 else self.classification_net_1
                #     net_2 = self.classification_net if step % 2 == 1 else self.classification_net_1
                #     label = self.label_GT[img_no].unsqueeze(0)
                #
                #     y_ori = net_1(attacked_original.detach())
                #     l_ori = self.criterion(y_ori,label)
                #     y_pred = net_1(attacked_image_1)
                #     loss_adv = self.criterion(y_pred,label)
                #     y_rev = net_2(reversed_image_1)
                #     l_rev = self.criterion(y_rev, label)
                #     loss_CW = self.criterion_adv(y_pred, label, False, num_classes=1000)
                #     loss_adv_sum = 0
                #     loss_adv_sum += 1 * gen_loss_forward # + 1 * gen_loss
                #     loss_adv_sum += 1 * l_back_image_1 + 1.2 * l_forward_1
                #     psnr_forward_1 = self.psnr(self.postprocess(cover_image), self.postprocess(forward_image_1)).item()
                #     psnr_back_1 = self.psnr(self.postprocess(GT_1), self.postprocess(reversed_image_1)).item()
                #
                #     alpha = 0.02 if loss_adv<6 else 0.002
                #     beta = 0.001 if l_rev-l_ori>0.5 else 0.0005
                #     loss_adv_sum += -alpha * loss_adv + beta * l_rev
                #     loss_adv_sum.backward()
                #     if self.train_opt['gradient_clipping']:
                #         nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
                #     self.optimizer_G.step()
                #     self.optimizer_G.zero_grad()
                #
                #     if step % 200 == 100:
                #         if forward_image_sum is None:
                #             forward_image_sum, reversed_image_sum = forward_image_1, reversed_image_1
                #         else:
                #             forward_image_sum = torch.cat((forward_image_sum,forward_image_1),dim=0)
                #             reversed_image_sum = torch.cat((reversed_image_sum,reversed_image_1),dim=0)

                ####### Attack Net train ##########
                # image_real_attack = attack_layer(self.real_H)
                # image_simulated_attack = self.attack_net(self.real_H)
                # _, l_forward_atk_l1, l_forward_atk = self.loss_forward_and_backward_imuge \
                #     (image_simulated_attack, image_real_attack.clone().detach(), masks=None, use_l1=use_l1)
                # l_forward_atk.backward()
                # # gradient clipping
                # if self.train_opt['gradient_clipping']:
                #     nn.utils.clip_grad_norm_(self.attack_net.parameters(), self.train_opt['gradient_clipping'])
                # self.optimizer_attacker.step()

    def print_individual_image(self, cropped_GT, name):
        for image_no in range(cropped_GT.shape[0]):
            camera_ready = cropped_GT[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         name, nrow=1, padding=0, normalize=False)

    def load_image(self,path, grayscale):
        image_c = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1] if not grayscale else cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image_c = cv2.resize(image_c, dsize=(self.width_height, self.width_height), interpolation=cv2.INTER_LINEAR)
        img = image_c.copy().astype(np.float32)
        img /= 255.0
        if not grayscale:
            img = img.transpose(2, 0, 1)
        tensor_c = torch.from_numpy(img).unsqueeze(0).cuda()
        if grayscale:
            tensor_c = tensor_c.unsqueeze(0)

        return tensor_c

    def perform_attack(self,step,forward_image,masks,disable_hard=False):
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

        diffused_image = attacked_image * (1 - masks) \
                         + (attacked_previous_images if self.previous_images is not None else torch.zeros_like(
            self.real_H)) * masks


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
        image = tensor.permute(1,2,0).detach().cpu().numpy()
        # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def tensor_to_image_batch(self, tensor):

        tensor = tensor * 255.0
        image = tensor.permute(0,2,3,1).detach().cpu().numpy()
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
        img_t = F.to_tensor(img).float()
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

                load_path_G = self.opt['path']['pretrain_model']+"_G.pth"
                if load_path_G is not None:
                    if self.opt['train']['load'] == 2.0:
                        load_path_G = '../experiments/pretrained_models/G_latest.pth'
                    logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                    if os.path.exists(load_path_G):
                        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
                    else:
                        logger.info('Did not find model for G [{:s}] ...'.format(load_path_G))

                load_path_L = self.opt['path']['pretrain_model']+"_L.pth"
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
        self.save_network(self.generator_additional, 'A', iter_label)
        if self.opt['train']['save_to_latest'] > 0.0:
            self.save_network(self.generator_additional, 'A', iter_label, save_dir='../experiments/pretrained_zxy/')
        if self.task_name == self.TASK_TEST:
            self.save_network(self.attack_net, 'A_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.attack_net, 'A_zxy', iter_label, save_dir='../experiments/pretrained_zxy/')
        elif self.task_name == self.TASK_IMUGEV2:
            self.save_network(self.dis_adv_cov, 'dis_adv_cov', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.dis_adv_cov, 'dis_adv_cov', iter_label,save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.dis_adv_fw, 'dis_adv_fw', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.dis_adv_fw, 'dis_adv_fw', iter_label,save_dir='../experiments/pretrained_models/')
            self.save_network(self.attack_net, 'A_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.attack_net, 'A_zxy', iter_label, save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.CropPred_net, 'apex_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.CropPred_net, 'apex_zxy', iter_label,save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.discriminator, 'D_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.discriminator, 'D_zxy', iter_label, save_dir='../experiments/pretrained_models/')
            self.save_network(self.netG, 'G', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.netG, 'G', iter_label, save_dir='../experiments/pretrained_models/')
            self.save_network(self.generator, 'G_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.generator, 'G_zxy', iter_label, save_dir='../experiments/pretrained_models/')
            self.save_network(self.localizer, 'L', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.localizer, 'L', iter_label, save_dir='../experiments/pretrained_models/')
            self.save_network(self.discriminator_mask, 'D_mask_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.discriminator_mask, 'D_mask_zxy', iter_label,save_dir='../experiments/pretrained_models/')
        elif self.task_name == self.TASK_CropLocalize:
            #### netG localizer attack_net generator discriminator discriminator_mask CropPred_net dis_adv_fw dis_adv_cov
            self.save_network(self.netG, 'G', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.netG, 'G', iter_label, save_dir='../experiments/pretrained_models/')
            self.save_network(self.localizer, 'L_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.localizer, 'L_zxy', iter_label, save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.attack_net, 'A_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.attack_net, 'A_zxy', iter_label, save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.generator, 'G_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.generator, 'G_zxy', iter_label, save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.discriminator, 'D_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.discriminator, 'D_zxy', iter_label, save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.discriminator_mask, 'D_mask_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.discriminator_mask, 'D_mask_zxy', iter_label, save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.CropPred_net, 'apex_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.CropPred_net, 'apex_zxy', iter_label,
                                  save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.dis_adv_fw, 'dis_adv_fw', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.dis_adv_fw, 'dis_adv_fw', iter_label,
                                  save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.dis_adv_cov, 'dis_adv_cov', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.dis_adv_cov, 'dis_adv_cov', iter_label,
                                  save_dir='../experiments/pretrained_zxy/')
        else:
            self.save_network(self.discriminator_mask, 'D_mask_zxy', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.discriminator_mask, 'D_mask_zxy', iter_label,
                                  save_dir='../experiments/pretrained_zxy/')
            self.save_network(self.netG, 'G', iter_label)
            if self.opt['train']['save_to_latest'] > 0.0:
                self.save_network(self.netG, 'G', iter_label, save_dir='../experiments/pretrained_models/')

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
