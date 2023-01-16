import copy
# from .networks import SPADE_UNet
import os
import random
from collections import OrderedDict

from skimage.feature import canny
import cv2
import torch.distributed as dist
import torch.nn as nn
# from data.pipeline import pipeline_tensor2image
# import matlab.engine
import torch.nn.functional as Functional
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from skimage.color import rgb2gray
from torch.nn.parallel import DistributedDataParallel

# from cycleisp_models.cycleisp import Raw2Rgb
import pytorch_ssim
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
from detection_methods.MantraNet.mantranet import pre_trained_model
# from .invertible_net import Inveritible_Decolorization_PAMI
# from models.networks import UNetDiscriminator
# from loss import PerceptualLoss, StyleLoss
# from .networks import SPADE_UNet
# from lama_models.HWMNet import HWMNet
# import contextual_loss as cl
# import contextual_loss.functional as F
from losses.loss import GrayscaleLoss
from losses.loss import PerceptualLoss, StyleLoss
from models.modules.Quantization import diff_round
from noise_layers import *
from noise_layers.dropout import Dropout
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.resize import Resize
from utils.JPEG import DiffJPEG
from utils.commons import create_folder
# from MVSS.models.mvssnet import get_mvss
# from MVSS.models.resfcn import ResFCN
from utils.metrics import PSNR


class BaseModel():
    def __init__(self, opt,  args, train_set=None, val_set=None):
        ### todo: options
        self.opt = opt
        # self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.world_size = torch.distributed.get_world_size()
        self.train_set = train_set
        self.val_set = val_set
        self.rank = torch.distributed.get_rank()
        self.opt = opt
        self.args = args
        self.train_opt = opt['train']
        # self.test_opt = opt['test']

        self.task_name = args.task_name
        print("Task Name: {}".format(self.task_name))

        ### todo: constants
        self.global_step = 0
        self.width_height = opt['datasets']['train']['GT_size']

        self.amount_of_augmentation = len(
            self.opt['simulated_hue'] +
            self.opt['simulated_contrast'] +
            self.opt['simulated_saturation'] +
            self.opt['simulated_brightness'] +
            self.opt['simulated_gamma']
        )

        # self.amount_of_detectors = len(
        #     self.opt['detector_using_MPF_indices'] +
        #     self.opt['detector_using_MVSS_indices'] +
        #     self.opt['detector_using_OSN_indices']
        # )

        self.amount_of_inpainting = len(
            self.opt['zits_as_inpainting'] +
            self.opt['edgeconnect_as_inpainting'] +
            self.opt['lama_as_inpainting'] +
            self.opt['ideal_as_inpainting']
        )

        self.amount_of_tampering = len(
            self.opt['simulated_splicing_indices'] +
            self.opt['simulated_copymove_indices'] +
            self.opt['simulated_inpainting_indices'] +
            self.opt['simulated_copysplicing_indices']
        )
        self.amount_of_benign_attack = len(
            self.opt['simulated_resize_indices'] +
            self.opt['simulated_gblur_indices'] +
            self.opt['simulated_mblur_indices'] +
            self.opt['simulated_AWGN_indices'] +
            self.opt['simulated_strong_JPEG_indices'] +
            self.opt['simulated_weak_JPEG_indices']
        )
        self.history_attack_loss = {"resize":0, "gblur":0, "mblur":0, "AWGN":0, "JPEG":0}
        self.history_attack_times = {"resize":0, "gblur":0, "mblur":0, "AWGN":0, "JPEG":0}
        self.history_attack_PSNR = {"resize":0, "gblur":0, "mblur":0, "AWGN":0, "JPEG":0}
        self.history_attack_CE = {"resize":0, "gblur":0, "mblur":0, "AWGN":0, "JPEG":0}
        self.history_attack_data = {"resize": 0, "gblur": 0, "mblur": 0, "AWGN": 0, "JPEG": 0}
        self.history_term_mapping = {
                                        'simulated_resize_indices':'resize',
                                        'simulated_gblur_indices':'gblur',
                                        'simulated_mblur_indices':'mblur',
                                        'simulated_AWGN_indices':'AWGN',
                                        'simulated_strong_JPEG_indices':'JPEG',
                                        'simulated_weak_JPEG_indices':'JPEG'
        }
        self.index_to_attack = {}
        for attack_name in self.history_term_mapping:
            for item in self.opt[attack_name]:
                self.index_to_attack[item] = self.history_term_mapping[attack_name]


        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

        ### todo: network definitions
        self.network_list = []
        self.real_H, self.real_H_path, self.previous_images = None, None, None
        self.previous_previous_images, self.previous_previous_canny = None, None
        self.previous_protected, self.previous_canny = None, None

        # self.optimizer_G = None
        # self.optimizer_localizer = None
        # self.optimizer_discriminator_mask = None
        # self.optimizer_discriminator = None
        # self.optimizer_KD_JPEG = None
        # self.optimizer_generator = None
        # self.optimizer_qf_predict = None
        # self.netG = None
        # self.localizer = None
        # self.discriminator = None
        # self.discriminator_mask = None
        # self.KD_JPEG = None
        self.global_step = 0
        self.out_space_storage = ""


        ### todo: losses and attack layers
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
        self.gaussian_blur = GaussianBlur(opt=self.opt).cuda()
        self.median_blur = MiddleBlur(opt=self.opt).cuda()
        self.resize = Resize(opt=self.opt).cuda()
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

        self.ce_loss = nn.CrossEntropyLoss().cuda()
        self.bce_loss = nn.BCELoss().cuda()
        self.bce_with_logit_loss = nn.BCEWithLogitsLoss().cuda()
        self.l1_loss = nn.SmoothL1Loss(beta=0.5).cuda()  # reduction="sum"
        self.consine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.hard_l1_loss = nn.L1Loss().cuda()  # reduction="sum"
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
        self.batch_size = opt['datasets']['train']['batch_size']
        self.init_gaussian = None
        # self.adversarial_loss = AdversarialLoss(type="nsgan").cuda()


    ### todo: Abstract Methods

    def optimize_parameters_router(self, mode, step=None, epoch=None):
        pass

    def validate_router(self, mode, step=None, epoch=None):
        pass

    def feed_data_router(self, batch, mode):
        pass

    def feed_data_val_router(self, batch, mode):
        pass

    def gaussian_batch(self, dims):
        return self.clamp_with_grad(torch.randn(tuple(dims)).cuda())

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def define_optimizers(self):
        wd_G = self.train_opt['weight_decay_G'] if 'weight_decay_G' in self.train_opt else 0

        if 'netG' in self.network_list:
            lr = 'lr_scratch'
            self.optimizer_G = self.create_optimizer(self.netG,
                                                     lr=self.train_opt[lr], weight_decay=wd_G)
            print(f"optimizer netG: {lr}")

        if 'discriminator_mask' in self.network_list:
            lr = 'lr_scratch'
            self.optimizer_discriminator_mask = self.create_optimizer(self.discriminator_mask,
                                                                      lr=self.train_opt[lr],weight_decay=wd_G)
            print(f"optimizer discriminator_mask: {lr}")
        if 'localizer' in self.network_list:
            lr = 'lr_scratch'
            self.optimizer_localizer = self.create_optimizer(self.localizer,
                                                             lr=self.train_opt[lr], weight_decay=wd_G)
            print(f"optimizer localizer: {lr}")
        if 'KD_JPEG' in self.network_list:
            lr = 'lr_scratch'
            self.optimizer_KD_JPEG = self.create_optimizer(self.KD_JPEG,
                                                           lr=self.train_opt[lr], weight_decay=wd_G)
        if 'discriminator' in self.network_list:
            self.optimizer_discriminator = self.create_optimizer(self.discriminator,
                                                                 lr=self.train_opt['lr_scratch'], weight_decay=wd_G)

        if 'generator' in self.network_list:
            lr = 'lr_scratch' if not 'lr_generator' in self.train_opt else 'lr_generator'
            self.optimizer_generator = self.create_optimizer(self.generator,
                                                             lr=self.train_opt[lr], weight_decay=wd_G)
        if 'qf_predict_network' in self.network_list:
            lr = 'lr_scratch'
            self.optimizer_qf = self.create_optimizer(self.qf_predict_network,
                                                      lr=self.train_opt[lr], weight_decay=wd_G)


    ### todo: Helper functions
    def exponential_weight_for_backward(self, *, value, exp=1.5, norm=1, alpha=0.5, psnr_thresh=None):
        '''
            exponential loss for recovery loss's weight.
            PSNR  29     30     31     32     33(base)   34     35
            Weigh 0.161  0.192  0.231  0.277  0.333      0.400  0.500
        '''
        if 'exp_weight' in self.opt:
            exp = self.opt['exp_weight']
        if 'CE_hyper_param' in self.opt:
            alpha = self.opt['CE_hyper_param']
        if psnr_thresh is None:
            psnr_thresh = self.opt['psnr_thresh']
        return min(1, alpha*((exp)**(norm*(value-psnr_thresh))))

    def update_history_losses(self, *, index, PSNR, loss, loss_CE, PSNR_attack):
        '''
            print loss history so that we can see
            under which attack the performance is the worst.
        '''
        if index is None:
            index = self.global_step
        index = index % self.amount_of_benign_attack
        prev_loss = self.history_attack_loss[self.index_to_attack[index]]
        prev_psnr = self.history_attack_PSNR[self.index_to_attack[index]]
        prev_times = self.history_attack_times[self.index_to_attack[index]]
        prev_CE = self.history_attack_CE[self.index_to_attack[index]]
        prev_attack = self.history_attack_data[self.index_to_attack[index]]
        self.history_attack_loss[self.index_to_attack[index]] = (prev_loss*prev_times + loss)/(prev_times+1)
        self.history_attack_PSNR[self.index_to_attack[index]] = (prev_psnr*prev_times + PSNR)/(prev_times+1)
        self.history_attack_CE[self.index_to_attack[index]] = (prev_CE * prev_times + loss_CE) / (prev_times + 1)
        self.history_attack_data[self.index_to_attack[index]] = (prev_attack * prev_times + PSNR_attack) / (prev_times + 1)
        self.history_attack_times[self.index_to_attack[index]] = prev_times + 1

        if self.global_step % 200 == 199 or self.global_step <= 10:
            print(f"history loss: {self.history_attack_loss}")
            print(f"history PSNR: {self.history_attack_PSNR}")
            print(f"history CE: {self.history_attack_CE}")
            print(f"history times: {self.history_attack_times}")
            print(f"history attack: {self.history_attack_data}")

    def print_this_image(self, image, filename):
        '''
            the input should be sized [C,H,W], not [N,C,H,W]
        '''
        camera_ready = image.unsqueeze(0)
        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                     filename, nrow=1,
                                     padding=0, normalize=False)

    ### todo: optimizer
    def create_optimizer(self, net, lr=1e-4, weight_decay=0):
        ## lr should be train_opt['lr_scratch'] in default
        optim_params = []
        for k, v in net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            # else:
            #     if self.rank <= 0:
            #         print('Params [{:s}] will not optimize.'.format(k))
        optimizer = torch.optim.AdamW(optim_params, lr=lr,
                                      weight_decay=weight_decay,
                                      betas=(0.9, 0.99))  # train_opt['beta1'], train_opt['beta2']
        self.optimizers.append(optimizer)

        return optimizer

    ### todo: folders
    def create_folders_for_the_experiment(self):
        pass

    def define_ddpm_unet_network(self, out_dim=3, dim = 32, use_bayar=False, use_fft=False, use_classification=False,
                                 use_middle_features=False, use_hierarchical_class=False):
        from network.CNN_architectures.ddpm_lucidrains import Unet
        # input = torch.ones((3, 3, 128, 128)).cuda()
        # output = model(input, torch.zeros((1)).cuda())

        print(f"using ddpm_unet, use_fft: {use_fft}, use_bayar: {use_bayar}, use_classification: {use_classification}, use_middle_features: {use_middle_features}")
        model = Unet(out_dim=out_dim, dim=dim, use_bayar=use_bayar, use_fft=use_fft, use_classification=use_classification,
                     use_middle_features=use_middle_features, use_hierarchical_class=use_hierarchical_class).cuda()
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_tres_network(self):
        from models.IFA.network.tres_model import Net

        print("using tres_model")
        model = Net(num_embeddings=1024).cuda()
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_restormer(self):
        print("using restormer as testing ISP...")
        from restoration_methods.restormer.model_restormer import Restormer
        model = Restormer(dim=16, ).cuda()
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_convnext(self, num_classes=1000, size='base'):
        print("using convnext")
        from network.CNN_architectures.convnext_official import convnext_base, convnext_large
        b = convnext_base if 'base' in size else convnext_large
        model = b(num_classes=num_classes).cuda()
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_CMT(self):
        print("using CMT as hybrid CNN+transformer model")
        from network.advanced_transformers.cmt import CMT, cmt_b
        model = cmt_b(img_size=self.width_height,num_classes=1).cuda()
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    ### todo: cropping attack
    def cropping_mask_generation(self, forward_image, locs=None, min_rate=0.6, max_rate=1.0):
        # batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        # masks_GT = torch.ones_like(self.canny_image)
        '''
            if locs is specified, the cropped mask is determined
        '''

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

    def get_quality_idx_by_iteration(self, *, index=None):
        if index is None:
            index = self.global_step
        if index % self.amount_of_benign_attack not in set(
                self.opt['simulated_strong_JPEG_indices']
        ):
            ## perform weak JPEG compression after other attacks
            quality_idx = np.random.randint(self.opt["weak_JPEG_lower_bound"], 21)
        else:
            quality_idx = np.random.randint(self.opt["strong_JPEG_lower_bound"], self.opt["strong_JPEG_upper_bound"])
        return quality_idx

    def mask_generation(self, *, modified_input, percent_range=None, index=None):
        '''
            generate free-form mask. percent can be determined.
        '''
        if index is None:
            index = self.global_step
        index = index % self.amount_of_tampering
        if percent_range is None:
            percent_range = (0.0, 0.3) if index not in self.opt["simulated_copymove_indices"] else (0.0, 0.25)

        batch_size, height_width = modified_input.shape[0], modified_input.shape[2]
        masks_GT = torch.zeros(batch_size, 1, modified_input.shape[2], modified_input.shape[3]).cuda()
        ## THE RECOVERY STAGE WILL ONLY WORK UNDER LARGE TAMPERING
        ## TO LOCALIZE SMALL TAMPERING, WE ONLY UPDATE LOCALIZER NETWORK

        for imgs in range(batch_size):
            masks_origin, _ = self.generate_stroke_mask(
                [modified_input.shape[2], modified_input.shape[3]], percent_range=percent_range)
            masks_GT[imgs, :, :, :] = masks_origin.cuda()
        masks = masks_GT.repeat(1, 3, 1, 1)

        # masks is just 3-channel-version masks_GT
        return masks, masks_GT, percent_range

    ### todo: image manipulations
    def splicing(self, *, forward_image, masks):
        return  forward_image * (1 - masks) + (self.previous_previous_images) * masks

    def inpainting_for_PAMI(self, *, forward_image, masks, modified_canny):
        with torch.no_grad():
            reversed_stuff, reverse_feature = self.netG(
                torch.cat((forward_image * (1 - masks),
                           torch.zeros_like(modified_canny)), dim=1),
                rev=True)  # torch.zeros_like(modified_canny).cuda()
            reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
            reversed_image = self.clamp_with_grad(reversed_ch1)
            # attacked_forward = forward_image * (1 - masks) + modified_input.clone().detach() * masks
            attacked_forward = forward_image * (1 - masks) + reversed_image.clone().detach() * masks
            # reversed_image = reversed_image.repeat(way_attack,1,1,1)
        del reversed_stuff
        del reverse_feature

        return attacked_forward

    def get_shifted_image_for_copymove(self, *, forward_image, percent_range, masks):
        batch_size, channels, height_width = forward_image.shape[0], forward_image.shape[1], forward_image.shape[2]
        lower_bound_percent = max(0.1, percent_range[0] + (percent_range[1] - percent_range[0]) * np.random.rand())
        ###### IMPORTANT NOTE: for ideal copy-mopv, here should be forward_image. If you want to ease the condition, can be changed to forward_iamge
        tamper = forward_image.clone().detach()
        max_x_shift, max_y_shift, valid, retried, max_valid, mask_buff = 0, 0, 0, 0, 0, None
        mask_shifted = masks
        while retried <= 20 and valid < lower_bound_percent:
            x_shift = int((height_width) * (0.2+0.4*np.random.rand())) * (1 if np.random.rand()>0.5 else -1)
            y_shift = int((height_width) * (0.2+0.4*np.random.rand())) * (1 if np.random.rand()>0.5 else -1)
            # if abs(x_shift) <= (height_width / 4) or abs(y_shift) <= (height_width / 4):
            #     continue
            ### two times padding ###
            mask_buff = torch.zeros((masks.shape[0], masks.shape[1],
                                     masks.shape[2] + abs(2 * x_shift),
                                     masks.shape[3] + abs(2 * y_shift))).cuda()

            mask_buff[:, :,
            abs(x_shift) + x_shift:abs(x_shift) + x_shift + height_width,
            abs(y_shift) + y_shift:abs(y_shift) + y_shift + height_width] = masks

            mask_buff = mask_buff[:, :,
                        abs(x_shift):abs(x_shift) + height_width,
                        abs(y_shift):abs(y_shift) + height_width]

            valid = torch.mean(mask_buff)
            retried += 1
            if valid >= max_valid:
                max_valid = valid
                mask_shifted = mask_buff
                max_x_shift, max_y_shift = x_shift, y_shift

        tamper_shifted = torch.zeros((batch_size, channels,
                                      height_width + abs(2 * max_x_shift),
                                      height_width + abs(2 * max_y_shift))).cuda()
        tamper_shifted[:, :,
        abs(max_x_shift) + max_x_shift: abs(max_x_shift) + max_x_shift + height_width,
        abs(max_y_shift) + max_y_shift: abs(max_y_shift) + max_y_shift + height_width] = tamper

        tamper_shifted = tamper_shifted[:, :,
                         abs(max_x_shift): abs(max_x_shift) + height_width,
                         abs(max_y_shift): abs(max_y_shift) + height_width]

        masks = mask_shifted.clone().detach()

        masks_GT = masks[:, :1, :, :]

        return tamper_shifted, masks, masks_GT

    def copymove(self,*, forward_image,masks, masks_GT, percent_range):
        batch_size, channels, height_width = forward_image.shape[0], forward_image.shape[1], forward_image.shape[2]
        tamper_shifted, masks, masks_GT = self.get_shifted_image_for_copymove(forward_image=forward_image, percent_range=percent_range,
                                                                                masks=masks)
        attacked_forward = forward_image * (1 - masks) + tamper_shifted.clone().detach() * masks

        return attacked_forward, masks, masks_GT

    def copysplicing(self, *, forward_image, masks, percent_range, another_immunized=None):
        with torch.no_grad():
            if another_immunized is None:
                another_generated = self.netG(
                    torch.cat([self.previous_previous_images, self.previous_previous_canny], dim=1))
                another_immunized = another_generated[:, :3, :, :]
                another_immunized = self.clamp_with_grad(another_immunized)
            tamper_shifted, masks, masks_GT = self.get_shifted_image_for_copymove(forward_image=another_immunized,
                                                                                  percent_range=percent_range, masks=masks)
            attacked_forward = forward_image * (1 - masks) + another_immunized.clone().detach() * masks
        # del another_generated

        return attacked_forward, masks, masks_GT

    def get_canny(self, input, masks_GT=None, sigma=1):
        cannied_list = torch.zeros_like(input)[:,:1]
        gray_list = torch.zeros_like(input)[:,:1]
        for i in range(input.shape[0]):
            grid = input[i]
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to('cpu', torch.uint8).numpy()
            ndarr = ndarr.astype(np.float32) / 255.
            img_gray = rgb2gray(ndarr)
            cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
            gray_list[i] = torch.from_numpy(np.ascontiguousarray(img_gray)).contiguous().float()
            cannied_list[i] = torch.from_numpy(np.ascontiguousarray(cannied)).contiguous().float()
        return cannied_list, gray_list

    def index_helper_for_testing(self,*, attack_indices_amounts: list, indices_you_want: list):
        """
        训练和测试的时候inference_tamper_index指定了混合攻击的方式
        把你想要的攻击模式送进来，这个函数会自己帮你算满足要求的最小的index
        example:
        inpainting用Edgeconnect， edgeconnect_as_inpainting: [1,4]
        color adjust用CE，simulated_contrast: [1,4]
        distortion用高斯滤波， simulated_gblur_indices: [6]
        >> ans = index_helper_for_testing(attack_indices_amounts=[self.amount_of_inpainting,
                                                                  self.amount_of_augmentation,
                                                                  self.amount_of_tampering], # 也就是[6,7,8]
                                          indices_you_want=[self.opt['edgeconnect_as_inpainting'],
                                                            self.opt['simulated_contrast'],
                                                            self.opt['simulated_gblur_indices']) # 也就是[[1,4],[1,4],[6]]
        >> ans
        >> 22
        友情提示，是有可能会触发下面的NotImplementedError的;)
        """
        for i in range(1000):
            valid=True
            for idx, cur_amount in enumerate(attack_indices_amounts):
                valid = (i % cur_amount) in indices_you_want[idx]
                if not valid:
                    break
            if valid:
                return i
        raise NotImplementedError("大神，你想找的index过于稀有了，在1000以内都找不到这样的组合，修改一下吧！")

    ####################################################################################################
    # todo: settings for beginning training
    ####################################################################################################
    def data_augmentation_on_rendered_rgb(self, modified_input, index=None, scale=1):
        if index is None:
            index = self.global_step
        index = index % self.amount_of_augmentation

        is_stronger = np.random.rand() > 0.5
        if index in self.opt['simulated_hue']:
            ## careful!
            magnitude = 0.05
            strength = np.random.rand() * (magnitude if is_stronger > 0 else -magnitude)
            modified_adjusted = F.adjust_hue(modified_input, hue_factor=0 + strength)  # 0.5 ave
        elif index in self.opt['simulated_contrast']:
            magnitude = 0.3 * scale
            strength = np.random.rand() * (magnitude if is_stronger > 0 else -magnitude)
            modified_adjusted = F.adjust_contrast(modified_input, contrast_factor=1 + strength)  # 1 ave
        elif index in self.opt['simulated_gamma']:
            ## careful!
            magnitude = 0.05
            strength = np.random.rand() * (magnitude if is_stronger > 0 else -magnitude)
            modified_adjusted = F.adjust_gamma(modified_input, gamma=1 + strength)  # 1 ave
        elif index in self.opt['simulated_saturation']:
            magnitude = 0.3 * scale
            strength = np.random.rand() * (magnitude if is_stronger > 0 else -magnitude)
            modified_adjusted = F.adjust_saturation(modified_input, saturation_factor=1 + strength)
        elif index in self.opt['simulated_brightness']:
            magnitude = 0.3 * scale
            strength = np.random.rand() * (magnitude if is_stronger > 0 else -magnitude)
            modified_adjusted = F.adjust_brightness(modified_input,
                                                    brightness_factor=1 + strength)  # 1 ave
        else:
            raise NotImplementedError(f"图像增强的index错误，请检查！index: {index}/{self.amount_of_augmentation}")
        modified_adjusted = self.clamp_with_grad(modified_adjusted)

        return modified_adjusted  # modified_input + (modified_adjusted - modified_input).detach()

    def benign_attacks(self, *, attacked_forward, quality_idx, kernel_size=None, resize_ratio=None, index=None):
        '''
            contains both simulation and real-world attack
            we restrict the lower bound of PSNR by attack, can be modified in setting.
        '''
        batch_size, height_width = attacked_forward.shape[0], attacked_forward.shape[2]
        attacked_real_jpeg = torch.empty_like(attacked_forward).cuda()
        if index is None:
            index = self.global_step
        index = index % self.amount_of_benign_attack
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7])  # 3,5,7
        if resize_ratio is None:
            resize_ratio = (int(self.random_float(0.7, 1.5) * self.width_height),
                        int(self.random_float(0.7, 1.5) * self.width_height))


        ## id of weak JPEG: 0,1,2,4,6,7
        if index in self.opt['simulated_resize_indices']:
            ## resize sometimes will also cause very low PSNR
            blurring_layer = self.resize
            processed_image, resize_ratio = blurring_layer(attacked_forward, resize_ratio=resize_ratio)
        elif index in self.opt['simulated_gblur_indices']:
            ## additional care for gaussian and median blur
            blurring_layer = self.gaussian_blur
            processed_image, kernel_size = blurring_layer(attacked_forward, kernel_size=kernel_size)
        elif index in self.opt['simulated_mblur_indices']:
            blurring_layer = self.median_blur
            processed_image, kernel_size = blurring_layer(attacked_forward, kernel=kernel_size)
        elif index in self.opt['simulated_AWGN_indices']:
            ## we dont simulate gaussian but direct add
            blurring_layer = self.identity
            processed_image = blurring_layer(attacked_forward)
        elif index in (self.opt['simulated_strong_JPEG_indices']+self.opt['simulated_weak_JPEG_indices']):
            blurring_layer = self.identity
            processed_image = blurring_layer(attacked_forward)
        else:
            raise NotImplementedError("postprocess的Index没有找到，请检查！")

        ## we regulate that jpeg attack also should not cause PSNR to be lower than 30dB
        jpeg_result = processed_image
        quality = quality_idx * 5
        for q_index in range(quality_idx,21):
            quality = int(q_index * 5)
            jpeg_layer_after_blurring = self.jpeg_simulate[q_index - 10][0] if quality < 100 else self.identity
            jpeg_result = jpeg_layer_after_blurring(processed_image)
            psnr = self.psnr(self.postprocess(jpeg_result), self.postprocess(processed_image)).item()
            if psnr>=self.opt['minimum_PSNR_caused_by_attack']:
                break
        attacked_real_jpeg_simulate = self.clamp_with_grad(jpeg_result)

        ## real-world attack
        for idx_atkimg in range(batch_size):
            grid = attacked_forward[idx_atkimg]
            realworld_attack = self.real_world_attacking_on_ndarray(grid=grid, qf_after_blur=quality,
                                                                    kernel=kernel_size, resize_ratio=resize_ratio,
                                                                    index=index)
            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        attacked_real_jpeg = attacked_real_jpeg.clone().detach()
        attacked_image = attacked_real_jpeg_simulate + (
                    attacked_real_jpeg - attacked_real_jpeg_simulate).clone().detach()

        # error_scratch = attacked_real_jpeg - attacked_forward
        # l_scratch = self.l1_loss(error_scratch, torch.zeros_like(error_scratch).cuda())
        # logs.append(('SCRATCH', l_scratch.item()))
        return attacked_image, attacked_real_jpeg_simulate, (kernel_size, quality_idx, resize_ratio)

    def benign_attacks_without_simulation(self, *, forward_image, quality_idx=None, kernel_size=None,
                                                         resize_ratio=None, index=None):
        '''
            real-world attack, whose setting should be fed.
        '''
        ## note: create tensor directly on device:
        ## torch.ones((1,1),device=a.get_device())

        if quality_idx is None:
            kernel_size = random.choice([3, 5, 7])  # 3,5,7
            resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
                            int(self.random_float(0.5, 2) * self.width_height))
            index_for_postprocessing = index #self.global_step

            quality_idx = self.get_quality_idx_by_iteration(index=index_for_postprocessing)

        batch_size, height_width = forward_image.shape[0], forward_image.shape[2]
        attacked_real_jpeg = torch.empty_like(forward_image)
        quality = int(quality_idx * 5)

        for idx_atkimg in range(batch_size):
            grid = forward_image[idx_atkimg]
            realworld_attack = self.real_world_attacking_on_ndarray(grid=grid, qf_after_blur=quality,
                                                                    index=index, kernel=kernel_size,
                                                                    resize_ratio=resize_ratio)
            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        return attacked_real_jpeg.cuda()

    def benign_attack_ndarray_auto_control(self, *, forward_image, psnr_requirement=None, get_label=False,
                                           local_compensate=True, global_compensate=False):
        '''
            real-world attack, whose setting should be fed.
        '''
        ## note: create tensor directly on device:
        ## torch.ones((1,1),device=a.get_device())
        # compare_image = forward_image.detach().cpu()
        # index_for_postprocessing = index  # self.global_step
        if psnr_requirement is None:
            psnr_requirement = self.opt['minimum_PSNR_caused_by_attack']

        # kernel_size = random.choice([3, 5, 7, 9])  # 3,5,7

        batch_size, height_width = forward_image.shape[0], forward_image.shape[2]
        attacked_real_jpeg = torch.empty_like(forward_image,device=forward_image.device)
        # quality = int(quality_idx * 5)

        for idx_atkimg in range(batch_size):
            grid = forward_image[idx_atkimg]
            max_try, tried, psnr, psnr_best = 1, 0, 0, 0
            realworld_attack = None

            index = np.random.randint(0,10000)

            kernel_list = random.sample([3, 5, 7], 3)
            resize_list = [
                (
                int(self.random_float(0.5, 2) * self.width_height), int(self.random_float(0.5, 2) * self.width_height)),
                (
                int(self.random_float(0.5, 2) * self.width_height), int(self.random_float(0.5, 2) * self.width_height)),
                (
                int(self.random_float(0.5, 2) * self.width_height), int(self.random_float(0.5, 2) * self.width_height)),
            ]
            quality_list = [
                int(self.get_quality_idx_by_iteration(index=index) * 5),
                int(self.get_quality_idx_by_iteration(index=index) * 5),
                int(self.get_quality_idx_by_iteration(index=index) * 5),
            ]

            while tried<max_try:
                realworld_candidate = self.real_world_attacking_on_ndarray(grid=grid, qf_after_blur=quality_list[tried],
                                                                        index=index, kernel=kernel_list[tried],
                                                                        resize_ratio=resize_list[tried])
                if max_try>1:
                    psnr = self.psnr(self.postprocess(forward_image[idx_atkimg:idx_atkimg + 1]), self.postprocess(realworld_candidate)).item()
                    if psnr>psnr_best:
                        realworld_attack = realworld_candidate
                        psnr_best = psnr
                    if psnr>psnr_requirement:
                        break
                else:
                    realworld_attack = realworld_candidate

                tried += 1

            if psnr<self.opt['minimum_PSNR_caused_by_attack']:
                beta = np.random.rand()
                realworld_attack = beta * forward_image[idx_atkimg:idx_atkimg + 1] + (1 - beta) * realworld_attack

            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        # ## eliminate too poor images
        # if not (psnr < 1 or psnr > psnr_requirement):  # , f"PSNR {psnr} is not allowed! {self.global_step}"
        #     mixed_conpensate = None
        #     ### blurring attack is tough, special care on that
        #     for alpha in [i * 0.1 for i in range(1, 10)]:
        #         mixed_conpensate = alpha * forward_image[idx_atkimg:idx_atkimg + 1] + (1 - alpha) * realworld_attack
        #         psnr = self.psnr(self.postprocess(mixed_conpensate),
        #                          self.postprocess(forward_image[idx_atkimg:idx_atkimg + 1])).item()
        #
        #         if psnr > psnr_requirement:
        #             break
        #     realworld_attack = mixed_conpensate

        psnr_distort, mse_distort = self.psnr.with_mse(self.postprocess(attacked_real_jpeg), self.postprocess(forward_image))
        psnr_distort = psnr_distort.item()
        # psnr_standard = self.opt['minimum_PSNR_caused_by_attack'] \
        #                 + np.random.rand()*(self.opt['max_psnr']-self.opt['minimum_PSNR_caused_by_attack'])
        # psnr_tolerant = 0.5
        ## global compensate (distort) to make dense probabilities
        # condition = psnr_distort<self.opt['max_psnr'] and \
        #             ((global_compensate and np.random.rand() > 0.5) or psnr_distort<self.opt['minimum_PSNR_caused_by_attack'])
        # attack_backup = attacked_real_jpeg
        # while condition: # and psnr_distort < self.opt['minimum_PSNR_caused_by_attack']:
        #     beta = np.random.rand()
        #     attack_backup = beta * forward_image + (1 - beta) * attacked_real_jpeg
        #     psnr_distort = self.psnr(self.postprocess(attack_backup), self.postprocess(forward_image)).item()
        #     condition = not (psnr_distort>self.opt['minimum_PSNR_caused_by_attack'] and psnr_distort<self.opt['max_psnr'])
        # attacked_real_jpeg = attack_backup

        ## color adjustment
        # if self.opt['do_augment'] and np.random.rand() > 0.5:
        #     attacked_adjusted = self.data_augmentation_on_rendered_rgb(attacked_real_jpeg,
        #                                                                 index=np.random.randint(0, 10000),
        #                                                                 scale=1)
        #
        #
        #     # psnr = self.psnr(self.postprocess(attacked_adjusted), self.postprocess(forward_image)).item()
        #     ## global compensate (color) to make dense probailities
        #     if global_compensate and np.random.rand() > 0.5: # and psnr<self.opt['minimum_PSNR_caused_by_attack']:
        #         beta = np.random.rand()
        #         attacked_real_jpeg = beta * attacked_real_jpeg + (1 - beta) * attacked_adjusted
        #     else:
        #         attacked_real_jpeg = attacked_adjusted

        ## calculate psnr label
        psnr_label, mix_num, psnr_avg, mse_avg = self.calculate_psnr_label(real=forward_image, fake=attacked_real_jpeg,
                                               psnr_max=self.opt['max_psnr'], psnr_min=psnr_requirement)

        psnr_label_mean = torch.mean(psnr_label).item()
        REAL_PSNR = self.opt['minimum_PSNR_caused_by_attack'] + \
                    (self.opt['max_psnr'] - self.opt['minimum_PSNR_caused_by_attack']) * psnr_label_mean
        PSNR_DIFF = REAL_PSNR - psnr_distort
        mse_diff = mse_distort-mse_avg
        if get_label:
            return attacked_real_jpeg, psnr_label, psnr_distort
        else:
            return attacked_real_jpeg, psnr_distort

    def calculate_psnr_label(self, *, real, fake, psnr_max, psnr_min):
        batch_size = real.shape[0]
        psnr_label = torch.zeros((batch_size, 1), device=real.device)
        mix_num, psnr_avg,mse_avg = 0, 0, 0
        for idx_atkimg in range(batch_size):
            psnr, mse = self.psnr.with_mse(self.postprocess(fake[idx_atkimg:idx_atkimg + 1]),
                             self.postprocess(real[idx_atkimg:idx_atkimg + 1]))
            psnr_avg += psnr.item()/batch_size
            mse_avg += mse.item() / batch_size
            psnr_label[idx_atkimg:idx_atkimg + 1] = max(0., min(1., (psnr - psnr_min) / (psnr_max - psnr_min)))
            if psnr_label[idx_atkimg:idx_atkimg + 1]==0.:
                mix_num += 1
        return psnr_label, mix_num, psnr_avg, mse_avg

    def to_jpeg(self, *, forward_image):
        batch_size, height_width = forward_image.shape[0], forward_image.shape[2]
        attacked_real_jpeg = torch.empty_like(forward_image)

        for idx_atkimg in range(batch_size):
            grid = forward_image[idx_atkimg]
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to('cpu', torch.uint8).numpy()

            _, realworld_attack = cv2.imencode('.jpeg', ndarr,
                                               (int(cv2.IMWRITE_JPEG_QUALITY), 100))
            realworld_attack = cv2.imdecode(realworld_attack, cv2.IMREAD_UNCHANGED)

            realworld_attack = realworld_attack.astype(np.float32) / 255.
            realworld_attack = torch.from_numpy(
                np.ascontiguousarray(np.transpose(realworld_attack, (2, 0, 1)))).contiguous().float()
            realworld_attack = realworld_attack.unsqueeze(0)

            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        return attacked_real_jpeg.cuda()

    def real_world_attacking_on_ndarray(self, *,  grid, qf_after_blur, kernel, resize_ratio, index=None):
        # batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        '''
            real-world attack (CV2)
            ref: https://www.geeksforgeeks.org/python-opencv-imencode-function/
            imencode will produce the exact result compared to that by imwrite, but much quicker
        '''
        if index is None:
            index = self.global_step
        index = index % self.amount_of_benign_attack

        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to('cpu', torch.uint8).numpy()
        if index in self.opt['simulated_resize_indices']:
            # grid = self.resize(grid.unsqueeze(0))[0]
            # newH, newW = int((0.7+0.6*np.random.rand())*self.width_height), int((0.7+0.6*np.random.rand())*self.width_height)
            newH, newW = resize_ratio
            realworld_attack = cv2.resize(np.copy(ndarr), (newH,newW),
                                          interpolation=cv2.INTER_LINEAR)
            realworld_attack = cv2.resize(np.copy(realworld_attack), (self.width_height, self.width_height),
                                interpolation=cv2.INTER_LINEAR)
        elif index in self.opt['simulated_gblur_indices']:
            # kernel_list = [5]
            # kernel = random.choice(kernel_list)
            realworld_attack = cv2.GaussianBlur(ndarr, (kernel, kernel), 0) if kernel > 0 else ndarr
        elif index in self.opt['simulated_mblur_indices']:
            # kernel_list = [5]
            # kernel = random.choice(kernel_list)
            realworld_attack = cv2.medianBlur(ndarr, kernel) if kernel > 0 else ndarr

        elif index in self.opt['simulated_AWGN_indices']:
            mean, sigma = 0, 1.0
            gauss = np.random.normal(mean, sigma, (self.width_height, self.width_height, 3))
            # 给图片添加高斯噪声
            realworld_attack = ndarr + gauss
        elif index in (self.opt['simulated_strong_JPEG_indices']+self.opt['simulated_weak_JPEG_indices']):
            realworld_attack = ndarr
        else:
            raise NotImplementedError("postprocess的Index没有找到，请检查！")

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
        realworld_attack = realworld_attack.unsqueeze(0)
        return realworld_attack.cuda()

    ### todo: trivial stuffs
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def random_float(self, min, max):
        """
        Return a random number
        :param min:
        :param max:
        :return:
        """
        return np.random.rand() * (max - min) + min

    def get_paths_from_images(self, path):
        '''
            get image path list from image folder
        '''
        # assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
        if path is None:
            return None, None

        images_dict = {}
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if self.is_image_file(fname):
                    img_path = os.path.join(dirpath, fname)
                    # images.append((path, dirpath[len(path) + 1:], fname))
                    images_dict[fname] = img_path
        assert images_dict, '{:s} has no valid image file'.format(path)

        return images_dict

    def print_individual_image(self, cropped_GT, name):
        for image_no in range(cropped_GT.shape[0]):
            camera_ready = cropped_GT[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         name, nrow=1, padding=0, normalize=False)


    def load_image(self, path, readimg=False, grayscale=False, require_canny=False):
        import data.util as util
        GT_path = path

        img_GT = util.read_img(GT_path)

        # change color space if necessary
        img_GT = util.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]
        if grayscale:
            img_GT = rgb2gray(img_GT)

        img_GT = cv2.resize(copy.deepcopy(img_GT), (self.width_height, self.width_height),
                            interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if not grayscale:
            image = img_GT[:, :, [2, 1, 0]]
            image = torch.from_numpy(
                np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float()
        else:
            image = torch.from_numpy(
                np.ascontiguousarray(img_GT)).float()

        if require_canny and not grayscale:
            img_gray = rgb2gray(img_GT)
            sigma = 2  # random.randint(1, 4)
            cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
            canny_image = torch.from_numpy(
                np.ascontiguousarray(cannied)).float()
            return image.cuda().unsqueeze(0), canny_image.cuda().unsqueeze(0).unsqueeze(0)
        else:
            return image.cuda().unsqueeze(0)

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
        load_net = torch.load(load_path, map_location='cpu')
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    # def save_training_state(self, epoch, iter_step, model_path, network_list):
    #     '''Saves training state during training, which will be used for resuming'''
    #     state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
    #     if 'localizer' in network_list:
    #         state['optimizer_localizer'] = self.optimizer_localizer.state_dict()
    #     if 'discriminator_mask' in network_list:
    #         state['optimizer_discriminator_mask'] = self.optimizer_discriminator_mask.state_dict()
    #
    #     if 'discriminator' in network_list:
    #         state['optimizer_discriminator'] = self.optimizer_discriminator.state_dict()
    #
    #     if 'netG' in network_list:
    #         state['optimizer_G'] = self.optimizer_G.state_dict()
    #         state['clock'] = self.netG.module.clock
    #
    #     if 'generator' in network_list:
    #         state['optimizer_generator'] = self.optimizer_generator.state_dict()
    #     if 'KD_JPEG' in network_list:
    #         state['optimizer_KD_JPEG'] = self.optimizer_KD_JPEG.state_dict()
    #     if 'qf_predict' in network_list:
    #         state['optimizer_qf_predict'] = self.optimizer_qf_predict.state_dict()
    #     # for s in self.schedulers:
    #     #     state['schedulers'].append(s.state_dict())
    #     # for o in self.optimizers:
    #     #     state['optimizers'].append(o.state_dict())
    #
    #     save_filename = '{}.state'.format(iter_step)
    #     save_path = os.path.join(model_path , save_filename)
    #     print("State saved to: {}".format(save_path))
    #     torch.save(state, save_path)

    # def resume_training(self, state_path, network_list):
    #     resume_state = torch.load(state_path)
    #     if 'clock' in resume_state and 'netG' in network_list:
    #         self.localizer.module.clock = resume_state['clock']
    #     ##  Resume the optimizers and schedulers for training
    #     if 'optimizer_G' in resume_state and 'netG' in network_list:
    #         self.optimizer_G.load_state_dict(resume_state['optimizer_G'])
    #     if 'optimizer_localizer' in resume_state and 'localizer' in network_list:
    #         self.optimizer_localizer.load_state_dict(resume_state['optimizer_localizer'])
    #     if 'optimizer_discriminator_mask' in resume_state and 'discriminator_mask' in network_list:
    #         self.optimizer_discriminator_mask.load_state_dict(resume_state['optimizer_discriminator_mask'])
    #     if 'optimizer_discriminator' in resume_state and 'discriminator' in network_list:
    #         self.optimizer_discriminator.load_state_dict(resume_state['optimizer_discriminator'])
    #     if 'optimizer_qf_predict' in resume_state and 'qf_predict' in network_list:
    #         self.optimizer_qf_predict.load_state_dict(resume_state['optimizer_qf_predict'])
    #     if 'optimizer_generator' in resume_state and 'generator' in network_list:
    #         self.optimizer_generator.load_state_dict(resume_state['optimizer_generator'])
    #     if 'optimizer_KD_JPEG' in resume_state and 'KD_JPEG' in network_list:
    #         self.optimizer_KD_JPEG.load_state_dict(resume_state['optimizer_KD_JPEG'])

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
        minVertex, maxVertex = 1, int(20*percent_range[1])
        minLength, maxLength = 8, int(im_size[0] * 0.8 * percent_range[1])
        minBrushWidth, maxBrushWidth = 8, int(im_size[0] * 0.8 * percent_range[1])
        mask = np.zeros((im_size[0], im_size[1]), dtype=np.float32)
        lower_bound_percent = percent_range[0] + (percent_range[1] - percent_range[0]) * np.random.rand()

        while True:
            mask = mask + self.np_free_form_mask(mask, minVertex, maxVertex, minLength, maxLength, minBrushWidth,
                                                 maxBrushWidth,
                                                 maxAngle, im_size[0],
                                                 im_size[1])
            mask = np.minimum(mask, 1.0)
            percent = np.mean(mask)
            # print(percent, lower_bound_percent)
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


    def metric_single_image(self, get_auc, thresh, test_images):
        predicted_binary, gt_image = test_images
        # predicted_binary = self.tensor_to_image(predict_image)
        # gt_image = self.tensor_to_image(gt_image)
        if get_auc:
            AUC = getAUC(predicted_binary / 255, gt_image / 255)
        ret, predicted_binary = cv2.threshold(predicted_binary, int(255 * thresh), 255, cv2.THRESH_BINARY)
        ret, gt_image = cv2.threshold(gt_image, int(255 * thresh), 255, cv2.THRESH_BINARY)
        if get_auc:
            IoU = getIOU(predicted_binary / 255, gt_image / 255)
        # print(predicted_binary.shape)

        [TN, TP, FN, FP] = getLabels(predicted_binary, gt_image)
        # print("{} {} {} {}".format(TN,TP,FN,FP))
        F1 = getF1(TP, FP, FN)
        RECALL = getTPR(TP, FN)
        return (F1, RECALL, AUC, IoU) if get_auc else (F1, RECALL)

    def F1score(self, predict_image, gt_image, thresh=0.2, get_auc=False):
        # gt_image = cv2.imread(src_image, 0)
        # predict_image = cv2.imread(dst_image, 0)
        # ret, gt_image = cv2.threshold(gt_image[0], int(255 * thresh), 255, cv2.THRESH_BINARY)
        # ret, predicted_binary = cv2.threshold(predict_image[0], int(255*thresh), 255, cv2.THRESH_BINARY)

        predict_image = predict_image.permute(0, 2, 3, 1).detach().cpu().numpy()
        predict_image = np.clip(predict_image, 0, 1).flatten()
        # predict_image = self.tensor_to_image_batch(predict_image)
        # gt_image = self.tensor_to_image_batch(gt_image)
        gt_image = gt_image.permute(0, 2, 3, 1).detach().cpu().numpy()
        gt_image = np.clip(gt_image, 0, 1).flatten()
        if get_auc:
            avg_AUC = getAUC(predict_image, gt_image)
        predicted_binary = np.where(predict_image < 0.5, 0, 1)
        avg_F1, avg_RECALL = getFScore(predicted_binary, gt_image)
        if get_auc:
            avg_IoU = getIOU(predicted_binary, gt_image)
        return (avg_F1, avg_RECALL, avg_AUC, avg_IoU) if get_auc else (avg_F1, avg_RECALL)
        # return (0, 0, 0, 0) if get_auc else (0, 0)

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

def getFScore(pre, gt):
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(gt, pre, average='binary')
    return f1, rec


def getAUC(pre, gt):
    # 输入都是0-1区间内的 mask
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(gt.flatten(), pre.flatten())
    except ValueError:
        return 0.0
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model_path = './MantraNetv4.pt'
    localizer = nn.DataParallel(pre_trained_model(model_path)).cuda()
    image_files = './data/test2/'
    os.makedirs('./data/test', exist_ok=True)
    import glob
    import imageio
    rgb_files = glob.glob(image_files+'*.png')
    for rgb_file in rgb_files:
        image = imageio.imread(rgb_file)
        image = image / 255
        image = np.expand_dims(image, axis=3)
        image = torch.Tensor(image).permute(3, 2, 0, 1).cuda()
        out = localizer(image)
        out = torch.sigmoid(out)
        out = out.permute(2, 3, 0, 1).detach().squeeze(2).cpu().numpy()
        out_image = (out * 255).astype(np.uint8)
        bname = os.path.basename(rgb_file)
        imageio.imwrite(os.path.join('./data/test', bname), out_image)
    # gt = np.random.choice(2, [32, 256, 256, 1])
    # pre = np.random.choice(2, [32, 256, 256, 1])
    # f1, recall = getFScore(pre.flatten(), gt.flatten())
    # exit(0)
    # iou = getIOU(pre, gt)
    # print(iou)
    # from sklearn.metrics import f1_score
    # api_f1 = f1_score(gt.flatten(), pre.flatten())
    # print(api_f1)
    # auc = getAUC(pre.flatten(), gt.flatten())
    # # auc = getAUC(pre, gt)
    # print(auc)
    # [TN, TP, FN, FP] = getLabels(pre*255, gt*255)
    # # print("{} {} {} {}".format(TN,TP,FN,FP))
    # our_f1 = getF1(TP, FP, FN)
    # print(our_f1)

# if 'localizer' in self.network_list:
#     ####################################################################################################
#     # todo: (Deprecated!!!!) Image Manipulation Detection Network (Downstream task) will be loaded
#     # todo: mantranet: localizer mvssnet: netG resfcn: discriminator
#     ####################################################################################################
#     print("Building MantraNet...........please wait...")
#     self.localizer = pre_trained_model(weight_path='./MantraNetv4.pt').cuda()
#     self.localizer = DistributedDataParallel(self.localizer, device_ids=[torch.cuda.current_device()],
#                                              find_unused_parameters=True)
#
#     print("Building MVSS...........please wait...")
#     model_path = './MVSS/ckpt/mvssnet_casia.pt'
#     self.netG = get_mvss(backbone='resnet50',
#                          pretrained_base=True,
#                          nclass=1,
#                          sobel=True,
#                          constrain=True,
#                          n_input=3,
#                          ).cuda()
#     checkpoint = torch.load(model_path, map_location='cpu')
#     self.netG.load_state_dict(checkpoint, strict=True)
#     self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],
#                                         find_unused_parameters=True)
#     print("Building ResFCN...........please wait...")
#     self.discriminator_mask = ResFCN().cuda()
#     checkpoint = torch.load('./resfcn_coco_1013.pth', map_location='cpu')
#     self.discriminator_mask.load_state_dict(checkpoint, strict=True)
#     self.discriminator_mask = DistributedDataParallel(self.discriminator_mask,
#                                                       device_ids=[torch.cuda.current_device()],
#                                                       find_unused_parameters=True)
#     ## AS for ResFCN, we found no checkpoint in the official repo currently
#
#     self.scaler_localizer = torch.cuda.amp.GradScaler()
#     self.scaler_G = torch.cuda.amp.GradScaler()
#     self.scaler_discriminator_mask = torch.cuda.amp.GradScaler()