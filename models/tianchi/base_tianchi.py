import copy
# from .networks import SPADE_UNet
import os
from collections import OrderedDict

import numpy as np
import torch
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


class BaseTianchi():
    def __init__(self, opt,  args):
        ### todo: options
        self.opt = opt
        # self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

        self.rank = torch.distributed.get_rank()
        self.opt = opt
        self.args = args
        # self.train_opt = opt['train']
        # self.test_opt = opt['test']


        ### todo: constants
        self.global_step = 0
        self.width_height = opt['datasets']['train']['GT_size']

        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

        ### todo: network definitions
        self.network_list = []
        self.real_H, self.real_H_path, self.previous_images = None, None, None
        self.previous_previous_images, self.previous_previous_canny = None, None
        self.previous_protected, self.previous_canny = None, None


        self.out_space_storage = "/groupshare/tianchi2023_results"
        self.task_name = args.task_name
        self.create_folders_for_the_experiment()

        ### todo: losses and attack layers
        self.psup = nn.PixelShuffle(upscale_factor=2).cuda()
        self.psdown = nn.PixelUnshuffle(downscale_factor=2).cuda()
        self.tanh = nn.Tanh().cuda()
        self.psnr = PSNR(255.0).cuda()
        # self.lpips_vgg = lpips.LPIPS(net="vgg").cuda()
        # self.exclusion_loss = ExclusionLoss().type(torch.cuda.FloatTensor).cuda()

        self.crop = Crop().cuda()
        self.dropout = Dropout().cuda()
        self.gaussian = Gaussian().cuda()
        self.salt_pepper = SaltPepper(prob=0.01).cuda()
        self.gaussian_blur = GaussianBlur(opt=self.opt).cuda()
        self.median_blur = MiddleBlur(opt=self.opt).cuda()
        self.resize = Resize(opt=self.opt).cuda()
        self.identity = Identity().cuda()

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
        self.CE_loss = nn.CrossEntropyLoss().cuda()




    ### todo: Abstract Methods


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
        create_folder(self.out_space_storage)
        create_folder(self.out_space_storage + "/models")
        create_folder(self.out_space_storage + "/images")
        create_folder(self.out_space_storage + "/observe/")
        create_folder(self.out_space_storage + "/observe/" + self.task_name)
        create_folder(self.out_space_storage + "/models/" + self.task_name)
        create_folder(self.out_space_storage + "/images/" + self.task_name)
        create_folder(self.out_space_storage + "/submission/")
        create_folder(self.out_space_storage + "/submission/" + self.task_name)

    def load_model_wrapper(self,*,folder_name,model_name,network, network_name, strict=True):
        load_detector_storage = self.opt[folder_name]
        model_path = str(self.opt[model_name])  # last time: 10999
        load_models = self.opt[model_name] > 0
        if load_models:
            print(f"loading models: {network_name}")
            pretrain = load_detector_storage + model_path
            self.reload(pretrain, network, strict=strict)

    def reload(self, pretrain, network, strict=True):
        load_path_G = pretrain
        if load_path_G is not None:
            print('Loading models for class [{:s}] ...'.format(load_path_G))
            if os.path.exists(load_path_G):
                self.load_network(load_path_G, network, strict=strict)
            else:
                print('Did not find models for class [{:s}] ...'.format(load_path_G))

    def define_CATNET(self, NUM_CLASSES=1, num_bayar=0, load_pretrained_cat_model=False):
        print("using CATnet")
        from detection_methods.CATNet.model import get_model
        model = get_model(NUM_CLASSES, use_SRM=num_bayar, load_pretrained_cat_model=load_pretrained_cat_model).cuda()
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

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

    def save_network(self, *, pretrain, network):
        # save_dir = '../experiments/pretrained_models/'
        # if model_path == None:
        #     model_path = self.opt['path']['models']
        # if save_dir is None:
        #     save_filename = '{}_{}_{}.pth'.format(accuracy, iter_label, network_label)
        #     save_path = os.path.join(model_path, save_filename)
        # else:
        #     save_filename = '{}_latest.pth'.format(network_label)
        #     save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        print("Model saved to: {}".format(pretrain))
        torch.save(state_dict, pretrain)

    def load_network(self, load_path, network, strict=False):
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


if __name__ == '__main__':
    pass