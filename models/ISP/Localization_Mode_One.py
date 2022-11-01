import math
import os

import cv2
import torch
import torch.nn as nn
# from cycleisp_models.cycleisp import Raw2Rgb
# from MVSS.models.mvssnet import get_mvss
# from MVSS.models.resfcn import ResFCN
# from data.pipeline import pipeline_tensor2image
# import matlab.engine
import torch.nn.functional as Functional
import torchvision.transforms.functional as F
from torch.nn.parallel import DistributedDataParallel
from data.pipeline import isp_tensor2image
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
from MantraNet.mantranet import pre_trained_model
# from .networks import SPADE_UNet
from lama_models.HWMNet import HWMNet
from lama_models.my_own_elastic_dtcwt import my_own_elastic
from models.base_model import BaseModel
# from .invertible_net import Inveritible_Decolorization_PAMI
# from models.networks import UNetDiscriminator
# from loss import PerceptualLoss, StyleLoss
# from .networks import SPADE_UNet
# from lama_models.HWMNet import HWMNet
# import contextual_loss as cl
# import contextual_loss.functional as F
from models.invertible_net import Inveritible_Decolorization_PAMI
from models.networks import UNetDiscriminator
from noise_layers import *
from utils import stitch_images
from models.ISP.Modified_invISP import Modified_invISP


class Localization_Mode_One(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """

            this file is mode 1
            this model does not implement abstract method

        """
        super(Localization_Mode_One, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        """

            currently the same as mode 0. feel free to change.

        """

        self.network_list = self.default_ISP_networks + self.default_RAW_to_RAW_networks + self.default_detection_networks_for_training
        self.save_network_list = []
        self.training_network_list = []

        ### ISP networks
        self.define_ISP_network_training()
        self.load_model_wrapper(folder_name='ISP_folder', model_name='load_ISP_models',
                                network_lists=self.default_ISP_networks)
        ### RAW2RAW network
        self.define_RAW2RAW_network()
        self.load_model_wrapper(folder_name='protection_folder', model_name='load_RAW_models',
                                network_lists=self.default_RAW_to_RAW_networks)
        ### detector networks: 这个可能需要改一下，要看到底加载哪个模型，现在是MPF
        print("using my_own_elastic as discriminator_mask.")
        self.discriminator_mask = self.define_my_own_elastic_as_detector()
        self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
                                network_lists=['discriminator_mask'])