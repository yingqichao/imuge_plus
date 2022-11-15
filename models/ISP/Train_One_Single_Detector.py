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


class Train_One_Single_Detector(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """

            this file is mode 6

        """
        super(Train_One_Single_Detector, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        self.network_list = ['discriminator_mask']
        self.save_network_list = self.network_list
        self.training_network_list = self.network_list

        ## define detector, 这个可以根据需要去做修改
        print(f"using {self.opt['finetune_detector_name']} as discriminator_mask.")
        self.discriminator_mask = self.define_detector(opt_name='finetune_detector_name')
        self.load_model_wrapper(folder_name='detector_folder', model_name='load_discriminator_models',
                                network_lists=['discriminator_mask'])

        ## inpainting model
        self.define_inpainting_edgeconnect()
        self.define_inpainting_ZITS()
        self.define_inpainting_lama()

    def train_resfcn(self, step=None):
        self.discriminator_mask.train()

        logs, debug_logs = {}, []

        self.label = self.clamp_with_grad(self.label)
        batch_size, num_channels, height_width, _ = self.label.shape
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        if not (self.previous_images is None or self.previous_previous_images is None):

            with torch.enable_grad():  # cuda.amp.autocast():

                attacked_image, attacked_adjusted, attacked_forward, masks, masks_GT = self.standard_attack_layer(
                    modified_input=self.label, gt_rgb=self.label, logs=logs)

                ############    Image Manipulation Detection Network (Downstream task)   ###############################
                pred_resfcn, CE_resfcn = self.detector_predict(model=self.discriminator_mask,
                                                               attacked_image=attacked_image.detach().contiguous(),
                                                               opt_name='finetune_detector_name',
                                                               masks_GT=masks_GT)

                logs['CE'] = CE_resfcn.item()

                self.optimizer_discriminator_mask.zero_grad()
                CE_resfcn.backward()
                if self.train_opt['gradient_clipping']:
                    nn.utils.clip_grad_norm_(self.discriminator_mask.parameters(), 1)
                self.optimizer_discriminator_mask.step()
                self.optimizer_discriminator_mask.zero_grad()

                if self.global_step % 1000 == 3 or self.global_step <= 10:
                    images = stitch_images(
                        self.postprocess(self.label),
                        self.postprocess(attacked_image),
                        self.postprocess(attacked_adjusted),
                        self.postprocess(attacked_forward),
                        self.postprocess(pred_resfcn),
                        self.postprocess(masks_GT),
                        img_per_row=1
                    )

                    name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                           f"_0_ {str(self.rank)}.png"
                    print('\nsaving sample ' + name)
                    images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.network_list)
        if self.label is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.label
            self.previous_protected = self.label
        self.global_step = self.global_step + 1

        # print(logs)
        # print(debug_logs)
        return logs, debug_logs, False
