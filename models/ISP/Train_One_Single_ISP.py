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


class Train_One_Single_ISP(Modified_invISP):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """

            this file is mode 5

        """
        super(Train_One_Single_ISP, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants

    def network_definitions(self):
        self.network_list = ['localizer']
        self.save_network_list = self.network_list
        self.training_network_list = self.network_list

        ## define localizer
        self.localizer = self.define_restormer() #self.define_restormer()
        self.load_model_wrapper(folder_name='Restormer_folder', model_name='load_Restormer_models',
                                network_lists=['localizer'])


    def train_ISP_using_rstormer(self, step=None):

        #### SYMBOL FOR NOTIFYING THE OUTER VAL LOADER #######
        did_val = False
        if step is not None:
            self.global_step = step

        logs, debug_logs = {}, []
        # self.real_H = self.clamp_with_grad(self.real_H)
        lr = self.get_current_learning_rate()
        logs['lr'] = lr


        ### camera_white_balance SIZE (B,3)
        camera_white_balance = self.camera_white_balance
        file_name = self.file_name
        ### bayer_pattern sized (B,1) ranging from [0,3]
        bayer_pattern = self.bayer_pattern

        input_raw_one_dim = self.real_H
        gt_rgb = self.label
        input_raw = self.visualize_raw(input_raw_one_dim, bayer_pattern=bayer_pattern,
                                       white_balance=camera_white_balance,
                                       eval=not self.opt['train_isp_networks'])

        batch_size, num_channels, height_width, _ = input_raw.shape
        # input_raw = self.clamp_with_grad(input_raw)

        with torch.enable_grad():

            self.localizer.train()
            ####################   Image ISP training   ##############################
            ### we first train several nn-based ISP networks BEFORE TRAINING THE PIPELINE

            ####### UNetDiscriminator ##############
            modified_input_qf_predict, CYCLE_loss = self.ISP_image_generation_general(
                network=self.localizer,
                input_raw=input_raw.detach().contiguous(),
                target=gt_rgb)

            modified_input_qf_predict_detach = self.clamp_with_grad(modified_input_qf_predict.detach())
            CYCLE_PSNR = self.psnr(self.postprocess(modified_input_qf_predict_detach),
                                   self.postprocess(gt_rgb)).item()
            logs['CYCLE_PSNR'] = CYCLE_PSNR
            logs['CYCLE_L1'] = CYCLE_loss.item()

            CYCLE_loss.backward()

            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.localizer.parameters(), 1)
            self.optimizer_localizer.step()
            self.optimizer_localizer.zero_grad()

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(input_raw),
                self.postprocess(modified_input_qf_predict_detach),
                self.postprocess(gt_rgb),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/isp_images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            print(f'Bayer: {bayer_pattern}. Saving sample {name}')
            images.save(name)

        ######## Finally ####################
        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)

        self.global_step = self.global_step + 1

        # print(logs)
        # print(debug_logs)
        return logs, debug_logs, did_val

