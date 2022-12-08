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
from models.IFA.base_IFA import base_IFA


class RR_IFA(base_IFA):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """
            this file is mode 0

        """
        super(RR_IFA, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants


    def network_definitions(self):
        ### mode=8: InvISP to bi-directionally convert RGB and RAW,
        self.network_list = ['qf_predict_network']
        self.save_network_list = ['qf_predict_network']
        self.training_network_list = ['qf_predict_network']

        ### network

        self.generator = self.define_IFA_net()
        # self.qf_predict_network = self.define_UNet_as_ISP()
        self.load_model_wrapper(folder_name='predictor_folder', model_name='load_predictor_models',
                                network_lists=["qf_predict_network"], strict=True)


    def define_IFA_net(self):
        from models.IFA.tres_model import Net
        self.qf_predict_network = Net(num_embeddings=10).cuda()
        self.qf_predict_network = DistributedDataParallel(self.qf_predict_network,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)

    def predict_IFA_with_reference(self, step=None):
        ####################  Image Manipulation Detection Network (Downstream task)  ##################
        ### mantranet: localizer mvssnet: netG resfcn: discriminator

        #### SYMBOL FOR NOTIFYING THE OUTER VAL LOADER #######
        did_val = False
        if step is not None:
            self.global_step = step

        logs, debug_logs = {}, []
        # self.real_H = self.clamp_with_grad(self.real_H)
        lr = self.get_current_learning_rate()
        logs['lr'] = lr


        # self.real_H, self.canny_image

        with torch.enable_grad():

            self.qf_predict_network.train()
            gt_rgb = self.real_H
            masks_GT = self.canny_image

            label = torch.mean(masks_GT,dim=[1,2,3]).float()

            #######
            predictionQA, feat, quan_loss = self.qf_predict_network(self.real_H)


            l_class = self.bce_with_logit_loss(predictionQA, label)

            l_sum = quan_loss+l_class
            logs['CE'] = l_class.item()
            logs['Quan'] = quan_loss.item()

            l_sum.backward()

            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()


        # if (self.global_step % 1000 == 3 or self.global_step <= 10):
        #     images = stitch_images(
        #         self.postprocess(gt_rgb),
        #         self.postprocess(raw_invisp),
        #         self.postprocess(modified_input),
        #         self.postprocess(10 * torch.abs(modified_input - gt_rgb)),
        #         self.postprocess(pred_resfcn),
        #         self.postprocess(masks_GT),
        #         # self.postprocess(rendered_rgb_invISP),
        #         img_per_row=1
        #     )
        #
        #     name = f"{self.out_space_storage}/isp_images/{self.task_name}/{str(self.global_step).zfill(5)}" \
        #            f"_{str(self.rank)}.png"
        #     images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (self.opt['model_save_period']-1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)

        self.global_step = self.global_step + 1

        return logs, debug_logs, did_val
