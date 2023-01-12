import os

import torch
import torch.nn as nn
# from cycleisp_models.cycleisp import Raw2Rgb
# from MVSS.models.mvssnet import get_mvss
# from MVSS.models.resfcn import ResFCN
# from data.pipeline import pipeline_tensor2image
# import matlab.engine
from torch.nn.parallel import DistributedDataParallel
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
# from .networks import SPADE_UNet
# from .invertible_net import Inveritible_Decolorization_PAMI
# from models.networks import UNetDiscriminator
# from loss import PerceptualLoss, StyleLoss
# from .networks import SPADE_UNet
# from lama_models.HWMNet import HWMNet
# import contextual_loss as cl
# import contextual_loss.functional as F
from noise_layers import *
from utils import stitch_images
from network.common import MLP
from models.IFA.base_IFA import base_IFA


class IFA_baseline(base_IFA):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """
            this file is mode 3
        """
        super(IFA_baseline, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants
        self.history_accuracy = 0.1


    def network_definitions(self):
        self.network_list = ['qf_predict_network']
        self.save_network_list = ['qf_predict_network']
        # self.training_network_list = ['qf_predict_network'] unused

        ### todo: network

        self.network_definitions_binary_classification()


        if self.opt['load_predictor_models'] is not None:
            # self.load_model_wrapper(folder_name='predictor_folder', model_name='load_predictor_models',
            #                         network_lists=["qf_predict_network"], strict=False)
            load_detector_storage = self.opt['predictor_folder']
            model_path = str(self.opt['load_predictor_models'])  # last time: 10999

            print(f"loading models: {self.network_list}")
            pretrain = load_detector_storage + model_path
            load_path_G = pretrain+"_qf_predict.pth"

            print('Loading model for class [{:s}] ...'.format(load_path_G))
            if os.path.exists(load_path_G):
                self.load_network(load_path_G, self.qf_predict_network, strict=False)

                # pkl_path = f'{self.out_space_storage}/model/{self.task_name}/{model_path}_qf_predict.pkl'
                # with open(pkl_path, 'rb') as f:
                #     data = pickle.load(f)
                #     self.qf_predict_network.embedding = data['embedding']
                #     print("Pickle loaded to: {}".format(pkl_path))

            else:
                print('Did not find model for class [{:s}] ...'.format(load_path_G))

        # self.generator = self.define_IFA_net()
        # # self.qf_predict_network = self.define_UNet_as_ISP()
        # self.load_model_wrapper(folder_name='predictor_folder', model_name='load_predictor_models',
        #                         network_lists=["qf_predict_network"], strict=True)

        # ### todo: recovery network
        # if self.consider_mask_prediction:
        #     self.network_list += ['generator']
        #     self.save_network_list += ['generator']
        #     self.training_network_list += ['generator']
        #     from models.networks import UNetDiscriminator
        #     self.generator = UNetDiscriminator(in_channels=3, out_channels=1, use_SRM=False, use_sigmoid=False,
        #                                        residual_blocks=8,dim=32, output_middle_feature=True).cuda()
        #     self.generator = DistributedDataParallel(self.generator,
        #                                                     device_ids=[torch.cuda.current_device()],
        #                                                     find_unused_parameters=True)
        #     if self.opt['load_predictor_models'] > 0:
        #         self.load_model_wrapper(folder_name='predictor_folder', model_name='load_predictor_models',
        #                                 network_lists=["generator"], strict=True)
        #
        # ### todo: inpainting model
        # self.define_inpainting_edgeconnect()
        # self.define_inpainting_ZITS()
        # self.define_inpainting_lama()

        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1).cuda()

    def network_definitions_binary_classification(self):
        self.IFA_bc_label = None
        ### todo: network
        # predictor is mlp
        self.detector = self.define_detector(opt_name='which_model_for_detector')
        feature_dim = self.opt['feature_dim']
        # 特征维度先假定是1000吧
        self.qf_predict_network = self.define_MLP(feature_dim=feature_dim, class_dim=1)

        if self.opt['load_detector_models'] is None:
            # to be continued
            pass

        # self.restore_restormer = self.define_restormer()
        # self.restore_unet = self.define_ddpm_unet_network()
        # self.restore_invisp = self.define_invISP(block_num=[4, 4, 4])
        #
        # model_paths = [
        #     str(self.opt['load_detector_models']),
        #     str(self.opt['load_unet_models']),
        #     str(self.opt['load_invisp_models']),
        # ]
        # models = [
        #     self.restore_restormer, self.restore_unet, self.restore_invisp
        # ]
        # folders = [
        #     'Restormer_restoration', 'Unet_restoration', 'Invisp_restoration'
        # ]
        #
        # print(f"loading pretrained restoration models")
        # for idx, model_path in enumerate(model_paths):
        #     model = models[idx]
        #     pretrain = f'{self.out_space_storage}/model/{folders[idx]}/' + model_path
        #     load_path_G = pretrain
        #
        #     print('Loading model for class [{:s}] ...'.format(load_path_G))
        #     if os.path.exists(load_path_G):
        #         self.load_network(load_path_G, model, strict=False)
        #     else:
        #         print('Did not find model for class [{:s}] ...'.format(load_path_G))

    def baseline_method(self, step=None, epoch=None):
        self.qf_predict_network.train()
        self.detector.eval()

        tampered_image_original, authentic_image_original = self.real_H, self.canny_image
        batch_size = tampered_image_original.shape[0]

        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        if step is not None:
            self.global_step = step

        batch = torch.cat([tampered_image_original, authentic_image_original], dim=0)
        which_model = self.opt['which_model_for_detector']
        with torch.no_grad():
            out = self.detector(batch)
            # if 'MVSS' in which_model:
            ### all mode return (feature, predicted_mask)
            feat_before_bottleneck, predicted_seg = out
            if self.IFA_bc_label is None:  # gt 0 means fake 1 means true
                self.IFA_bc_label = torch.tensor([0.] * batch_size + [1.] * batch_size).unsqueeze(1).cuda()

        with torch.enable_grad():
            predicted_score = self.qf_predict_network(feat_before_bottleneck)
            loss = self.bce_with_logit_loss(predicted_score, self.IFA_bc_label)

            loss.backward()
            nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()

            logs['sum_loss'] = loss.item()

        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                # self.save(self.global_step, folder='model', network_list=self.save_network_list)
                self.save_network(self.qf_predict_network, 'qf_predict', f"{which_model}_{epoch}_{self.global_step}",
                                  model_path=f'{self.out_space_storage}/model/{self.task_name}/')

        self.global_step = self.global_step + 1

        return logs, None, False





    def define_MLP(self, feature_dim, class_dim):
        from network.common import SimpleConv
        mlp = SimpleConv(feature_dim=feature_dim, class_dim=class_dim).cuda()
        model = DistributedDataParallel(mlp,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

