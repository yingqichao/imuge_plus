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
import pickle


class RR_IFA(base_IFA):
    def __init__(self, opt, args, train_set=None, val_set=None):
        """
            this file is mode 0

        """
        super(RR_IFA, self).__init__(opt, args, train_set, val_set)
        ### todo: options

        ### todo: constants
        self.history_accuracy = 0.1


    def network_definitions(self):
        self.consider_mask_prediction = self.opt['consider_mask_prediction']
        ### mode=8: InvISP to bi-directionally convert RGB and RAW,
        self.network_list = ['qf_predict_network']
        self.save_network_list = ['qf_predict_network']
        self.training_network_list = ['qf_predict_network']

        ### todo: network
        from CNN_architectures.resnet_feat_extract import ResNet_feat_extract
        # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True).cuda()
        # from CNN_architectures.pytorch_inceptionet import GoogLeNet
        # self.qf_predict_network = ResNet50(img_channel=4, num_classes=7, use_SRM=False, feat_concat=True).cuda()
        self.qf_predict_network = ResNet_feat_extract().cuda()

        if self.opt['load_predictor_models'] > 0:
            # self.load_model_wrapper(folder_name='predictor_folder', model_name='load_predictor_models',
            #                         network_lists=["qf_predict_network"], strict=False)
            load_detector_storage = self.opt['predictor_folder']
            model_path = str(self.opt['load_predictor_models'])  # last time: 10999
            load_models = self.opt['load_predictor_models'] > 0
            if load_models:
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

        self.qf_predict_network = DistributedDataParallel(self.qf_predict_network,
                                                          device_ids=[torch.cuda.current_device()],
                                                          find_unused_parameters=True)

        # self.generator = self.define_IFA_net()
        # # self.qf_predict_network = self.define_UNet_as_ISP()
        # self.load_model_wrapper(folder_name='predictor_folder', model_name='load_predictor_models',
        #                         network_lists=["qf_predict_network"], strict=True)

        ### todo: recovery network
        if self.consider_mask_prediction:
            self.network_list += ['generator']
            self.save_network_list += ['generator']
            self.training_network_list += ['generator']
            from models.networks import UNetDiscriminator
            self.generator = UNetDiscriminator(in_channels=3, out_channels=1, use_SRM=False, use_sigmoid=False,
                                               residual_blocks=8,dim=32, output_middle_feature=True).cuda()
            self.generator = DistributedDataParallel(self.generator,
                                                            device_ids=[torch.cuda.current_device()],
                                                            find_unused_parameters=True)
            if self.opt['load_predictor_models'] > 0:
                self.load_model_wrapper(folder_name='predictor_folder', model_name='load_predictor_models',
                                        network_lists=["generator"], strict=True)

        ### todo: inpainting model
        self.define_inpainting_edgeconnect()
        self.define_inpainting_ZITS()
        self.define_inpainting_lama()

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1).cuda()


    def define_IFA_net(self):
        from models.IFA.tres_model import Net
        self.qf_predict_network = Net(num_embeddings=1024).cuda()
        self.qf_predict_network = DistributedDataParallel(self.qf_predict_network,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)

    def predict_IFA_with_reference(self, step=None, epoch=None):
        # >> > triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        # >> > anchor = torch.randn(100, 128, requires_grad=True)
        # >> > positive = torch.randn(100, 128, requires_grad=True)
        # >> > negative = torch.randn(100, 128, requires_grad=True)
        # >> > output = triplet_loss(anchor, positive, negative)
        # >> > output.backward()
        ### todo: the first half of the batch is reserved as positive example, and the rest are modified as negative ones.
        self.qf_predict_network.train()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr
        batch_size = self.real_H.shape[0]//2
        positive_examples, negative_examples = self.real_H[:].clone(), self.real_H[:].clone()

        attacked_positive, _, _, _ = self.standard_attack_layer(
            modified_input=positive_examples, skip_tamper=True
        )
        attacked_tampered_negative, _, _, _ = self.standard_attack_layer(
            modified_input=negative_examples, skip_tamper=False
        )

        # anchor = self.qf_predict_network.module.embedding.weight
        # feat_positive = self.qf_predict_network(attacked_positive)
        # feat_negative = self.qf_predict_network(attacked_tampered_negative)
        # loss = self.triplet_loss(anchor, feat_positive, feat_negative)
        losses, feats = self.qf_predict_network(attacked_positive=attacked_positive,attacked_tampered_negative=attacked_tampered_negative)
        loss, loss_triplet, loss_class = losses
        anchor, feat_positive, feat_negative = feats
        l1_pos = self.l2_loss(feat_positive, anchor)
        l1_neg = self.l2_loss(feat_negative, anchor)
        loss.backward()
        nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
        self.optimizer_qf.step()
        self.optimizer_qf.zero_grad()

        logs['sum_loss'] = loss.item()
        logs['triplet_loss'] = loss.item()
        logs['class_loss'] = loss_class.item()
        logs['embedding'] = self.qf_predict_network.module.embedding.mean().item()
        logs['pos'] = l1_pos.item()
        logs['neg'] = l1_neg.item()

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(positive_examples),
                self.postprocess(negative_examples),
                self.postprocess(attacked_positive),
                self.postprocess(attacked_tampered_negative),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                # self.save(self.global_step, folder='model', network_list=self.save_network_list)
                self.save_network(self.qf_predict_network, 'qf_predict', f"{epoch}_{self.global_step}",
                                  model_path=self.out_space_storage + f'/model/{self.task_name}/')
                # pkl_path = f'{self.out_space_storage}/model/{self.task_name}/{self.global_step}_qf_predict.pkl'
                # with open(pkl_path, 'wb') as f:
                #     pickle.dump({'embedding': self.qf_predict_network.module.embedding}, f)
                #     print("Pickle saved to: {}".format(pkl_path))


        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = positive_examples

        self.global_step = self.global_step + 1

        return logs, None, False

    def predict_IFA_with_mask_prediction(self,step=None):
        ## received: real_H, canny_image
        self.generator.train()
        self.qf_predict_network.train()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        modified_input = self.real_H
        # masks_GT = self.canny_image

        ###### auto-generated tampering and post-processing
        attacked_image, attacked_forward, masks, masks_GT = self.standard_attack_layer(
            modified_input=modified_input
        )

        error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(
            attacked_forward)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
        logs['ERROR'] = error_l1
        attacked_image = attacked_image.detach().contiguous()

        ## todo: label is 0-5, representing 0-5% 5%-10% 10%-15% 15%-20% 25%-30% >30%
        label = 20*torch.mean(masks_GT, dim=[1, 2, 3])
        self.label = torch.where(label>6*torch.ones_like(label), 6*torch.ones_like(label), label).long()


        ## todo: first step: reference recovery
        estimated_mask, mid_feats = self.generator(attacked_image)
        loss_aux = self.bce_with_logit_loss(estimated_mask, masks_GT)

        output = self.qf_predict_network(torch.cat([attacked_image, estimated_mask.detach()], dim=1),
                                      mid_feats_from_recovery=mid_feats)
        loss_ce = self.ce_loss(output, self.label)

        # loss = emd_loss(labels, outputs)

        loss = loss_ce + loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
        nn.utils.clip_grad_norm_(self.generator.parameters(), 1)
        self.optimizer_qf.step()
        self.optimizer_generator.step()
        self.optimizer_qf.zero_grad()
        self.optimizer_generator.zero_grad()

        acc = (output.argmax(dim=1) == self.label).float().mean()
        logs['epoch_accuracy'] = acc
        logs['loss_ce'] = loss_ce
        logs['loss_aux'] = loss_aux

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.real_H),
                self.postprocess(attacked_image),
                self.postprocess(masks_GT),
                self.postprocess(estimated_mask),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)

        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H

        self.global_step = self.global_step + 1

        return logs, None, False


    def train_regression(self, step=None):
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
            modified_input = self.real_H
            # masks_GT = self.canny_image

            ###### auto-generated tampering and post-processing
            attacked_image, attacked_forward, masks, masks_GT = self.standard_attack_layer(
                modified_input=modified_input
            )

            error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(
                attacked_forward)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
            logs['ERROR'] = error_l1


            label = torch.mean(masks_GT, dim=[1, 2, 3]).float().unsqueeze(1)

            ###### just post-processing
            # index_for_postprocessing = self.global_step
            # quality_idx = self.get_quality_idx_by_iteration(index=index_for_postprocessing)
            # ## settings for attack
            # kernel = random.choice([3, 5, 7])  # 3,5,7
            # resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
            #                 int(self.random_float(0.5, 2) * self.width_height))
            # skip_robust = np.random.rand() > self.opt['skip_attack_probability']
            # if not skip_robust and self.opt['consider_robost']:
            #
            #     attacked_image, attacked_real_jpeg_simulate, _ = self.benign_attacks(attacked_forward=attacked_image,
            #                                                                          quality_idx=quality_idx,
            #                                                                          index=index_for_postprocessing,
            #                                                                          kernel_size=kernel,
            #                                                                          resize_ratio=resize_ratio
            #                                                                          )
            # else:
            #     attacked_image = attacked_image

            #######
            predictionQA, feat, quan_loss = self.qf_predict_network(attacked_image)


            l_class = self.l2_loss(predictionQA, label)

            l_sum = 0.1*quan_loss+l_class
            logs['CE'] = l_class.item()
            logs['Quan'] = quan_loss.item()

            l_sum.backward()

            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.qf_predict_network.parameters(), 1)
            self.optimizer_qf.step()
            self.optimizer_qf.zero_grad()


        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.real_H),
                self.postprocess(attacked_forward),
                self.postprocess(attacked_image),
                self.postprocess(10 * torch.abs(attacked_image - attacked_forward)),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (self.opt['model_save_period']-1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.save_network_list)

        if self.real_H is not None:
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H

        self.global_step = self.global_step + 1

        return logs, debug_logs, did_val

    def inference_RR_IFA(self, val_loader, num_images=None):
        self.qf_predict_network.eval()
        epoch_pos, epoch_neg, epoch_loss = 0, 0, 0
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                self.real_H, _ = batch
                self.real_H = self.real_H.cuda()
                batch_size = self.real_H.shape[0] // 2
                positive_examples, negative_examples = self.real_H[:batch_size], self.real_H[batch_size:]

                attacked_positive, _, _, _ = self.standard_attack_layer(
                    modified_input=positive_examples, skip_tamper=True
                )
                attacked_tampered_negative, _, _, _ = self.standard_attack_layer(
                    modified_input=negative_examples, skip_tamper=False
                )

                loss, feats = self.qf_predict_network(attacked_positive=attacked_positive,
                                               attacked_tampered_negative=attacked_tampered_negative)
                anchor,feat_positive,feat_negative = feats
                l1_pos = self.l1_loss(feat_positive,0.5*torch.ones_like(feat_positive))
                l1_neg = self.l1_loss(feat_negative,1.5*torch.ones_like(feat_positive))
                epoch_pos += l1_pos.item()
                epoch_neg += l1_neg.item()
                epoch_loss += loss.item()
                print(
                    f"[{idx}] digit: pos {epoch_pos/(idx+1)} neg {epoch_neg/(idx+1)} loss {epoch_loss/(idx+1)}")

                if (idx % 1000 == 3 or self.global_step <= 10):
                    images = stitch_images(
                        self.postprocess(positive_examples),
                        self.postprocess(negative_examples),
                        self.postprocess(attacked_positive),
                        self.postprocess(attacked_tampered_negative),
                        img_per_row=1
                    )

                    name = f"{self.out_space_storage}/images/{self.task_name}/{str(idx).zfill(5)}" \
                           f"_{str(self.rank)}_val.png"
                    images.save(name)

                if num_images is not None and num_images>=idx:
                    break

            # print(f"loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n")

    def inference_RR_IFA_regression(self,val_loader):
        epoch_loss = 0
        epoch_accuracy = 0
        self.qf_predict_network.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                self.real_H, _ = batch
                self.real_H = self.real_H.cuda()

                ###### auto-generated tampering and post-processing
                attacked_image, attacked_forward, masks, masks_GT = self.standard_attack_layer(
                    modified_input=self.real_H
                )

                # error_l1 = self.psnr(self.postprocess(attacked_image), self.postprocess(
                #     attacked_forward)).item()  # self.l1_loss(input=ERROR, target=torch.zeros_like(ERROR))
                # logs['ERROR'] = error_l1

                label = torch.mean(masks_GT, dim=[1, 2, 3]).float().unsqueeze(1)

                # masks_GT = masks_GT.unsqueeze(1).cuda()

                # label = torch.mean(masks_GT, dim=[1, 2, 3]).float().unsqueeze(1)

                # index_for_postprocessing = self.global_step
                # quality_idx = self.get_quality_idx_by_iteration(index=index_for_postprocessing)
                # ## settings for attack
                # kernel = random.choice([3, 5, 7])  # 3,5,7
                # resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
                #                 int(self.random_float(0.5, 2) * self.width_height))
                #
                # skip_robust = np.random.rand() > self.opt['skip_attack_probability']
                # if not skip_robust and self.opt['consider_robost']:
                #
                #     attacked_image, attacked_real_jpeg_simulate, _ = self.benign_attacks(attacked_forward=attacked_image,
                #                                                                          quality_idx=quality_idx,
                #                                                                          index=index_for_postprocessing,
                #                                                                          kernel_size=kernel,
                #                                                                          resize_ratio=resize_ratio
                #                                                                          )
                # else:
                #     attacked_image = attacked_image

                #######
                predictionQA, feat, quan_loss = self.qf_predict_network(attacked_image)
                l_class = self.l2_loss(predictionQA, label)
                l_sum = 0.1 * quan_loss + l_class

                label_int = (20*label).int()
                predictionQA_int = (20*predictionQA).int()

                acc = (predictionQA_int == label_int).float().mean()
                epoch_accuracy += acc.item() / len(val_loader)
                epoch_loss += l_sum.item() / len(val_loader)
                print(f"[{idx}] digit: pred {predictionQA.item()} gt {label.item()} | class: pred {predictionQA_int.item()} gt {label_int.item()}]")

            print(f"loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n")

            if epoch_accuracy > self.history_accuracy:
                print(f'Saving models and training states.')
                # self.model_save(path='checkpoint/latest', epochs=self.global_step)
                self.save(self.global_step, folder='model', network_list=self.save_network_list)
                self.history_accuracy = epoch_accuracy

