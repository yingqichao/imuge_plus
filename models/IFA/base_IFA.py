import math
import os

# from cycleisp_models.cycleisp import Raw2Rgb
# from MVSS.models.mvssnet import get_mvss
# from MVSS.models.resfcn import ResFCN
# from data.pipeline import pipeline_tensor2image
# import matlab.engine
import torch.nn.functional as Functional
import torchvision.transforms.functional as F
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf
import yaml
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
from inpainting_methods.lama_models.my_own_elastic import my_own_elastic
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


# import matlab.engine
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
# import contextual_loss as cl
# import contextual_loss.functional as F

class base_IFA(BaseModel):
    def __init__(self, opt, args, train_set=None, val_set=None, prepare_networks_optimizers=True):
        """
            prepare_networks_optimizers: set True current, preserved for future uses that only invoke static methods without creating an instances.

        """
        super(base_IFA, self).__init__(opt, args, train_set, val_set)
        ### todo: options
        self.default_detection_networks_for_training = ['qf_predict_network']
        self.global_step_for_inpainting = 0

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

        # self.mode_dict = {
        #     0: 'RR IFA',
        #     1: 'val RR IFA',
        # }
        print(f"network list:{self.network_list}")
        print(f"Current mode: {self.args.mode}")
        # print(f"Function: {self.mode_dict[self.args.mode]}")


        self.out_space_storage = f"{self.opt['name']}"

        if prepare_networks_optimizers:
            ### todo: network definitions
            self.network_definitions()
            print(f"network list: {self.network_list}")

            ### todo: Optimizers
            self.define_optimizers()

            ### todo: Scheduler
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30000))

            ### todo: creating dirs
            self.create_folders_for_the_experiment()

        # ## load states, deprecated, because there can be too many models from different paths.
        # state_path = self.load_space_storage + self.load_storage + '{}.state'.format(self.model_path)
        # if load_state:
        #     print('Loading training state')
        #     if os.path.exists(state_path):
        #         self.resume_training(state_path, self.network_list)
        #     else:
        #         print('Did not find state [{:s}] ...'.format(state_path))

    def create_folders_for_the_experiment(self):
        from utils.commons import create_folder
        create_folder(self.out_space_storage)
        create_folder(self.out_space_storage + "/model")
        create_folder(self.out_space_storage + "/images")
        create_folder(self.out_space_storage + "/model/" + self.task_name)
        create_folder(self.out_space_storage + "/images/" + self.task_name)

    def network_definitions(self):
        pass

    def load_model_wrapper(self, *, folder_name, model_name, network_lists, strict=True):
        load_detector_storage = self.opt[folder_name]
        model_path = str(self.opt[model_name])  # last time: 10999
        load_models = self.opt[model_name] > 0
        if load_models:
            print(f"loading models: {network_lists}")
            pretrain = load_detector_storage + model_path
            self.reload(pretrain, network_list=network_lists, strict=strict)


    def feed_data_router(self, batch, mode):

        self.feed_data_COCO_like(batch, mode='train')


    def feed_data_val_router(self, batch, mode):

        self.feed_data_COCO_like(batch, mode='val')


    def feed_data_COCO_like(self, batch, mode='train'):
        if mode == 'train':
            img, mask = batch
            self.real_H = img.cuda()
            self.canny_image = mask.cuda()
            if len(self.canny_image.shape)==3:
                self.canny_image = self.canny_image.unsqueeze(1)
        else:
            img, mask = batch
            self.real_H_val = img.cuda()
            self.canny_image_val = mask.unsqueeze(1).cuda()
            if len(self.canny_image_val.shape)==3:
                self.canny_image_val = self.canny_image_val.unsqueeze(1)

    def optimize_parameters_router(self, mode, step=None, epoch=None):
        if mode == 0.0:
            return self.IFA_dense_prediction_postprocess(step=step, epoch=epoch) #self.IFA_binary_classification(step=step, epoch=epoch)
        elif mode == 1.0:
            return self.pretrain_restormer_restoration(step=step, epoch=epoch)
        elif mode == 2.0:
            return self.predict_PSNR(step=step, epoch=epoch)
        elif mode ==3.0:
            return self.baseline_method(step=step, epoch=epoch)
        else:
            raise NotImplementedError(f"没有找到mode {mode} 对应的方法，请检查！")

    def validate_router(self, mode, step=None, epoch=None):
        if mode == 0.0:
            return self.validate_IFA_dense_prediction_postprocess(step=step, epoch=epoch)
        elif mode == 1.0:
            pass
        elif mode == 2.0:
            pass
        elif mode == 3.0:
            pass
        else:
            raise NotImplementedError(f"没有找到mode {mode} 对应的 validate 方法，请检查！")

    ####################################################################################################
    # todo: MODE == 0
    # todo: train restoration models (restormer, unet and invisp)
    ####################################################################################################
    def IFA_binary_classification(self, step=None, epoch=None):
        pass

    def validate_IFA_dense_prediction_postprocess(self, step=None, epoch=None):
        pass

    def IFA_dense_prediction_postprocess(self, step=None, epoch=None):
        pass

    ####################################################################################################
    # todo: MODE == 2
    # todo: predict PSNR using CMT
    ####################################################################################################
    def predict_PSNR(self, step=None, epoch=None):
        pass

    ####################################################################################################
    # todo: MODE == 1
    # todo: get_protected_RAW_and_corresponding_images
    ####################################################################################################
    def pretrain_restormer_restoration(self, step=None, epoch=None):
        pass

    ####################################################################################################
    # todo: MODE == 3
    # todo: get_protected_RAW_and_corresponding_images
    ####################################################################################################
    def baseline_method(self, step=None, epoch=None):
        pass


    def define_detector(self, *, opt_name):

        which_model = self.opt[opt_name]
        if 'OSN' in which_model:
            model = self.define_OSN_as_detector()
        elif 'CAT' in which_model:
            model = self.define_CATNET()
        elif 'MVSS' in which_model:
            model = self.define_MVSS_as_detector()
        elif 'Mantra' in which_model:
            model = self.define_MantraNet_as_detector()
        elif 'Resfcn' in which_model:
            model = self.define_resfcn_as_detector()
        elif 'MPF' in which_model:
            print("using my_own_elastic as localizer.")
            model = self.define_my_own_elastic_as_detector()
        else:
            raise NotImplementedError("测试要用的detector没找到！请检查！")

        return model

    def detector_predict(self, *, model, attacked_image, opt_name, masks_GT=None):
        if 'CAT' in self.opt[opt_name]:
            _, pred_resfcn = self.CAT_predict(model=model,
                                              attacked_image=attacked_image)

        elif 'MVSS' in self.opt[opt_name]:
            _, pred_resfcn = self.MVSS_predict(model=model,
                                               attacked_image=attacked_image)

        elif 'Resfcn' in self.opt[opt_name] \
                or 'Mantra' in self.opt[opt_name]:
            pred_resfcn = self.predict_with_sigmoid_eg_resfcn_mantra(model=model,
                                                                     attacked_image=attacked_image)

        elif "MPF" in self.opt[opt_name]:
            _, pred_resfcn = self.MPF_predict(model=model,
                                              masks_GT=masks_GT,
                                              attacked_image=attacked_image)

        elif "OSN" in self.opt[opt_name]:
            pred_resfcn = self.predict_with_NO_sigmoid(model=model,
                                                       attacked_image=attacked_image)
        else:
            _, pred_resfcn = self.CAT_predict(model=model,
                                              attacked_image=attacked_image)

        CE_resfcn = self.bce_loss(pred_resfcn, masks_GT)

        return pred_resfcn, CE_resfcn

    def CAT_predict(self, *, model, attacked_image):
        pred_resfcn = model(attacked_image, None)
        pred_resfcn = Functional.interpolate(pred_resfcn, size=(self.width_height, self.width_height), mode='bilinear')
        pred_resfcn = Functional.softmax(pred_resfcn, dim=1)
        _, pred_resfcn = torch.split(pred_resfcn, 1, dim=1)

        return None, pred_resfcn

    def MVSS_predict(self, *, model, attacked_image):
        _, pred_resfcn = model(attacked_image)
        pred_resfcn = torch.sigmoid(pred_resfcn)

        return None, pred_resfcn

    def predict_with_sigmoid_eg_resfcn_mantra(self, *, model, attacked_image):
        """
            eg: resfcn, mantranet
        """
        pred_resfcn = model(attacked_image)
        pred_resfcn = torch.sigmoid(pred_resfcn)

        return pred_resfcn

    def predict_with_NO_sigmoid(self, *, model, attacked_image):
        """
            eg: OSN
        """
        pred_resfcn = model(attacked_image)

        return pred_resfcn

    def MPF_predict(self, *, model, masks_GT, attacked_image):
        attacked_cannied, _ = self.get_canny(attacked_image, masks_GT)

        pred_resfcn = model(attacked_image, canny=attacked_cannied)
        if isinstance(pred_resfcn, (tuple)):
            # pred_resfcn,  _ = pred_resfcn
            _, pred_resfcn = pred_resfcn
        pred_resfcn = torch.sigmoid(pred_resfcn)

        return pred_resfcn


    ####################################################################################################
    # todo: define how to tamper the rendered RGB
    ####################################################################################################
    def tampering_RAW(self, *, masks, masks_GT, modified_input, percent_range, index=None):
        batch_size, height_width = modified_input.shape[0], modified_input.shape[2]
        ####### Tamper ###############
        # attacked_forward = torch.zeros_like(modified_input)
        # for img_idx in range(batch_size):
        if index is None:
            index = self.global_step % self.amount_of_tampering

        if index in self.opt['simulated_splicing_indices']:  # self.using_splicing():
            ### todo: splicing
            attacked_forward = self.splicing(forward_image=modified_input, masks=masks)

        elif index in self.opt['simulated_copymove_indices']:  # self.using_copy_move():
            ### todo: copy-move
            attacked_forward, masks, masks_GT = self.copymove(forward_image=modified_input, masks=masks,
                                                              masks_GT=masks_GT,
                                                              percent_range=percent_range)
            # del self.tamper_shifted
            # del self.mask_shifted
            # torch.cuda.empty_cache()

        elif index in self.opt['simulated_copysplicing_indices']:  # self.using_simulated_inpainting:
            ### todo: copy-splicing
            attacked_forward, masks, masks_GT = self.copysplicing(forward_image=modified_input, masks=masks,
                                                                  percent_range=percent_range,
                                                                  another_immunized=self.previous_protected)
        elif index in self.opt['simulated_inpainting_indices']:  # self.using_splicing():
            ### todo: inpainting
            ## note! self.global_step_for_inpainting, not index, decides which inpainting model will be used

            use_which_inpainting = self.global_step_for_inpainting % self.amount_of_inpainting

            if use_which_inpainting in self.opt['edgeconnect_as_inpainting']:
                ## edgeconnect
                attacked_forward_edgeconnect = self.inpainting_edgeconnect(forward_image=modified_input, masks=masks_GT)
            elif use_which_inpainting in self.opt['zits_as_inpainting']:
                ## zits
                attacked_forward_edgeconnect = self.inpainting_ZITS(forward_image=modified_input,
                                                                    masks=masks_GT)
            else:  # if use_which_inpainting in self.opt['lama_as_inpainting']:
                ## lama
                attacked_forward_edgeconnect = self.inpainting_lama(forward_image=modified_input,
                                                                    masks=masks_GT)

            attacked_forward = attacked_forward_edgeconnect
            self.global_step_for_inpainting += 1

        else:
            print(index)
            raise NotImplementedError("Tamper的方法没找到！请检查！")

        attacked_forward = self.clamp_with_grad(attacked_forward)
        # attacked_forward = self.Quantization(attacked_forward)

        return attacked_forward, masks, masks_GT

    def define_CATNET(self):
        print("using CATnet")
        from detection_methods.CATNet.model import get_model
        model = get_model().cuda()
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model


    def define_my_own_elastic_as_detector(self):
        print("using MPF_net as detector")
        model = my_own_elastic(nin=3, nout=1, depth=4, nch=36, num_blocks=self.opt['dtcwt_layers'],
                               use_norm_conv=True).cuda()
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_MantraNet_as_detector(self):
        print("using mantranet as detector")
        from detection_methods.MantraNet.mantranet import pre_trained_model
        model_path = './MantraNetv4.pt'
        model = pre_trained_model(model_path).cuda()
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_MVSS_as_detector(self):
        print("using MVSS as detector")
        model_path = '/groupshare/codes/detection_methods/MVSS/ckpt/mvssnet_casia.pt'
        from detection_methods.MVSS.models.mvssnet import get_mvss
        model = get_mvss(
            backbone='resnet50',
            pretrained_base=True,
            nclass=1,
            sobel=True,
            constrain=True,
            n_input=3
        ).cuda()
        ckp = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckp, strict=True)
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_OSN_as_detector(self):
        print("using OSN as detector")
        from detection_methods.ImageForensicsOSN.test import get_model
        # self.localizer = #HWMNet(in_chn=3, wf=32, depth=4, use_dwt=False).cuda()
        # self.localizer = DistributedDataParallel(self.localizer, device_ids=[torch.cuda.current_device()],
        #                                     find_unused_parameters=True)
        model = get_model('/groupshare/ISP_results/models/').cuda()

        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_RAW2RAW_network(self, n_channels=None):
        # if 'UNet' not in self.task_name:
        print("using my_own_elastic as KD_JPEG.")
        if n_channels is None:
            n_channels = 3 if "ablation" in self.opt['task_name_KD_JPEG_model'] else 4  # 36/48 12
        self.KD_JPEG = my_own_elastic(nin=n_channels, nout=n_channels, depth=4, nch=48,
                                      num_blocks=self.opt['dtcwt_layers'],
                                      use_norm_conv=False).cuda()
        # else:
        #     self.KD_JPEG = HWMNet(in_chn=1, out_chn=1, wf=32, depth=4, subtask=0, style_control=False,
        #                           use_dwt=False).cuda()
        # SPADE_UNet(in_channels=1, out_channels=1, residual_blocks=2).cuda()
        # Inveritible_Decolorization_PAMI(dims_in=[[1, 64, 64]], block_num=[2, 2, 2], augment=False, ).cuda()
        # InvISPNet(channel_in=3, channel_out=3, block_num=4, network="ResNet").cuda() HWMNet(in_chn=1, wf=32, depth=4).cuda() # UNetDiscriminator(in_channels=1,use_SRM=False).cuda()
        self.KD_JPEG = DistributedDataParallel(self.KD_JPEG, device_ids=[torch.cuda.current_device()],
                                               find_unused_parameters=True)

    def define_ISP_network_training(self):
        print("using two network-based networks")
        self.generator = self.define_invISP()
        self.qf_predict_network = self.define_UNet_as_ISP()
        self.netG = self.define_MPF_as_ISP()

    def define_invISP(self, block_num=[2, 2, 2]):
        print("using invISP as ISP")
        model = Inveritible_Decolorization_PAMI(dims_in=[[3, 64, 64]], block_num=block_num, augment=False,
                                                ).cuda()  # InvISPNet(channel_in=3, channel_out=3, block_num=4, network="ResNet").cuda()
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model


    def define_UNet_as_ISP(self):
        print("using Unet as ISP")
        model = UNetDiscriminator(in_channels=3, out_channels=3, use_SRM=False).cuda()
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_MPF_as_ISP(self):
        print("using MPF as ISP")
        model = my_own_elastic(nin=3, nout=3, depth=4, nch=36, num_blocks=self.opt['dtcwt_layers'],
                               use_norm_conv=False).cuda()
        # model = HWMNet(in_chn=3, wf=32, depth=4, use_dwt=True).cuda()
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_resfcn_as_detector(self):
        print("using ResFCN as detector")
        from detection_methods.MVSS.models.resfcn import ResFCN
        model = ResFCN().cuda()
        checkpoint = torch.load('/groupshare/codes/resfcn_coco_1013.pth', map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model

    def define_inpainting_ZITS(self):
        from inpainting_methods.ZITSinpainting.src.FTR_trainer import ZITSModel
        from shutil import copyfile
        from inpainting_methods.ZITSinpainting.src.config import Config
        print("Building ZITS...........please wait...")
        model_path = '/groupshare/ckpt/zits_places2_hr'
        config_path = os.path.join(model_path, 'config.yml')

        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(config_path):
            copyfile('./ZITSinpainting/config_list/config_ZITS_HR_places2.yml', config_path)

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        config = Config(config_path)
        config.MODE = 1
        # config.GPUS = 1
        # config.GPU_ids = '0'
        # config.world_size = 1
        self.ZITS_model = ZITSModel(config=config, test=True).cuda()
        self.ZITS_model = DistributedDataParallel(self.ZITS_model,
                                                  device_ids=[torch.cuda.current_device()],
                                                  find_unused_parameters=True)
        self.ZITS_model.eval()

    def define_inpainting_edgeconnect(self):
        from inpainting_methods.edgeconnect.main import load_config
        from inpainting_methods.edgeconnect.src.models import EdgeModel, InpaintingModel
        print("Building edgeconnect...........please wait...")
        config = load_config(mode=2)
        self.edge_model = EdgeModel(config)
        self.inpainting_model = InpaintingModel(config)
        self.edge_model.load()
        self.inpainting_model.load()
        self.edge_model = self.edge_model.cuda()
        self.inpainting_model = self.inpainting_model.cuda()
        self.edge_model = DistributedDataParallel(self.edge_model,
                                                  device_ids=[torch.cuda.current_device()],
                                                  find_unused_parameters=True)
        self.inpainting_model = DistributedDataParallel(self.inpainting_model,
                                                        device_ids=[torch.cuda.current_device()],
                                                        find_unused_parameters=True)
        self.edge_model.eval()
        self.inpainting_model.eval()
        # self.edgeconnect_model = get_model()
        # self.edgeconnect_model = DistributedDataParallel(self.edgeconnect_model,
        #                                               device_ids=[torch.cuda.current_device()],
        #                                               find_unused_parameters=True)

    def define_inpainting_lama(self):
        from inpainting_methods.saicinpainting.training.trainers import load_checkpoint
        print("Building LAMA...........please wait...")
        checkpoint_path = './big-lama/models/best.ckpt'
        train_config_path = './big-lama/config.yaml'
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # out_ext = predict_config.get('out_ext', '.png')

        self.lama_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.lama_model.freeze()
        # if not refine == False:
        # if not predict_config.get('refine', False):
        self.lama_model = self.lama_model.cuda()

    @torch.no_grad()
    def inpainting_lama(self, *, forward_image, masks):
        batch = {
            'image': forward_image,
            'mask': masks
        }
        # batch['mask'] = (batch['mask'] > 0) * 1
        batch = self.lama_model(batch)
        result = batch['inpainted']
        return forward_image * (1 - masks) + result.clone().detach() * masks

    @torch.no_grad()
    def inpainting_edgeconnect(self, *, forward_image, masks, image_gray=None, image_canny=None):
        # items = (forward_image, image_gray, image_canny, masks)
        if image_gray is None:
            modified_crop_out = forward_image * (1 - masks)
            image_gray, image_canny = self.get_canny(input=modified_crop_out, masks_GT=masks)

        self.edge_model.eval()
        self.inpainting_model.eval()
        edges = self.edge_model(image_gray, image_canny, masks).detach()
        outputs = self.inpainting_model(forward_image, edges, masks)
        result = (outputs * masks) + forward_image * (1 - masks)

        # result = self.edgeconnect_model(items)

        return forward_image * (1 - masks) + result.clone().detach() * masks

    def inpainting_for_RAW(self, *, forward_image, masks, gt_rgb):
        return forward_image * (1 - masks) + gt_rgb * masks

    @torch.no_grad()
    def inpainting_ZITS(self, *, forward_image, masks):
        from inpainting_methods.ZITSinpainting.single_image_test import wf_inference_test, \
            load_masked_position_encoding
        sigma = 3.0
        valid_th = 0.85
        # items = load_images_for_test(src_img, mask_img, sigma=sigma)
        ### load_image must be customized
        image_256 = Functional.interpolate(
            forward_image.clone(),
            size=[256, 256],
            mode='bilinear')
        image_gray, image_canny = self.get_canny(input=image_256, sigma=sigma)

        rel_pos, abs_pos, direct = None, None, None  # torch.zeros_like(masks)[:,0].long(), torch.zeros_like(masks)[:,0].long(), torch.zeros_like(masks)[:,0].long()
        for i in range(forward_image.shape[0]):
            mask_numpy = masks[i, 0].mul(255).add_(0.5).clamp_(0, 255).contiguous().to('cpu', torch.uint8).numpy()
            rel_pos_single, abs_pos_single, direct_single = load_masked_position_encoding(mask_numpy)
            rel_pos_single = torch.LongTensor(rel_pos_single).unsqueeze(0).cuda()
            abs_pos_single = torch.LongTensor(abs_pos_single).unsqueeze(0).cuda()
            direct_single = torch.LongTensor(direct_single).unsqueeze(0).cuda()
            rel_pos = rel_pos_single if rel_pos is None else torch.cat([rel_pos, rel_pos_single], dim=0)
            abs_pos = abs_pos_single if abs_pos is None else torch.cat([abs_pos, abs_pos_single], dim=0)
            direct = direct_single if direct is None else torch.cat([direct, direct_single], dim=0)

        batch = dict()
        batch['image'] = forward_image
        batch['img_256'] = image_256.clone()
        batch['mask'] = masks
        batch['mask_256'] = torch.where(Functional.interpolate(
            masks.clone(),
            size=[256, 256],
            mode='bilinear') > 0, 1.0, 0.0).cuda()
        batch['mask_512'] = masks.clone()
        batch['edge_256'] = image_canny
        batch['img_512'] = forward_image.clone()
        batch['rel_pos'] = rel_pos
        batch['abs_pos'] = abs_pos
        batch['direct'] = direct
        batch['h'] = forward_image.shape[2]
        batch['w'] = forward_image.shape[3]

        line = wf_inference_test(self.ZITS_model.module.wf, batch['img_512'], h=256, w=256, masks=batch['mask_512'],
                                 valid_th=valid_th, mask_th=valid_th)
        batch['line_256'] = line

        # for k in batch:
        #     if type(batch[k]) is torch.Tensor:
        #         batch[k] = batch[k].cuda()
        merged_image = self.ZITS_model(batch)

        return merged_image.clone().detach() * masks + forward_image * (1 - masks)

    ####################################################################################################
    # todo: settings for beginning training
    ####################################################################################################
    def data_augmentation_on_rendered_rgb(self, modified_input, index=None):
        if index is None:
            index = self.global_step % self.amount_of_augmentation

        is_stronger = np.random.rand() > 0.5
        if index in self.opt['simulated_hue']:
            ## careful!
            strength = np.random.rand() * (0.05 if is_stronger > 0 else -0.05)
            modified_adjusted = F.adjust_hue(modified_input, hue_factor=0 + strength)  # 0.5 ave
        elif index in self.opt['simulated_contrast']:
            strength = np.random.rand() * (0.3 if is_stronger > 0 else -0.3)
            modified_adjusted = F.adjust_contrast(modified_input, contrast_factor=1 + strength)  # 1 ave
        elif index in self.opt['simulated_gamma']:
            ## careful!
            strength = np.random.rand() * (0.05 if is_stronger > 0 else -0.05)
            modified_adjusted = F.adjust_gamma(modified_input, gamma=1 + strength)  # 1 ave
        elif index in self.opt['simulated_saturation']:
            strength = np.random.rand() * (0.3 if is_stronger > 0 else -0.3)
            modified_adjusted = F.adjust_saturation(modified_input, saturation_factor=1 + strength)
        elif index in self.opt['simulated_brightness']:
            strength = np.random.rand() * (0.3 if is_stronger > 0 else -0.3)
            modified_adjusted = F.adjust_brightness(modified_input,
                                                    brightness_factor=1 + strength)  # 1 ave
        else:
            raise NotImplementedError("图像增强的index错误，请检查！")
        modified_adjusted = self.clamp_with_grad(modified_adjusted)

        return modified_adjusted  # modified_input + (modified_adjusted - modified_input).detach()

    def gamma_correction(self, tensor, avg=4095, digit=2.2):
        ## gamma correction
        #             norm_value = np.power(4095, 1 / 2.2) if self.camera_name == 'Canon_EOS_5D' else np.power(16383, 1 / 2.2)
        #             input_raw_img = np.power(input_raw_img, 1 / 2.2)
        norm = math.pow(avg, 1 / digit)
        tensor = torch.pow(tensor * avg, 1 / digit)
        tensor = tensor / norm

        return tensor

    def do_aug_train(self, *, attacked_forward):
        skip_augment = np.random.rand() > 0.85
        if not skip_augment and self.opt["conduct_augmentation"]:
            attacked_adjusted = self.data_augmentation_on_rendered_rgb(attacked_forward)
        else:
            attacked_adjusted = attacked_forward

        return attacked_adjusted

    def do_postprocess_train(self, *, attacked_adjusted):
        skip_robust = np.random.rand() > 0.85
        if not skip_robust and self.opt["consider_robost"]:
            quality_idx = self.get_quality_idx_by_iteration(index=self.global_step)

            attacked_image = self.benign_attacks(attacked_forward=attacked_adjusted,
                                                 quality_idx=quality_idx)
        else:
            attacked_image = attacked_adjusted

        return attacked_image

    ####################################################################################################
    # todo:  Method specification
    # todo: RAW2RAW, baseline, ISP generation, localization
    ####################################################################################################

    def RAW_protection_by_my_own_elastic(self, *, input_raw_one_dim):
        input_psdown = self.psdown(input_raw_one_dim)
        modified_psdown = input_psdown + self.KD_JPEG(input_psdown)
        modified_raw_one_dim = self.psup(modified_psdown)
        return modified_raw_one_dim

    def render_image_using_ISP(self, *, input_raw, input_raw_one_dim, gt_rgb,
                               file_name, camera_name
                               ):
        ### gt_rgb is only loaded to read tensor size

        ### three networks used during training
        if self.opt['test_restormer'] == 0:
            self.generator.eval()
            self.qf_predict_network.eval()
            self.netG.eval()
            if self.global_step % 3 == 0:
                isp_model_0, isp_model_1 = self.generator, self.qf_predict_network
            elif self.global_step % 3 == 1:
                isp_model_0, isp_model_1 = self.netG, self.qf_predict_network
            else:  # if self.global_step%3==2:
                isp_model_0, isp_model_1 = self.netG, self.generator

            #### invISP AS SUBSEQUENT ISP####
            modified_input_0 = isp_model_0(input_raw)
            if self.opt['use_gamma_correction']:
                modified_input_0 = self.gamma_correction(modified_input_0)
            modified_input_0 = self.clamp_with_grad(modified_input_0)

            modified_input_1 = isp_model_1(input_raw)
            if self.opt['use_gamma_correction']:
                modified_input_1 = self.gamma_correction(modified_input_1)
            modified_input_1 = self.clamp_with_grad(modified_input_1)

            skip_the_second = np.random.rand() > 0.8
            alpha_0 = 1.0 if skip_the_second else np.random.rand()
            alpha_1 = 1 - alpha_0
            non_tampered_image = alpha_0 * modified_input_0
            non_tampered_image += alpha_1 * modified_input_1

        ### conventional ISP
        elif self.opt['test_restormer'] == 1:
            from data.pipeline import rawpy_tensor2image
            non_tampered_image = torch.zeros_like(gt_rgb)
            for idx_pipeline in range(gt_rgb.shape[0]):
                # [B C H W]->[H,W]
                raw_1 = input_raw_one_dim[idx_pipeline]
                # numpy_rgb = pipeline_tensor2image(raw_image=raw_1, metadata=metadata, input_stage='normal',
                #                                   output_stage='gamma')
                numpy_rgb = rawpy_tensor2image(raw_image=raw_1, template=file_name[idx_pipeline],
                                               camera_name=camera_name[idx_pipeline], patch_size=512) / 255

                non_tampered_image[idx_pipeline:idx_pipeline + 1] = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(numpy_rgb, (2, 0, 1)))).contiguous().float()

            # non_tampered_image = torch.zeros_like(gt_rgb)
            # for idx_pipeline in range(gt_rgb.shape[0]):
            #     metadata = self.val_set.metadata_list[file_name[idx_pipeline]]
            #     flip_val = metadata['flip_val']
            #     metadata = metadata['metadata']
            #     # 在metadata中加入要用的flip_val和camera_name
            #     metadata['flip_val'] = flip_val
            #     metadata['camera_name'] = camera_name
            #     # [B C H W]->[H,W]
            #     raw_1 = input_raw_one_dim[idx_pipeline].permute(1, 2, 0).squeeze(2)
            #     # numpy_rgb = pipeline_tensor2image(raw_image=raw_1, metadata=metadata, input_stage='normal',
            #     #                                   output_stage='gamma')
            #     numpy_rgb = isp_tensor2image(raw_image=raw_1, metadata=None, file_name=file_name[idx_pipeline],
            #                                  camera_name=camera_name[idx_pipeline])
            #
            #     non_tampered_image[idx_pipeline:idx_pipeline + 1] = torch.from_numpy(
            #         np.ascontiguousarray(np.transpose(numpy_rgb, (2, 0, 1)))).contiguous().float()
        ### restormer
        elif self.opt['test_restormer'] == 2:
            self.localizer.eval()
            non_tampered_image = self.localizer(input_raw)

        elif self.opt['test_restormer'] == 3:
            self.generator.eval()
            self.qf_predict_network.eval()
            self.netG.eval()
            if self.global_step % 6 == 0:
                isp_model_0, isp_model_1 = "pipeline", self.generator
            elif self.global_step % 6 == 1:
                isp_model_0, isp_model_1 = self.generator, self.qf_predict_network
            elif self.global_step % 6 == 2:
                isp_model_0, isp_model_1 = "pipeline", self.qf_predict_network
            elif self.global_step % 6 == 3:
                isp_model_0, isp_model_1 = self.netG, self.qf_predict_network
            elif self.global_step % 6 == 4:
                isp_model_0, isp_model_1 = "pipeline", self.netG
            else:  # if self.global_step % 6 == 5:
                isp_model_0, isp_model_1 = self.netG, self.generator

            ### first
            if isinstance(isp_model_0, str):
                modified_input_0 = self.pipeline_ISP_gathering(modified_raw_one_dim=input_raw_one_dim,
                                                               file_name=file_name, gt_rgb=gt_rgb)
            else:
                modified_input_0 = isp_model_0(input_raw)
            modified_input_0 = self.clamp_with_grad(modified_input_0)

            # ### second
            # modified_input_1 = isp_model_1(input_raw)
            # modified_input_1 = self.clamp_with_grad(modified_input_1)
            #
            # ##################   doing mixup on the images   ###############################################
            # ### note: our goal is that the rendered rgb by the protected RAW should be close to that rendered by unprotected RAW
            # ### thus, we are not let the ISP network approaching the ground-truth RGB.
            # skip_the_second = np.random.rand() > 0.8
            # alpha_0 = 1.0 if skip_the_second else np.random.rand()
            # alpha_1 = 1 - alpha_0

            # modified_input = alpha_0 * modified_input_0
            # modified_input += alpha_1 * modified_input_1

            # ISP_L1_sum = self.l1_loss(input=modified_input, target=tamper_source)
            # ISP_SSIM_sum = - self.ssim_loss(modified_input, tamper_source)

            ### collect the protected images ###
            modified_input = self.clamp_with_grad(modified_input_0)
            non_tampered_image = modified_input

        elif self.opt['test_restormer'] == 4:
            modified_input_0 = self.pipeline_ISP_gathering(modified_raw_one_dim=input_raw_one_dim,
                                                           file_name=file_name, gt_rgb=gt_rgb)
            modified_input = self.clamp_with_grad(modified_input_0)
            non_tampered_image = modified_input
        else:
            # print('here')
            non_tampered_image = gt_rgb

        # non_tampered_image = gt_rgb
        return non_tampered_image

    def ISP_image_generation_general(self, *, network, input_raw, target):
        modified_input_generator = network(input_raw)
        ISP_L1_FOR = self.l1_loss(input=modified_input_generator, target=target)
        # ISP_SSIM = - self.ssim_loss(modified_input_generator, gt_rgb)
        modified_input_generator = self.clamp_with_grad(modified_input_generator)
        if self.opt['use_gamma_correction']:
            modified_input_generator = self.gamma_correction(modified_input_generator)

        ISP_loss = ISP_L1_FOR

        return modified_input_generator, ISP_loss

    def standard_attack_layer(self, *, modified_input, tamper_index=None,
                              skip_tamper=False, skip_augment=None, skip_robust=None, skip_mask=False,
                              percent_range=[0.00, 0.25]):
        ##############    cropping   ###################################################################################

        ## settings for attack
        kernel = random.choice([3, 5, 7])  # 3,5,7
        resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
                        int(self.random_float(0.5, 2) * self.width_height))
        index_for_postprocessing = self.global_step

        quality_idx = self.get_quality_idx_by_iteration(index=index_for_postprocessing)

        modified_cropped, locs = modified_input, None
        # if self.global_step % 10 in self.opt['crop_indices']:
        #     # not (self.global_step%self.amount_of_benign_attack in (self.opt['simulated_gblur_indices'] + self.opt['simulated_mblur_indices'])) and \
        #     # not (self.global_step%self.amount_of_benign_attack in self.opt['simulated_strong_JPEG_indices'] and quality_idx<16):
        #     logs["cropped"] = True
        #     locs, _, modified_cropped = self.cropping_mask_generation(forward_image=modified_input,
        #                                                               min_rate=self.opt['cropping_lower_bound'],
        #                                                               max_rate=1.0)
        #     _, _, gt_rgb = self.cropping_mask_generation(forward_image=gt_rgb, locs=locs)
        #     h_start, h_end, w_start, w_end = locs
        #     crop_rate = (h_end - h_start) / self.width_height * (w_end - w_start) / self.width_height
        # else:
        #     logs["cropped"] = False

        # percent_range = [0.00, 0.25]
            # if self.global_step % self.amount_of_tampering not in \
            #                            (self.opt['simulated_copymove_indices'] + self.opt[
            #                                'simulated_inpainting_indices']) \
            # else [0.00, 0.2]

        if not skip_mask:
            masks, masks_GT, percent_range = self.mask_generation(modified_input=modified_input,
                                                                  percent_range=percent_range, index=self.global_step)
        else:
            masks, masks_GT = None, None
        # rate_mask = torch.mean(masks_GT)

        if not skip_tamper:
            attacked_forward, masks, masks_GT = self.tampering_RAW(
                masks=masks, masks_GT=masks_GT,
                modified_input=modified_cropped, percent_range=percent_range, index=tamper_index
            )
        else:
            attacked_forward = modified_cropped

        ###############   white-balance, gamma, tone mapping, etc.  ####################################################
        if skip_augment is None:
            skip_augment = np.random.rand() > self.opt['skip_aug_probability']
        if not skip_augment and self.opt['conduct_augmentation']:
            attacked_adjusted = self.data_augmentation_on_rendered_rgb(attacked_forward)
        else:
            attacked_adjusted = attacked_forward

        ###########################    Benign attacks   ################################################################
        if skip_robust is None:
            skip_robust = np.random.rand() > self.opt['skip_attack_probability']
        if not skip_robust and self.opt['consider_robost']:

            attacked_image, attacked_real_jpeg_simulate, _ = self.benign_attacks(attacked_forward=attacked_adjusted,
                                                                                 quality_idx=quality_idx,
                                                                                 index=index_for_postprocessing,
                                                                                 kernel_size=kernel,
                                                                                 resize_ratio=resize_ratio
                                                                                 )
        else:
            attacked_image = attacked_adjusted

        return attacked_image, attacked_adjusted, masks, masks_GT

    ####################################################################################################
    # todo:  Momentum update of the key encoder
    # todo: param_k: momentum
    ####################################################################################################
    # @torch.no_grad()
    # def _momentum_update_key_encoder(self, momentum=0.9):
    #
    #
    #     for param_q, param_k in zip(self.discriminator_mask.parameters(), self.discriminator.parameters()):
    #         param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def visualize_raw(self, raw_to_raw_tensor, bayer_pattern, white_balance=None, eval=False):
        batch_size, height_width = raw_to_raw_tensor.shape[0], raw_to_raw_tensor.shape[2]
        # 两个相机都是RGGB
        # im = np.expand_dims(raw, axis=2)
        # if self.kernel_RAW_k0.ndim!=4:
        #     self.kernel_RAW_k0 = self.kernel_RAW_k0.unsqueeze(0).repeat(batch_size,1,1,1)
        #     self.kernel_RAW_k1 = self.kernel_RAW_k1.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #     self.kernel_RAW_k2 = self.kernel_RAW_k2.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #     self.kernel_RAW_k3 = self.kernel_RAW_k3.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #     print(f"visualize_raw is inited. Current shape {self.kernel_RAW_k0.shape}")
        out_tensor = None
        for idx in range(batch_size):

            used_kernel = getattr(self, f"kernel_RAW_k{bayer_pattern[idx]}")
            v_im = raw_to_raw_tensor[idx:idx + 1].repeat(1, 3, 1, 1) * used_kernel

            if white_balance is not None:
                # v_im (1,3,512,512) white_balance (1,3)
                # print(white_balance[idx:idx+1].unsqueeze(2).unsqueeze(3).shape)
                # print(v_im.shape)
                v_im = v_im * white_balance[idx:idx + 1].unsqueeze(2).unsqueeze(3)

            out_tensor = v_im if out_tensor is None else torch.cat((out_tensor, v_im), dim=0)

        return out_tensor.float()  # .half() if not eval else out_tensor

    def pack_raw(self, raw_to_raw_tensor):
        # 两个相机都是RGGB
        batch_size, num_channels, height_width = raw_to_raw_tensor.shape[0], raw_to_raw_tensor.shape[1], \
                                                 raw_to_raw_tensor.shape[2]

        H, W = raw_to_raw_tensor.shape[2], raw_to_raw_tensor.shape[3]
        R = raw_to_raw_tensor[:, :, 0:H:2, 0:W:2]
        Gr = raw_to_raw_tensor[:, :, 0:H:2, 1:W:2]
        Gb = raw_to_raw_tensor[:, :, 1:H:2, 0:W:2]
        B = raw_to_raw_tensor[:, :, 1:H:2, 1:W:2]
        G_avg = (Gr + Gb) / 2
        out = torch.cat((R, G_avg, B), dim=1)
        print(out.shape)
        out = Functional.interpolate(
            out,
            size=[height_width, height_width],
            mode='bilinear')
        return out

    def reload(self, pretrain, network_list=['netG', 'localizer'], strict=True):
        if 'netG' in network_list:
            load_path_G = pretrain + "_netG.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.netG, strict=strict)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'KD_JPEG' in network_list:
            load_path_G = pretrain + "_KD_JPEG.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.KD_JPEG, strict=strict)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'discriminator_mask' in network_list:
            load_path_G = pretrain + "_discriminator_mask.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.discriminator_mask, strict=strict)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'discriminator' in network_list:
            load_path_G = pretrain + "_discriminator.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.discriminator, strict=strict)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'qf_predict_network' in network_list:
            load_path_G = pretrain + "_qf_predict.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.qf_predict_network, strict=strict)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'localizer' in network_list:
            load_path_G = pretrain + "_localizer.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.localizer, strict=strict)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'generator' in network_list:
            load_path_G = pretrain + "_generator.pth"
            if load_path_G is not None:
                print('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.generator, strict=strict)
                else:
                    print('Did not find model for class [{:s}] ...'.format(load_path_G))

    def save(self, iter_label, folder='model', network_list=['netG', 'localizer']):
        if 'netG' in network_list:
            self.save_network(self.netG, 'netG', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'localizer' in network_list:
            self.save_network(self.localizer, 'localizer', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'KD_JPEG' in network_list:
            self.save_network(self.KD_JPEG, 'KD_JPEG', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'discriminator_mask' in network_list:
            self.save_network(self.discriminator_mask, 'discriminator_mask', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'discriminator' in network_list:
            self.save_network(self.discriminator, 'discriminator', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'qf_predict_network' in network_list:
            self.save_network(self.qf_predict_network, 'qf_predict', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')
        if 'generator' in network_list:
            self.save_network(self.generator, 'generator', iter_label if self.rank == 0 else 0,
                              model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')


if __name__ == '__main__':
    pass