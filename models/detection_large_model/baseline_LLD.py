import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from utils import stitch_images
from models.detection_large_model.base_LLD import BaseLLD
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from losses.focal_loss import focal_loss
from losses.dice_loss import DiceLoss
from losses.F1_score_loss import SoftF1Loss
# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"
import os
from models.ISP.Modified_invISP import Modified_invISP
from large_models.models_mae import mae_vit_base_patch16_dec512d8b
class baseline_LLD(BaseLLD):
    def __init__(self, opt, args, train_loader=None, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = opt
        self.args = args
        self.history_accuracy = 0.2
        # self.methodology = opt['methodology']
        self.focal_loss = focal_loss(alpha=self.opt['focal_alpha']).cuda()
        self.dice_loss = DiceLoss().cuda()
        self.f1_loss = SoftF1Loss().cuda()

        """
            prepare_networks_optimizers: set True current, preserved for future uses that only invoke static methods without creating an instances.

        """
        super(baseline_LLD, self).__init__(opt, args)
        ### todo: 定义分类网络
        self.detection_model = self.define_CATNET()
        self.mae_model = mae_vit_base_patch16_dec512d8b().cuda()
        self.mae_model = DistributedDataParallel(self.mae_model,
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        ## todo: 加载之前的模型
        if self.opt['model_load_number'] is not None:
            self.reload(
                pretrain=f"{self.out_space_storage}/models/{self.args.task_name}/{self.opt['model_load_number']}",
                network=self.segmentation_model,
                strict=False)

        self.criterion = nn.CrossEntropyLoss()
        self.train_opt = opt['train']

        # self.optimizer = Adam(self.segmentation_model.parameters(), lr=lr)
        ## todo: optimizer和scheduler
        # wd_G = 1e-5
        # self.optimizer = self.create_optimizer(self.segmentation_model,
        #                                           lr=self.train_opt['lr_scratch'], weight_decay=wd_G)
        optim_params_CNN, optim_params_trans = [], []
        name_params_CNN, name_params_trans = [], []
        for k, v in self.segmentation_model.named_parameters():
            if v.requires_grad:

                name_params_CNN.append(k)
                optim_params_CNN.append(v)


        self.optimizer_CNN = torch.optim.SGD(optim_params_CNN,
                                               lr=self.opt['train']['lr_CNN'])

        self.optimizers.append(self.optimizer_CNN)

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


    def feed_data_router(self, *, batch, mode):
        img, GT_path, mask, mask_path = batch
        self.img = img.cuda()
        self.mask = mask.unsqueeze(1).cuda()
        self.GT_path = GT_path
        self.mask_path = mask_path

    def feed_data_val_router(self, *, batch, mode):
        img, GT_path = batch
        self.img_val = img.cuda()
        self.GT_val_path = GT_path

    def train_LLD(self, *, epoch=None, step=None):
        ## todo:具体实现你的功能
        self.detection_model.train()
        self.mae_model.eval()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr
        with torch.enable_grad():
            feature_mae = self.mae_model(self.img)
            feature_detection = self.detection_model(self.img)
            ## todo: distill loss
            distill_loss = self.customized_loss(input=feature_detection, target=feature_mae)

            pred_mask = F.interpolate(pred_mask, size=(self.mask.shape[2], self.mask.shape[3]), mode='bilinear')
            # x_stage_2 = F.interpolate(x_stage_2, size=(self.mask.shape[2], self.mask.shape[3]), mode='bilinear')
            # x_stage_3 = F.interpolate(x_stage_3, size=(self.mask.shape[2], self.mask.shape[3]), mode='bilinear')

            loss = 0
            ### todo: focal
            loss_focal = self.focal_loss.forward_segment(pred_mask,self.mask.long())
            loss += loss_focal
            logs['loss_focal'] = loss_focal.item()
            ### todo: emsemble
            pred_mask_emsemble = pred_mask  # + x_stage_2 + x_stage_3
            pred_mask_emsemble = torch.softmax(pred_mask_emsemble, dim=1)[:, 1:2]
            loss_bce = self.bce_loss(pred_mask_emsemble, self.mask)
            logs['loss_bce'] = loss_bce.item()
            ### todo: dice
            loss_dice = self.f1_loss(predict=pred_mask_emsemble, target=self.mask)
            # loss += loss_dice
            logs['loss_F1'] = loss_dice.item()
            ### todo: auxiliary_losses
            # loss_seg_stage_2 = self.focal_loss.forward_segment(x_stage_2, self.mask.long())
            # loss += loss_seg_stage_2
            # logs['loss_stage2'] = loss_seg_stage_2.item()
            # loss_seg_stage_3 = self.focal_loss.forward_segment(x_stage_3, self.mask.long())
            # loss += loss_seg_stage_3
            # logs['loss_stage3'] = loss_seg_stage_3.item()


            loss.backward()
            # if self.global_step%4==3:
            nn.utils.clip_grad_norm_(self.segmentation_model.parameters(), 1)
            self.optimizer_CNN.step()
            # self.optimizer_trans.step()
            self.optimizer_CNN.zero_grad()
            # self.optimizer_trans.zero_grad()

            # emsemble_prediction = 0
            # for idx, item in enumerate(output):
            #     name_acc = ['CNN', 'trans']
            #     emsemble_prediction += item[0]
            #     acc = (item[0].argmax(dim=1) == self.label).float().mean()
            #     logs[f'{name_acc[idx]}_acc'] = acc
            # acc = (emsemble_prediction.argmax(dim=1) == self.label).float().mean()
            # logs['overall_acc'] = acc
            logs['loss'] = loss.item()
            # logs['alpha'] = alpha.item()
            # logs['beta'] = beta.item()
            # logs['loss_class'] = loss_class.item()
            # logs['loss_mag'] = loss_mag.item()

        ######## Finally ####################
        # for scheduler in self.schedulers:
        #     scheduler.step()
        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            for image_no in range(self.img.shape[0]):
                filename = f"{self.out_space_storage}/images/{self.task_name}/{epoch}_{self.global_step}_ori_{self.rank}.png"
                # print(f"image saved at: {filename}")
                self.print_this_image(image=self.img[image_no], filename=filename[:filename.rfind('.')] + ".png")

                filename = f"{self.out_space_storage}/images/{self.task_name}/{epoch}_{self.global_step}_pred_{self.rank}.png"
                # print(f"image saved at: {filename}")
                self.print_this_image(image=pred_mask_emsemble[image_no], filename=filename[:filename.rfind('.')] + ".png")

                filename = f"{self.out_space_storage}/images/{self.task_name}/{epoch}_{self.global_step}_gt_{self.rank}.png"
                # print(f"image saved at: {filename}")
                self.print_this_image(image=self.mask[image_no], filename=filename[:filename.rfind('.')] + ".png")
            # images = stitch_images(
            #     self.postprocess(self.img),
            #     self.postprocess(self.mask),
            #     self.postprocess(pred_mask),
            #     img_per_row=1
            # )
            #
            # name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
            #        f"_{str(self.rank)}.png"
            # images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9:
            if self.rank == 0:
                print('Saving models and training states.')
                self.save(accuracy=epoch, iter_label=step, network_arch=self.opt["network_arch"])

        self.global_step = self.global_step + 1

        self.global_step = self.global_step + 1

        return logs

    def test_tianchi(self, *, epoch=None, step=None):
        self.segmentation_model.eval()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr
        with torch.no_grad():
            pred_mask = self.segmentation_model(self.img_val, qtable=None)
            pred_mask = F.interpolate(pred_mask, size=(self.img_val.shape[2], self.img_val.shape[3]), mode='bilinear')
            pred_mask = torch.softmax(pred_mask,dim=1)[:,1:2]

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.img_val),
                self.postprocess(pred_mask),
                img_per_row=1
            )

            filename = f"{self.out_space_storage}/observe/{self.task_name}/{self.global_step}.png"
            images.save(filename)

        ######## Finally ####################
        for image_no in range(self.img_val.shape[0]):

            filename = f"{self.out_space_storage}/submission/{self.task_name}/{self.GT_val_path[image_no]}"
            print(f"image saved at: {filename}")
            self.print_this_image(image=pred_mask[image_no], filename=filename[:filename.rfind('.')]+".png")


        self.global_step = self.global_step + 1

        return logs

    def save(self, *, accuracy, iter_label, network_arch):
        self.save_network(
            pretrain=f"{self.out_space_storage}/models/{self.args.task_name}/{accuracy}_{iter_label}_{network_arch}.pth",
            network=self.segmentation_model,
        )

    # def model_save(self, *, path, epochs):
    #     checkpoint = {'models': self.segmentation_model,
    #                   'model_state_dict': self.segmentation_model.state_dict(),
    #                   'optimizer_state_dict': self.optimizer.state_dict(),
    #                   'epochs': epochs,
    #                   # 'epoch_acc_list': epoch_acc_list,
    #                   # 'epoch_loss_list': epoch_loss_list}
    #                   }
    #     torch.save(checkpoint, f'{path}.pkl')

    # def model_load(self):
    #     from util import load_checkpoint
    #     self.segmentation_model, result = load_checkpoint('checkpoint.pkl')
    #     acc = result['epoch_acc_list']
    #     plt.plot(acc)
    #     loss_list = result['epoch_loss_list']
    #     plt.plot(loss_list)

    # # todo: 预测k张图片
    # k = 3
    # sample_idx = np.random.randint(1, len(test_list), size=k)
    # sample_test_list = [test_list[i] for i in sample_idx]
    # sample_test_labels = [test_labels[i] for i in sample_idx]
    # sample_test_data = meiyanDataset(sample_test_list, sample_test_labels, transform=test_transforms)
    # sample_test_loader = DataLoader(dataset=sample_test_data, batch_size=1, shuffle=True)
    #
    # models.eval()
    # i = 0
    # for data, label in tqdm(sample_test_loader):
    #     data = data.cuda()
    #     label = label.cuda()
    #
    #     output = models(data)  # 此时输出是11维，选择概率最大的
    #     pred = output.argmax(dim=1).cpu().numpy()
    #     img = Image.open(test_list[sample_idx[i]])
    #     i += 1
    #     plt.subplot(1, k, i)
    #     title = "label:" + str(label.cpu().numpy()[0]) + ",pred:" + str(pred[0])
    #     plt.title(title)
    #     plt.imshow(img)



