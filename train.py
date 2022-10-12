import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"
import math
import argparse
import random
import logging
from loss import AdversarialLoss, PerceptualLoss, StyleLoss
from utils import Progbar, create_dir, stitch_images, imsave
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from torchvision import datasets, transforms
from skimage.feature import canny
from utils.util import var_name

####################################################################################################################
# todo: Please notice, in order to apply this framework. Remember to customize the following:
# todo: Dataset
# todo: Model
# todo: option yaml
# todo: bash script
#
# todo: Qichao Ying 20220420
####################################################################################################################

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # torch.cuda._initialized = True
    # torch.backends.cudnn.benchmark = True
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    print("world: {},rank: {},num_gpus:{}".format(world_size,rank,num_gpus))
    return world_size, rank

def main(args,opt):
    print(args)
    #### options
    # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] ="3,4"
    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    GT_size = opt['datasets']['train']['GT_size']
    #### distributed training settings
    # if args.launcher == 'none':  # disabled distributed training
    #     opt['dist'] = False
    #     rank = -1
    #     print('Disabled distributed training.')
    # else:
    print('Enables distributed training.')
    opt['dist'] = True
    world_size, rank = init_dist()

    #### mkdir and loggers

    # util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
    #              and 'pretrain_model' not in key and 'resume' not in key))
    # 
    # # config loggers. Before it, the log will not work
    # util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
    #                   screen=True, tofile=True)
    # util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
    #                   screen=True, tofile=True)
    # logger = logging.getLogger('base')
    # print(option.dict2str(opt))
    # tensorboard logger
    # if opt['use_tb_logger'] and 'debug' not in opt['name']:
    #     version = float(torch.__version__[0:3])
    #     if version >= 1.1:  # PyTorch 1.1
    #         from torch.utils.tensorboard import SummaryWriter
    #     else:
    #         # print(
    #         #     'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
    #         from tensorboardX import SummaryWriter
    #     tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    # if dist.get_rank()==0:
    # print(opt)

    ####################################################################################################
    # todo: SEED SELECTION
    # todo: The seeds are randomly shuffled each time. It is to prevent frequent debugging causing the model
    # todo: to remember the first several examples
    ####################################################################################################
    # seed = opt['train']['manual_seed']
    # if seed is None:
    # seed = random.randint(1, 1000)
    import time
    seed = int(time.time())%1000
    print('Random seed: {}'.format(seed))
    util.set_random_seed(seed)
    ## Slower but more reproducible
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    ## Faster but less reproducible
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    

    ####################################################################################################
    # todo: TRAINING DATASET DEFINITION
    # todo: Define the training set
    ####################################################################################################
    # for phase, dataset_opt in opt['datasets'].items():
    dataset_opt = opt['datasets']['train']
    print("#################################### train set ####################################")
    print(dataset_opt)
    train_set_1 = None
    if "PAMI" in opt['model'] or "CLR" in opt['model']:
        print("dataset with canny")
        from data.LQGT_dataset import LQGTDataset as D
        train_set = D(opt, dataset_opt)
    # elif "ICASSP_RHI" in opt['model']:
    #     print("dataset with jpeg")
    #     from data.tianchi_dataset import LQGTDataset as D
    #     train_set = D()
    #     # train_set = D(opt, dataset_opt)
    elif "ISP" in opt['model'] and args.mode!=1:
        print("dataset with ISP")
        from data.fivek_dataset import FiveKDataset_total
        with open("./data/camera.txt",'r') as t:
            camera_name = [i.strip() for i in t.readlines()]
        dataset_root = ['/ssd/FiveK_Dataset/'] * len(camera_name)
        # camera_name = ['Canon_EOS_5D','NIKON_D700']
        print(f'FiveK dataset size:{GT_size}')
        train_set = FiveKDataset_total(dataset_root, camera_name, stage='train', patch_size=GT_size)

        # dataset_root_1 = [ '/ssd/invISP_skip/']
        # camera_name_1 = [ 'NIKON_D700']
        # train_set_1 = FiveKDataset_skip(dataset_root_1, camera_name_1, stage='train', rgb_scale=False, uncond_p=0.,patch_size=GT_size)

        # from data.LQGT_dataset import LQGTDataset as D
        # train_set = D(opt, dataset_opt)
    elif "ISP" in opt['model'] and args.mode==1:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        train_set = D(opt, dataset_opt)

    else:
        raise NotImplementedError('大神是不是搞错了？')

    train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
    total_epochs = 100
    # train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
    world_size = torch.distributed.get_world_size()
    print("World size: {}".format(world_size))
    print("Batch size: {}".format(dataset_opt['batch_size']))
    num_workers = dataset_opt['n_workers']
    assert dataset_opt['batch_size'] % world_size == 0
    batch_size = dataset_opt['batch_size'] // world_size
    ####################################################################################################
    # todo: Data loader
    # todo:
    ####################################################################################################
    if opt['dist']:
        train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio,seed=seed)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                pin_memory=True)
    if rank <= 0:
        print('Number of train images: {:,d}, iters: {:,d}'.format(
            len(train_set), train_size))

    # if train_set_1 is not None:
    #     if opt['dist']:
    #         train_sampler_1 = DistIterSampler(train_set_1, world_size, rank, dataset_ratio, seed=seed)
    #     else:
    #         train_sampler_1 = None
    #     train_loader_1 = torch.utils.data.DataLoader(train_set_1, batch_size=batch_size, shuffle=False,
    #                                                num_workers=num_workers, sampler=train_sampler_1, drop_last=True,
    #                                                pin_memory=True)
    #     if rank <= 0:
    #         print('Number of train images: {:,d}, iters: {:,d}'.format(
    #             len(train_set_1), train_size))
    ####################################################################################################
    # todo: TEST DATASET DEFINITION
    # todo: Define the testing set
    ####################################################################################################
    # for phase, dataset_opt in opt['datasets'].items():
    dataset_opt = opt['datasets']['val']
    print('#################################### val set ####################################')
    print(dataset_opt)
    if "PAMI" in opt['model'] or "CLR" in opt['model']:
        print("dataset with canny")
        from data.LQGT_dataset import LQGTDataset as D
        val_set = D(opt, dataset_opt)
    # elif "ICASSP_RHI" in opt['model']:
    #     print("dataset with jpeg")
    #     from data.tianchi_dataset import LQGTDataset as D
    #     val_set = D()
    #     # train_set = D(opt, dataset_opt)
    elif "ISP" in opt['model'] and args.mode!=1:
        print("dataset with ISP")
        from data.fivek_dataset import FiveKDataset_total
        with open("./data/camera.txt", 'r') as t:
            camera_name = [i.strip() for i in t.readlines()]
        dataset_root = ['/ssd/FiveK_Dataset/'] * len(camera_name)
        # camera_name = ['Canon_EOS_5D','NIKON_D700']
        print(f'FiveK dataset size:{GT_size}')
        val_set = FiveKDataset_total(dataset_root, camera_name, stage='test', patch_size=GT_size)

        # from data.LQGT_dataset import LQGTDataset as D
        # val_set = D(opt, dataset_opt)
    elif "ISP" in opt['model'] and args.mode==1:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        val_set = D(opt, dataset_opt)
    else:
        raise NotImplementedError('大神是不是搞错了？')

    val_size = int(math.ceil(len(val_set) / 1))
    # val_loader = create_dataloader(val_set, dataset_opt, opt, val_sampler)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0,
                                pin_memory=True)

    if rank <= 0:
        print('Number of val images: {:,d}, iters: {:,d}'.format(
            len(val_set), val_size))
    

    ####################################################################################################
    # todo: MODEL DEFINITION
    # todo: Create the new model file
    ####################################################################################################
    # model = create_model(opt,args)
    which_model = opt['model']

    if which_model == 'CVPR':
        from models.Modified_invISP import IRNModel as M
        model = M(opt, args)
    elif which_model == 'PAMI':
        from models.IRNp_model import IRNpModel as M
        model = M(opt, args)
    elif which_model == 'ICASSP_NOWAY':
        from models.IRNcrop_model import IRNcropModel as M
    elif which_model == 'ICASSP_RHI':
        from models.tianchi_model import IRNrhiModel as M
        # from .IRNrhi_model import IRNrhiModel as M
    elif which_model == 'CLRNet':
        from models.IRNclrNew_model import IRNclrModel as M
        model = M(opt, args)
    elif which_model == 'ISP':
        from models.Modified_invISP import Modified_invISP as M
        model = M(opt, args, train_set)
    else:
        raise NotImplementedError('大神是不是搞错了？')

    print('Model [{:s}] is created.'.format(model.__class__.__name__))

    import time
    ####################################################################################################
    # todo: FUNCTIONALITIES
    # todo: we support a bunch of operations according to variables in val.
    # todo: each instance must implement a feed_data and optimize_parameters
    ####################################################################################################
    start_epoch, current_step = 0,  opt['train']['current_step']
    variables_list = []
    if ('CLRNet' in which_model or 'PAMI' in which_model or 'ISP' in which_model):
        if 'PAMI' in which_model:
            variables_list = []
        elif 'ISP' in which_model and args.mode==0:
            variables_list = ['RAW_L1', 'RAW_PSNR','loss','ERROR', 'CE','CEL1','F1','F1_1','RECALL','RECALL_1',
                              'RGB_PSNR_0','RGB_PSNR_1','RGB_PSNR_2']
        elif 'ISP' in which_model and args.mode==4:
            variables_list = ['ERROR', 'CE', 'F1', 'RECALL']
        elif 'ISP' in which_model and args.mode==0:
            variables_list = ['RAW_L1', 'RAW_PSNR','loss','ERROR', 'CE','CEL1','F1','F1_1','RECALL','RECALL_1',
                              'RGB_PSNR_0','RGB_PSNR_1','RGB_PSNR_2']
        elif 'ISP' in which_model and args.mode in [2,3,4]:
            variables_list = ['ISP_PSNR', 'ISP_L1', 'CE', 'CE_ema', 'CEL1', 'l1_ema', 'Mean', 'Std', 'CYCLE_PSNR',
                              'CYCLE_L1', 'PIPE_PSNR', 'PIPE_L1', 'loss',
                              'RAW_L1', 'RAW_PSNR', 'PSNR_DIFF', 'ISP_PSNR_NOW', 'ISP_SSIM_NOW', 'Percept', 'Gray', 'Style',
                              'ERROR', 'inpaint', 'inpaintPSNR'
                              ]
        elif 'ISP' in which_model and args.mode == 5:
            variables_list = ['CYCLE_PSNR', 'CYCLE_L1']
        elif 'CLRNet' in which_model:
            variables_list = ['loss', 'PF', 'PB', 'CE', 'SSFW', 'SSBK', 'lF', 'local']

        print(f"variables_list: {variables_list}")
        ####################################################################################################
        # todo: Training
        # todo: the training procedure should ONLY include progbar, feed_data and optimize_parameters so far
        ####################################################################################################
        total = len(train_set)
        if rank<=0:
            print('Start training from epoch: {:d}, iter: {:d}, total: {:d}'.format(start_epoch, current_step, total))
        latest_values = None

        print_step, restart_step = 10, 1000
        start = time.time()

        # train_generator_1 = iter(train_loader_1)
        val_generator = iter(val_loader)
        val_item = next(val_generator)
        model.feed_data_val_router(batch=val_item, mode=args.mode)
        for epoch in range(start_epoch, total_epochs + 1):
            current_step = 0

            # stateful_metrics = ['CK','RELOAD','ID','CEv_now','CEp_now','CE_now','STATE','lr','APEXGT','empty',
            #                     'SIMUL','RECON','RealTime'
            #                     'exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
            #                     'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']
            # if rank <= 0:
            #     progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            running_list = [0.0]*len(variables_list)
            valid_idx = 0
            # running_CE_MVSS, running_CE_mantra, running_CE_resfcn, valid_idx = 0.0, 0.0, 0.0, 0.0
            if opt['dist']:
                train_sampler.set_epoch(epoch)
            for idx, train_data in enumerate(train_loader):
                #### get item from another dataset
                # try:
                #     train_item_1 = next(train_generator_1)
                # except StopIteration as e:
                #     print("The end of val set is reached. Refreshing...")
                #     train_generator_1 = iter(train_loader_1)
                #     train_item_1 = next(train_generator_1)
                #### training
                model.feed_data_router(batch=train_data, mode=args.mode)

                logs, debug_logs, did_val = model.optimize_parameters_router(mode=args.mode, step=current_step)
                if did_val:
                    try:
                        val_item = next(val_generator)
                    except StopIteration as e:
                        print("The end of val set is reached. Refreshing...")
                        val_generator = iter(val_loader)
                        val_item = next(val_generator)
                    model.feed_data_val_router(batch=val_item, mode=args.mode)

                if variables_list[0] in logs or variables_list[1] in logs or variables_list[2] in logs:
                    for i in range(len(variables_list)):
                        if variables_list[i] in logs:
                            running_list[i] += logs[variables_list[i]]
                        # running_CE_mantra += logs['CE_mantra']
                        # running_CE_resfcn += logs['CE_resfcn']
                    valid_idx += 1
                else:
                    ## which is kind of abnormal, print
                    print(variables_list)
                if valid_idx>0 and (idx<10 or valid_idx % print_step == print_step - 1):  # print every 2000 mini-batches
                    # print(f'[{epoch + 1}, {valid_idx + 1} {rank}] '
                    #       f'running_CE_MVSS: {running_CE_MVSS / print_step:.2f} '
                    #       f'running_CE_mantra: {running_CE_mantra / print_step:.2f} '
                    #       f'running_CE_resfcn: {running_CE_resfcn / print_step:.2f} '
                    #       )
                    end = time.time()
                    lr = logs['lr']
                    info_str = f'[{epoch + 1}, {valid_idx + 1} {idx*model.real_H.shape[0]} {rank} {lr}] '
                    for i in range(len(variables_list)):
                        info_str += f'{variables_list[i]}: {running_list[i] / valid_idx:.4f} '
                    info_str += f'time per sample {(end-start)/print_step/model.real_H.shape[0]:.4f} s'
                    print(info_str)
                    start = time.time()
                    if valid_idx>=restart_step:
                        running_list = [0.0] * len(variables_list)
                        valid_idx = 0

                current_step += 1
                # if rank <= 0:
                #     progbar.add(len(model.real_H), values=logs)

    elif which_model == 'CLRNet' and args.mode==1.0:
        ####################################################################################################
        # todo: Eval
        # todo: the evaluation procedure should ONLY include evaluate so far
        ####################################################################################################
        if rank<=0:
            print('Start evaluating... ')
            if 'COCO' in opt['eval_dataset']:
                root = '/home/qcying/real_world_test_images'
            elif 'ILSVRC' in opt['eval_dataset']:
                root = '/home/qcying/real_world_test_images_ILSVRC'
            else:
                root = '/home/qcying/real_world_test_images_CelebA'
            data_origin = os.path.join(root,opt['eval_kind'],'ori_COCO_0114')
            data_immunize = os.path.join(root,opt['eval_kind'],'immu_COCO_0114')
            data_tampered = os.path.join(root,opt['eval_kind'],'tamper_COCO_0114')
            data_tampersource = os.path.join(root,opt['eval_kind'],'tamper_COCO_0114')
            data_mask = os.path.join(root,opt['eval_kind'],'binary_masks_COCO_0114')
            print(data_origin)
            print(data_immunize)
            print(data_tampered)
            print(data_tampersource)
            print(data_mask)
            model.evaluate(data_origin,data_immunize,data_tampered,data_tampersource,data_mask)

    elif which_model == 'CVPR' and args.mode==2.0:
        ####################################################################################################
        # todo: Training the KD-JPEG
        # todo: customized for PAMI, KD_JPEG_Generator_training
        ####################################################################################################
        if rank<=0:
            print('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
        latest_values = None
        total = len(train_set)
        for epoch in range(start_epoch, total_epochs + 1):
            stateful_metrics = ['CK','RELOAD','ID','CEv_now','CEp_now','CE_now','STATE','LOCAL','lr','APEXGT','empty',
                                'SIMUL','RECON',
                                'exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
                                'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']
            if rank <= 0:
                progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            if opt['dist']:
                train_sampler.set_epoch(epoch)
            for idx, train_data in enumerate(train_loader):
                current_step += 1
                #### training
                model.feed_data(train_data)

                logs, debug_logs = model.KD_JPEG_Generator_training(current_step,latest_values)
                if rank <= 0:
                    progbar.add(len(model.real_H), values=logs)

    else:
        raise NotImplementedError('大神是不是搞错了？')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-mode', type=int, default=0, help='validate or not.')
    parser.add_argument('-task_name', type=str, default="COCO_base", help='will determine the name of the folder.')
    parser.add_argument('-loading_from', type=str, default="COCO_base", help='loading checkpoints from?')
    parser.add_argument('-load_models', type=int, default=1, help='load checkpoint or not.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    main(args, opt)


