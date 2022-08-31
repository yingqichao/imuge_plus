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
    #### options
    # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] ="3,4"
    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    # resize = opt['datasets']['train']['GT_size']
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

    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                 and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    # logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        version = float(torch.__version__[0:3])
        if version >= 1.1:  # PyTorch 1.1
            from torch.utils.tensorboard import SummaryWriter
        else:
            logger.info(
                'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
            from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    # if dist.get_rank()==0:
    print(opt)

    ####################################################################################################
    # SEED SELECTION
    # todo: The seeds are randomly shuffled each time. It is to prevent frequent debugging causing the model
    # todo: to remember the first several examples
    ####################################################################################################
    # seed = opt['train']['manual_seed']
    # if seed is None:
    seed = random.randint(1, 100)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)
    ## Slower but more reproducible
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    ## Faster but less reproducible
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    ####################################################################################################
    ## todo: END OF DEFINITION
    ####################################################################################################

    ####################################################################################################
    # todo: DATASET DEFINITION
    # todo: Define the training/test set
    ####################################################################################################
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':

            if "PAMI" in opt['model'] or "CLR" in opt['model']:
                print("dataset with canny")
                from data.LQGT_dataset import LQGTDataset as D
                train_set = D(opt, dataset_opt)
            elif "ICASSP_RHI" in opt['model']:
                print("dataset with jpeg")
                from data.tianchi_dataset import LQGTDataset as D
                train_set = D()
                # train_set = D(opt, dataset_opt)
            elif "diffusion" in opt['model']:
                print("Using FiveK dataset")
                from data.qian_rumor_dataset import QianDataset as D
                train_set = D(opt, dataset_opt)
            else:
                raise NotImplementedError("Dataset Not Implemented. Exit.")

            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = 100
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio,seed=seed)
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
    ####################################################################################################
    ## todo: END OF DEFINITION
    ####################################################################################################

    ####################################################################################################
    # todo: MODEL DEFINITION
    # todo: Create the new model file
    ####################################################################################################
    # model = create_model(opt,args)
    model = opt['model']

    if model == 'CVPR':
        from models.Modified_invISP import IRNModel as M
        model = M(opt, args)
    elif model == 'PAMI':
        from models.IRNp_model import IRNpModel as M
        model = M(opt, args)
    elif model == 'ICASSP_NOWAY':
        from models.IRNcrop_model import IRNcropModel as M
    elif model == 'ICASSP_RHI':
        from models.tianchi_model import IRNrhiModel as M
        # from .IRNrhi_model import IRNrhiModel as M
    elif model == 'CLRNet':
        from models.IRNclrNew_model import IRNclrModel as M
        model = M(opt, args)
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))

    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))
    ####################################################################################################
    ## todo: END OF DEFINITION
    ####################################################################################################

    ####################################################################################################
    # todo: FUNCTIONALITIES
    # todo: we support a bunch of operations according to variables in val.
    ####################################################################################################
    start_epoch = 0
    current_step = opt['train']['current_step']

    if args.val==0.0:
        ####################################################################################################
        # todo: Training
        # todo: the training procedure should ONLY include progbar, feed_data and optimize_parameters so far
        ####################################################################################################
        if rank<=0:
            logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
        latest_values = None
        total = len(train_set)
        for epoch in range(start_epoch, total_epochs + 1):
            stateful_metrics = ['CK','RELOAD','ID','CEv_now','CEp_now','CE_now','STATE','lr','APEXGT','empty',
                                'SIMUL','RECON','RealTime'
                                'exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
                                'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']
            if rank <= 0:
                progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            if opt['dist']:
                train_sampler.set_epoch(epoch)
            for idx, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > total_iters:
                    break
                #### training
                model.feed_data(train_data)

                logs, debug_logs = model.optimize_parameters(current_step,latest_values)
                if rank <= 0:
                    progbar.add(len(model.real_H), values=logs)
        ####################################################################################################
        ## todo: END OF DEFINITION
        ####################################################################################################
    elif args.val==1.0:
        ####################################################################################################
        # todo: Eval
        # todo: the evaluation procedure should ONLY include evaluate so far
        ####################################################################################################
        if rank<=0:
            logger.info('Start evaluating... ')
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
        ####################################################################################################
        ## todo: END OF DEFINITION
        ####################################################################################################
    elif args.val==2.0:
        ####################################################################################################
        # todo: Training the KD-JPEG
        # todo: customized for PAMI, KD_JPEG_Generator_training
        ####################################################################################################
        if rank<=0:
            logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
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
                if current_step > total_iters:
                    break
                #### training
                model.feed_data(train_data)

                logs, debug_logs = model.KD_JPEG_Generator_training(current_step,latest_values)
                if rank <= 0:
                    progbar.add(len(model.real_H), values=logs)
        ####################################################################################################
        ## todo: END OF DEFINITION
        ####################################################################################################
    elif args.val==3.0:
        ####################################################################################################
        # todo: ISP protection
        # todo: the training procedure should ONLY include progbar, feed_data and optimize_parameters so far
        ####################################################################################################
        if rank<=0:
            logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
        latest_values = None
        total = len(train_set)
        for epoch in range(start_epoch, total_epochs + 1):
            stateful_metrics = ['CK','RELOAD','ID','CEv_now','CEp_now','CE_now','STATE','lr','APEXGT','empty',
                                'SIMUL','RECON','RealTime'
                                'exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
                                'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']
            if rank <= 0:
                progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            if opt['dist']:
                train_sampler.set_epoch(epoch)
            for idx, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > total_iters:
                    break
                #### training
                model.feed_data(train_data)

                logs, debug_logs = model.optimize_parameters(current_step,latest_values)
                if rank <= 0:
                    progbar.add(len(model.real_H), values=logs)
        ####################################################################################################
        ## todo: END OF DEFINITION
        ####################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-val', type=int, default=0, help='validate or not.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    main(args,opt)


