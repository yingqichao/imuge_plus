import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import options.options as option
from utils import util

import time

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
    # import logging
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
    # # tensorboard logger
    # if opt['use_tb_logger'] and 'debug' not in opt['name']:
    #     version = float(torch.__version__[0:3])
    #     if version >= 1.1:  # PyTorch 1.1
    #         from torch.utils.tensorboard import SummaryWriter
    #     else:
    #         # print(
    #         #     'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
    #         from tensorboardX import SummaryWriter
    #     tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    #
    # # convert to NoneDict, which returns None for missing keys
    # opt = option.dict_to_nonedict(opt)
    # # if dist.get_rank()==0:
    # # print(opt)

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
    # todo: DATASET AND NETWORK DEFINITION
    ####################################################################################################
    which_model = opt['model']
    print(f"Model name: {which_model}")
    from data import create_dataset_and_loader
    from models import create_models
    ## data
    train_set, val_set, train_sampler, train_loader, val_loader = create_dataset_and_loader(opt=opt, args=args, rank=rank, seed=seed)
    model = create_models(opt=opt,args=args, train_set=train_set, val_set=val_set)

    from train_and_test_scripts import scripts_router
    scripts_router(which_model=which_model, opt=opt, args=args, rank=rank, model=model,
                              train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler)

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
    opt = option.parse(opt_path=args.opt,
                        args=args)

    main(args, opt)
