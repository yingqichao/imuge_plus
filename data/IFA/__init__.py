import torch
import torch.utils.data
from data.data_sampler import DistIterSampler
import math

print("package data.IFA initialized")

def create_dataset(*, opt, args):
    ####################################################################################################
    # todo: TRAINING DATASET DEFINITION
    # todo: Define the training set
    ####################################################################################################
    # for phase, dataset_opt in opt['datasets'].items():
    dataset_opt = opt['datasets']['train']
    print("#################################### train set ####################################")
    print(dataset_opt)
    ## mode 0 1 2, 3
    if args.mode==0:
        print("dataset Defacto")
        from data.CASIA_dataset import CASIA_dataset as D
        train_set = D(opt, dataset_opt, is_train=True, dataset=["CASIA1","CASIA2"], attack_list=None, with_mask=False, with_au=True)
    else:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        train_set = D(opt, dataset_opt, load_mask=False)

    ####################################################################################################
    # todo: TEST DATASET DEFINITION
    # todo: Define the testing set
    ####################################################################################################
    val_dataset_opt = opt['datasets']['val']
    print('#################################### val set ####################################')
    print(val_dataset_opt)
    if args.mode==0:
        print("dataset Defacto")
        from data.CASIA_dataset import CASIA_dataset as D
        val_set = D(opt, val_dataset_opt, is_train=False, dataset=["CASIA1","CASIA2"], attack_list=None, with_mask=False, with_au=True)
    else:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        val_set = D(opt, val_dataset_opt, load_mask=False)

    return train_set, val_set
