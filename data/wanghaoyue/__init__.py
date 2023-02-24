import torch
import torch.utils.data
from data.data_sampler import DistIterSampler
import math

print("package data.tianchi initialized")

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

    print("dataset tianchi")
    from data.Tianchi.Tianchi_dataset import TianchiDataset as D
    train_set = D(opt, dataset_opt, is_train=True)

    ####################################################################################################
    # todo: TEST DATASET DEFINITION
    # todo: Define the testing set
    ####################################################################################################
    val_dataset_opt = opt['datasets']['val']
    print('#################################### val set ####################################')
    print(val_dataset_opt)
    from data.Tianchi.Tianchi_dataset import TianchiDataset as D
    val_set = D(opt, dataset_opt, is_train=False, with_mask=False)

    return train_set, val_set
