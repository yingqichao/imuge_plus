import torch
import torch.utils.data
from data.data_sampler import DistIterSampler
import math

print("package data.PAMI initialized")

def create_dataset(*, opt, args):
    ####################################################################################################
    # todo: TRAINING DATASET DEFINITION
    # todo: Define the training set
    ####################################################################################################
    # for phase, dataset_opt in opt['datasets'].items():
    dataset_opt = opt['datasets']['train']
    print("#################################### train set ####################################")
    print(dataset_opt)

    print("dataset with canny")
    from data.LQGT_dataset import LQGTDataset as D
    train_set = D(opt, dataset_opt)

    ####################################################################################################
    # todo: TEST DATASET DEFINITION
    # todo: Define the testing set
    ####################################################################################################
    val_dataset_opt = opt['datasets']['val']
    print('#################################### val set ####################################')
    print(val_dataset_opt)
    print("dataset with canny")
    from data.LQGT_dataset import LQGTDataset as D
    val_set = D(opt, val_dataset_opt, is_train=False)

    return train_set, val_set
