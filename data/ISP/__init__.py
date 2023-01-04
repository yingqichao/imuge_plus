import torch
import torch.utils.data
from data.data_sampler import DistIterSampler
import math

print("package data.ISP initialized")

def create_dataset(*, opt, args):
    ####################################################################################################
    # todo: TRAINING DATASET DEFINITION
    # todo: Define the training set
    ####################################################################################################
    # for phase, dataset_opt in opt['datasets'].items():
    dataset_opt = opt['datasets']['train']
    print("#################################### train set ####################################")
    print(dataset_opt)

    if "ISP" in opt['model'] and args.mode in [1]:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        train_set = D(opt, dataset_opt)
    elif "ISP" in opt['model'] and args.mode in [3]:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        train_set = D(opt, dataset_opt, load_mask=False)
    elif ("ISP" in opt['model'] and args.mode in [9]):
        print("dataset Defacto")
        from data.CASIA_dataset import CASIA_dataset as D
        train_set = D(opt, dataset_opt, is_train=True, dataset="Defacto", attack_list={0, 1, 2})
    elif "ISP" in opt['model'] and args.mode in opt['using_RAW_dataset_training']:
        # print("dataset with ISP")
        # from data.fivek_dataset import FiveKDataset_total
        # with open("./data/camera.txt", 'r') as t:
        #     camera_name = [i.strip() for i in t.readlines()]
        # dataset_root = ['/ssd/FiveK_Dataset/'] * len(camera_name)
        # # camera_name = ['Canon_EOS_5D','NIKON_D700']
        # print(f'FiveK dataset size')
        # train_set = FiveKDataset_total(dataset_root, camera_name, stage='train', patch_size=GT_size)

        from data.RAISE import Raise, FiveKTest
        if "Raise" in opt['using_which_dataset_for_training']:
            print('wanna to test RAISE dataset')
            data_root = '/groupshare/raise_crop'
            stage = 'crop_train'  # crop_train crop_test
            train_set = Raise(data_root, stage=stage)
        elif "FiveK" in opt['using_which_dataset_for_training']:
            print('wanna to test CANON dataset')
            data_root = '/ssd/FiveK_train'
            camera_name = 'Canon' if 'Canon' in opt['using_which_dataset_for_training'] else 'NIKON'
            stage = 'train'
            train_set = FiveKTest(data_root, camera_name, stage=stage)
        else:
            raise NotImplementedError("ISP的数据集定义不对，请检查！")
    else:
        raise NotImplementedError("ISP的数据集定义不对，请检查！")

    ####################################################################################################
    # todo: TEST DATASET DEFINITION
    # todo: Define the testing set
    ####################################################################################################
    val_dataset_opt = opt['datasets']['val']
    print('#################################### val set ####################################')
    print(val_dataset_opt)
    if "ISP" in opt['model'] and args.mode in [1]:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        val_set = D(opt, val_dataset_opt)

    elif "ISP" in opt['model'] and args.mode in [3]:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        val_set = D(opt, val_dataset_opt, load_mask=False)
    elif "ISP" in opt['model'] and args.mode in [9]:
        print("dataset CASIA1")
        from data.CASIA_dataset import CASIA_dataset as D
        val_set = D(opt, val_dataset_opt, is_train=False, dataset="Defacto", attack_list={0, 1, 2})
    elif "ISP" in opt['model'] and args.mode in opt['using_RAW_dataset_testing']:
        # print("dataset with ISP")
        # from data.fivek_dataset import FiveKDataset_total
        # with open("./data/camera.txt", 'r') as t:
        #     camera_name = [i.strip() for i in t.readlines()]
        # dataset_root = ['/ssd/FiveK_Dataset/'] * len(camera_name)
        # # camera_name = ['Canon_EOS_5D','NIKON_D700']
        # print(f'FiveK dataset size:{GT_size}')
        # val_set = FiveKDataset_total(dataset_root, camera_name, stage='test', patch_size=GT_size)
        val_dataset_name = opt['using_which_dataset_for_test']
        if 'Raise' in val_dataset_name:
            from data.RAISE import Raise
            print('wanna to test RAISE dataset')
            data_root = '/groupshare/raise_crop'
            stage = 'crop_test'  # crop_train crop_test
            val_set = Raise(data_root, stage=stage)

        elif 'FiveK' in val_dataset_name:
            stage = 'Canon' if 'Canon' in val_dataset_name else 'NIKON'
            from data.RAISE import FiveKTest
            print(f'wanna to test {val_dataset_name} dataset')
            data_root = '/ssd/FiveK_test'
            val_set = FiveKTest(data_root, stage)

        elif 'SIDD' in val_dataset_name:
            from data.sidd import SIDD
            val_set = SIDD('/groupshare/SIDD_xxhu/', 'meta.pickle', use_skip=True)

        else:
            raise NotImplementedError('数据集搞错拉！')

    else:
        raise NotImplementedError('数据集搞错拉！')

    # from data.dnd import DND
    # val_set = DND('/groupshare/dnd_raw/', 'data/dnd.pickle')

    # from data.sr_raw import SrRaw
    # data_root = '/groupshare/sr_raw/train0/'
    # val_set = SrRaw(data_root)

    # from data.LQGT_dataset import LQGTDataset as D
    # val_set = D(opt, val_dataset_opt)

    return train_set, val_set
