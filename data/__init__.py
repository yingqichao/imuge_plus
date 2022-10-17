'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None, batch_size=None):
    phase = opt['phase']
    print("MODE: {}".format(phase))
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            print("World size: {}".format(world_size))
            print("Batch size: {}".format(dataset_opt['batch_size']))
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


from data.data_sampler import DistIterSampler
import math
def create_dataset(*, opt,args,rank,seed):
    ####################################################################################################
    # todo: TRAINING DATASET DEFINITION
    # todo: Define the training set
    ####################################################################################################
    # for phase, dataset_opt in opt['datasets'].items():
    dataset_opt = opt['datasets']['train']
    GT_size = opt['datasets']['train']['GT_size']
    print("#################################### train set ####################################")
    print(dataset_opt)
    if "PAMI" in opt['model'] or "CLR" in opt['model']:
        print("dataset with canny")
        from data.LQGT_dataset import LQGTDataset as D
        train_set = D(opt, dataset_opt)
    elif "ISP" in opt['model'] and args.mode == 1:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        train_set = D(opt, dataset_opt)
    elif "ISP" in opt['model'] and args.mode in [3, 6]:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        train_set = D(opt, dataset_opt, load_mask=False)
    elif "ISP" in opt['model'] and args.mode != 1:
        print("dataset with ISP")
        from data.fivek_dataset import FiveKDataset_total
        with open("./data/camera.txt", 'r') as t:
            camera_name = [i.strip() for i in t.readlines()]
        dataset_root = ['/ssd/FiveK_Dataset/'] * len(camera_name)
        # camera_name = ['Canon_EOS_5D','NIKON_D700']
        print(f'FiveK dataset size')
        train_set = FiveKDataset_total(dataset_root, camera_name, stage='train', patch_size=GT_size)

        # dataset_root_1 = [ '/ssd/invISP_skip/']
        # camera_name_1 = [ 'NIKON_D700']
        # train_set_1 = FiveKDataset_skip(dataset_root_1, camera_name_1, stage='train', rgb_scale=False, uncond_p=0.,patch_size=GT_size)

        # from data.LQGT_dataset import LQGTDataset as D
        # train_set = D(opt, dataset_opt)
    else:
        raise NotImplementedError('大神是不是搞错了？')

    train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
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
    dataset_ratio = 200  # enlarge the size of each epoch
    if opt['dist']:
        train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio, seed=seed)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                               pin_memory=True)

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
    elif "ISP" in opt['model'] and args.mode == 1:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        val_set = D(opt, dataset_opt)
    elif "ISP" in opt['model'] and args.mode in [3, 6]:
        print("dataset LQ")
        from data.LQ_dataset import LQDataset as D
        val_set = D(opt, dataset_opt, load_mask=False)
    elif "ISP" in opt['model'] and args.mode != 1:
        print("dataset with ISP")
        from data.fivek_dataset import FiveKDataset_total
        with open("./data/camera.txt", 'r') as t:
            camera_name = [i.strip() for i in t.readlines()]
        dataset_root = ['/ssd/FiveK_Dataset/'] * len(camera_name)
        # camera_name = ['Canon_EOS_5D','NIKON_D700']
        print(f'FiveK dataset size:{GT_size}')
        val_set = FiveKDataset_total(dataset_root, camera_name, stage='test', patch_size=GT_size)

        # from data.sidd import SIDD
        # val_set = SIDD('/groupshare/SIDD_xxhu/', 'meta.pickle', use_skip=True)

        # from data.dnd import DND
        # val_set = DND('/groupshare/dnd_raw/', 'data/dnd.pickle')

        # from data.sr_raw import SrRaw
        # data_root = '/groupshare/sr_raw/train0/'
        # val_set = SrRaw(data_root)

        # from data.LQGT_dataset import LQGTDataset as D
        # val_set = D(opt, dataset_opt)
    else:
        raise NotImplementedError('大神是不是搞错了？')

    val_size = int(math.ceil(len(val_set) / 1))
    # val_loader = create_dataloader(val_set, dataset_opt, opt, val_sampler)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0,
                                             pin_memory=True)

    if rank <= 0:
        print('Number of val images: {:,d}, iters: {:,d}'.format(
            len(val_set), val_size))


    return train_set, val_set, train_sampler, train_loader, val_loader
