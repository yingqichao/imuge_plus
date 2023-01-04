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
def create_dataset_and_loader(*, opt,args,rank,seed):

    ####################################################################################################
    # todo: TRAINING AND TESTING DATASET DEFINITION
    # todo: Define the training set
    ####################################################################################################
    # for phase, dataset_opt in opt['datasets'].items():
    dataset_opt = opt['datasets']['train']
    val_dataset_opt = opt['datasets']['val']
    world_size = torch.distributed.get_world_size()
    print("World size: {}".format(world_size))
    print("Batch size: {}".format(dataset_opt['batch_size']))
    num_workers = dataset_opt['n_workers']
    assert dataset_opt['batch_size'] % world_size == 0
    batch_size = dataset_opt['batch_size'] #// world_size
    print("#################################### train set ####################################")
    print(dataset_opt)
    if "PAMI" in opt['model'] or "CLR" in opt['model']:
        from data.PAMI import create_dataset
        train_set, val_set = create_dataset(opt=opt, args=args)
    elif "IFA" in opt['model']:
        from data.IFA import create_dataset
        train_set, val_set = create_dataset(opt=opt, args=args)
    elif "ISP" in opt['model']:
        from data.ISP import create_dataset
        train_set, val_set = create_dataset(opt=opt, args=args)
    else:
        raise NotImplementedError('大神是不是搞错了？')

    train_size = int(math.ceil(len(train_set) / batch_size))
    ####################################################################################################
    # todo: Data loader
    ####################################################################################################
    dataset_ratio = 1 # world_size  # enlarge the size of each epoch
    if opt['dist']:
        train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio, seed=seed)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                               pin_memory=True)

    print('Number of train images: {:,d}, iters: {:,d}'.format(
        len(train_set), train_size))


    # val_loader = create_dataloader(val_set, dataset_opt, opt, val_sampler)
    val_batch_size = 1 if 'batch_size' not in val_dataset_opt else val_dataset_opt['batch_size'] // world_size
    val_num_workers = 1 if 'batch_size' not in val_dataset_opt else num_workers
    val_size = int(math.ceil(len(val_set) / val_batch_size))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers,
                                             pin_memory=True)

    print('Number of val images: {:,d}, iters: {:,d}'.format(
        len(val_set), val_size))


    return train_set, val_set, train_sampler, train_loader, val_loader
