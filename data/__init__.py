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


def create_dataset(opt,dataset_opt):
    mode = dataset_opt['mode']

    from data.LQGT_dataset import LQGTDataset as D
    dataset = D(opt,dataset_opt)


    return dataset
