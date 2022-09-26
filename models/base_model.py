import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import numpy as np

class BaseModel():
    def __init__(self, opt,  args, train_set):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.optimizer_G = None
        self.optimizer_localizer = None
        self.optimizer_discriminator_mask = None
        self.optimizer_discriminator = None
        self.optimizer_KD_JPEG = None
        self.optimizer_generator = None
        self.optimizer_qf_predict = None
        self.netG = None
        self.localizer = None
        self.discriminator = None
        self.global_step = 0
        self.updated_step = 0
        ####################################################################################################
        # todo: constants
        ####################################################################################################
        self.width_height = opt['datasets']['train']['GT_size']
        self.kernel_RAW_k0 = torch.tensor([[[1, 0], [0, 0]],[[0, 1], [1, 0]],[[0, 0], [0, 1]]],device="cuda",
                                          requires_grad=False).unsqueeze(0)
        self.kernel_RAW_k1 = torch.tensor([[[0, 0], [1, 0]],[[1, 0], [0, 1]],[[0, 1], [0, 0]]],device="cuda",
                                          requires_grad=False).unsqueeze(0)
        self.kernel_RAW_k2 = torch.tensor([[[0, 0], [0, 1]],[[0, 1], [1, 0]],[[1, 0], [0, 0]]],device="cuda",
                                          requires_grad=False).unsqueeze(0)
        self.kernel_RAW_k3 = torch.tensor([[[0, 1], [0, 0]],[[0, 1], [1, 0]],[[0, 0], [1, 0]]],device="cuda",
                                          requires_grad=False)
        expand_times = int(self.width_height//2)
        self.kernel_RAW_k0 = self.kernel_RAW_k0.repeat(1, 1, expand_times,expand_times)
        self.kernel_RAW_k1 = self.kernel_RAW_k1.repeat(1, 1, expand_times, expand_times)
        self.kernel_RAW_k2 = self.kernel_RAW_k2.repeat(1, 1, expand_times, expand_times)
        self.kernel_RAW_k3 = self.kernel_RAW_k3.repeat(1, 1, expand_times, expand_times)



        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

    def using_invISP(self):
        return self.global_step % 4 == 0
    def using_cycleISP(self):
        return self.global_step % 4 == 1
    def using_my_own_pipeline(self):
        return self.global_step % 4 == 2
    def using_weak_jpeg_plus_blurring_etc(self):
        return self.global_step % 5 in {0, 1, 2}
    def using_simulated_inpainting(self):
        return self.global_step % 2 == 0
    def using_splicing(self):
        return False #self.global_step % 3 == 0
    def using_copy_move(self):
        return self.global_step % 2 == 1
    def using_gaussian_blur(self):
        return self.global_step % 5 == 1
    def using_median_blur(self):
        return self.global_step % 5 == 2
    def using_resizing(self):
        return self.global_step % 5 == 0
    def using_jpeg_simulation_only(self):
        return self.global_step % 5 == 4
    def begin_using_momentum(self):
        return self.global_step>=0



    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label,save_dir=None, model_path=None):
        # save_dir = '../experiments/pretrained_models/'
        if model_path == None:
            model_path = self.opt['path']['models']
        if save_dir is None:
            save_filename = '{}_{}.pth'.format(iter_label, network_label)
            save_path = os.path.join(model_path, save_filename)
        else:
            save_filename = '{}_latest.pth'.format(network_label)
            save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        print("Model saved to: {}".format(save_path))
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step, model_path, network_list):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        if 'localizer' in network_list:
            state['optimizer_localizer'] = self.optimizer_localizer.state_dict()
        if 'discriminator_mask' in network_list:
            state['optimizer_discriminator_mask'] = self.optimizer_discriminator_mask.state_dict()

        if 'discriminator' in network_list:
            state['optimizer_discriminator'] = self.optimizer_discriminator.state_dict()

        if 'netG' in network_list:
            state['optimizer_G'] = self.optimizer_G.state_dict()
            state['clock'] = self.netG.module.clock

        if 'generator' in network_list:
            state['optimizer_generator'] = self.optimizer_generator.state_dict()
        if 'KD_JPEG' in network_list:
            state['optimizer_KD_JPEG'] = self.optimizer_KD_JPEG.state_dict()
        if 'qf_predict' in network_list:
            state['optimizer_qf_predict'] = self.optimizer_qf_predict.state_dict()
        # for s in self.schedulers:
        #     state['schedulers'].append(s.state_dict())
        # for o in self.optimizers:
        #     state['optimizers'].append(o.state_dict())

        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(model_path , save_filename)
        print("State saved to: {}".format(save_path))
        torch.save(state, save_path)

    def resume_training(self, state_path, network_list):
        resume_state = torch.load(state_path)
        if 'clock' in resume_state and 'netG' in network_list:
            self.localizer.module.clock = resume_state['clock']
        ##  Resume the optimizers and schedulers for training
        if 'optimizer_G' in resume_state and 'netG' in network_list:
            self.optimizer_G.load_state_dict(resume_state['optimizer_G'])
        if 'optimizer_localizer' in resume_state and 'localizer' in network_list:
            self.optimizer_localizer.load_state_dict(resume_state['optimizer_localizer'])
        if 'optimizer_discriminator_mask' in resume_state and 'discriminator_mask' in network_list:
            self.optimizer_discriminator_mask.load_state_dict(resume_state['optimizer_discriminator_mask'])
        if 'optimizer_discriminator' in resume_state and 'discriminator' in network_list:
            self.optimizer_discriminator.load_state_dict(resume_state['optimizer_discriminator'])
        if 'optimizer_qf_predict' in resume_state and 'qf_predict' in network_list:
            self.optimizer_qf_predict.load_state_dict(resume_state['optimizer_qf_predict'])
        if 'optimizer_generator' in resume_state and 'generator' in network_list:
            self.optimizer_generator.load_state_dict(resume_state['optimizer_generator'])
        if 'optimizer_KD_JPEG' in resume_state and 'KD_JPEG' in network_list:
            self.optimizer_KD_JPEG.load_state_dict(resume_state['optimizer_KD_JPEG'])

        # resume_optimizers = resume_state['optimizers']
        # resume_schedulers = resume_state['schedulers']
        # assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        # assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        # for i, o in enumerate(resume_optimizers):
        #     self.optimizers[i].load_state_dict(o)
        # for i, s in enumerate(resume_schedulers):
        #     self.schedulers[i].load_state_dict(s)
