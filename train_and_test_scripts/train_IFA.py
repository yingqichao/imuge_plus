import time
import torch

def training_script_IFA(*, opt, args, rank, model, train_loader, val_loader, train_sampler):
    # total = len(train_set)
    ####################################################################################################
    # todo: TRAINING FUNCTIONALITIES
    ####################################################################################################

    if rank <= 0:
        print('Start training ...')

    print_step, restart_step = 40, opt['restart_step']
    print_step_val, restart_val_step = 10, 1000000
    start = time.time()

    if args.mode in [4]:
        from data.data_sampler import DistIterSampler
        dataset_opt, world_size= opt['datasets']['train'], torch.distributed.get_world_size()
        from data.CASIA_dataset import CASIA_dataset as D
        detection_set = D(opt, dataset_opt, split=False, dataset=[opt["detection_dataset"]], attack_list=None, with_mask=True, with_au=False)
        dataset_ratio = 1  # world_size  # enlarge the size of each epoch
        if opt['dist']:
            detection_sampler = DistIterSampler(detection_set, world_size, rank, dataset_ratio, seed=int(time.time())%1000)
        else:
            detection_sampler = None
        batch_size, num_workers = dataset_opt['batch_size'], dataset_opt['n_workers']
        detection_loader = torch.utils.data.DataLoader(detection_set, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers, sampler=detection_sampler, drop_last=True,
                                                   pin_memory=True)
        detection_generator = iter(detection_loader)
        detection_item = next(detection_generator)


    for epoch in range(100):


        ### todo: begin training
        current_step = 0

        running_list = {}  # [0.0]*len(variables_list)
        valid_idx_list = {}
        # valid_idx = 0

        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for idx, train_data in enumerate(train_loader):

            #### training
            model.feed_data_router(batch=train_data, mode=args.mode)
            if args.mode in [4]:
                model.feed_aux_data(detection_item)
                try:
                    detection_item = next(detection_generator)
                except StopIteration as e:
                    print("The end of val set is reached. Refreshing...")
                    detection_generator = iter(detection_loader)
                    detection_item = next(detection_generator)

            logs, debug_logs, did_val = model.optimize_parameters_router(mode=args.mode, step=current_step, epoch=epoch)


            for key in logs:
                if key!='status':
                    if key not in running_list:
                        running_list[key] = 0
                        valid_idx_list[key] = 0
                    running_list[key] += logs[key]
                    valid_idx_list[key] += 1


            if idx > 0 and (
                    idx < 10 or idx % print_step == print_step - 1):  # print every 2000 mini-batches

                end = time.time()
                lr = logs['lr']
                status = '' if 'status' not in logs else logs['status']
                info_str = f'[R{rank}, E{epoch}, B{idx} N{idx * model.real_H.shape[0]} LR{lr} {status}] '
                for key in running_list:
                    if key!='status':
                        info_str += f'{key}: {running_list[key] / valid_idx_list[key]:.4f} '
                info_str += f'time per sample {(end - start) / print_step / model.real_H.shape[0]:.4f} s'
                print(info_str)
                start = time.time()
                ## refresh the counter to see if the model behaves abnormaly.
                if idx >= restart_step:
                    running_list = {}  # [0.0] * len(variables_list)
                    for key in valid_idx_list:
                        valid_idx_list[key] = 0

            current_step += 1

        # ## todo: inference 100 images
        # ## todo: has bug, omit
        # if args.mode in [0]:
        #     print(f"do_evaluate: {opt['do_evaluate']}")
        #     if opt['do_evaluate'] > 0:
        #         print(f"Start evaluating for epoch {epoch}")
        #         # from inference_RR_IFA import inference_script_RR_IFA
        #         # inference_script_RR_IFA(opt=opt, args=args, rank=rank, model=model,
        #         #                         train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler,
        #         #                         num_images=100)
        #         current_step = 0
        #
        #         running_list = {}  # [0.0]*len(variables_list)
        #         valid_idx = 0
        #
        #         for idx, test_data in enumerate(val_loader):
        #
        #             #### training
        #             model.feed_data_val_router(batch=test_data, mode=args.mode)
        #
        #             logs, debug_logs, did_val = model.validate_router(mode=args.mode, step=current_step,
        #                                                               epoch=epoch)
        #
        #             for key in logs:
        #                 if key not in running_list:
        #                     running_list[key] = 0
        #                 running_list[key] += logs[key]
        #             valid_idx += 1
        #
        #             if valid_idx > 0 and (
        #                     idx < 10 or valid_idx % print_step_val == print_step_val - 1):  # print every 2000 mini-batches
        #
        #                 end = time.time()
        #                 lr = logs['lr']
        #                 info_str = f'[Val {epoch}, {valid_idx - 1} {idx * model.real_H_val.shape[0]} {rank} {lr}] '
        #                 for key in running_list:
        #                     info_str += f'{key}: {running_list[key] / valid_idx:.4f} '
        #                 info_str += f'time per sample {(end - start) / print_step_val / model.real_H_val.shape[0]:.4f} s'
        #                 print(info_str)
        #                 start = time.time()
        #                 ## refresh the counter to see if the model behaves abnormaly.
        #                 if valid_idx >= restart_val_step:
        #                     running_list = {}  # [0.0] * len(variables_list)
        #                     valid_idx = 0
        #
        #             current_step += 1
        #             if idx * model.real_H_val.shape[0] >= opt['do_evaluate']:
        #                 ### having tested enough images, quit
        #
        #                 break
        #
        #     print(f"End evaluating for epoch {epoch}")

