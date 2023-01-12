import time

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


    for epoch in range(100):


        ### todo: begin training
        current_step = 0

        running_list = {}  # [0.0]*len(variables_list)
        valid_idx = 0

        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for idx, train_data in enumerate(train_loader):

            #### training
            model.feed_data_router(batch=train_data, mode=args.mode)

            logs, debug_logs, did_val = model.optimize_parameters_router(mode=args.mode, step=current_step, epoch=epoch)


            for key in logs:
                if key not in running_list:
                    running_list[key] = 0
                running_list[key] += logs[key]
            valid_idx += 1


            if valid_idx > 0 and (
                    idx < 10 or valid_idx % print_step == print_step - 1):  # print every 2000 mini-batches

                end = time.time()
                lr = logs['lr']
                info_str = f'[{epoch}, {valid_idx-1} {idx * model.real_H.shape[0]} {rank} {lr}] '
                for key in running_list:
                    info_str += f'{key}: {running_list[key] / valid_idx:.4f} '
                info_str += f'time per sample {(end - start) / print_step / model.real_H.shape[0]:.4f} s'
                print(info_str)
                start = time.time()
                ## refresh the counter to see if the model behaves abnormaly.
                if valid_idx >= restart_step:
                    running_list = {}  # [0.0] * len(variables_list)
                    valid_idx = 0

            current_step += 1

        ## todo: inference 100 images
        print(f"do_evaluate: {opt['do_evaluate']}")
        if opt['do_evaluate'] > 0:
            print(f"Start evaluating for epoch {epoch}")
            # from inference_RR_IFA import inference_script_RR_IFA
            # inference_script_RR_IFA(opt=opt, args=args, rank=rank, model=model,
            #                         train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler,
            #                         num_images=100)
            current_step = 0

            running_list = {}  # [0.0]*len(variables_list)
            valid_idx = 0

            for idx, test_data in enumerate(val_loader):

                #### training
                model.feed_data_val_router(batch=test_data, mode=args.mode)

                logs, debug_logs, did_val = model.validate_router(mode=args.mode, step=current_step,
                                                                  epoch=epoch)

                for key in logs:
                    if key not in running_list:
                        running_list[key] = 0
                    running_list[key] += logs[key]
                valid_idx += 1

                if valid_idx > 0 and (
                        idx < 10 or valid_idx % print_step_val == print_step_val - 1):  # print every 2000 mini-batches

                    end = time.time()
                    lr = logs['lr']
                    info_str = f'[Val {epoch}, {valid_idx - 1} {idx * model.real_H_val.shape[0]} {rank} {lr}] '
                    for key in running_list:
                        info_str += f'{key}: {running_list[key] / valid_idx:.4f} '
                    info_str += f'time per sample {(end - start) / print_step_val / model.real_H_val.shape[0]:.4f} s'
                    print(info_str)
                    start = time.time()
                    ## refresh the counter to see if the model behaves abnormaly.
                    if valid_idx >= restart_val_step:
                        running_list = {}  # [0.0] * len(variables_list)
                        valid_idx = 0

                current_step += 1
                if idx * model.real_H_val.shape[0] >= opt['do_evaluate']:
                    ### having tested enough images, quit

                    break

        print(f"End evaluating for epoch {epoch}")

