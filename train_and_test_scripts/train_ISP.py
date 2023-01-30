import time

def training_script_ISP(*, opt, args, rank, model, train_loader, val_loader, train_sampler):
    # total = len(train_set)
    ####################################################################################################
    # todo: TRAINING FUNCTIONALITIES
    ####################################################################################################

    if rank <= 0:
        print('Start training ...')
    # latest_values = None

    print_step, restart_step = 40, opt['restart_step']
    start = time.time()

    # train_generator_1 = iter(train_loader_1)
    val_generator = iter(val_loader)
    val_item = next(val_generator)
    model.feed_data_val_router(batch=val_item, mode=args.mode)
    for epoch in range(50):
        current_step = 0

        # stateful_metrics = ['CK','RELOAD','ID','CEv_now','CEp_now','CE_now','STATE','lr','APEXGT','empty',
        #                     'SIMUL','RECON','RealTime'
        #                     'exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
        #                     'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']
        # if rank <= 0:
        #     progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
        running_list = {}  # [0.0]*len(variables_list)
        valid_idx = 0
        # running_CE_MVSS, running_CE_mantra, running_CE_resfcn, valid_idx = 0.0, 0.0, 0.0, 0.0
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        idx = 0
        # for idx, train_data in enumerate(train_loader):
        while True:

            #### get item from another dataset
            # try:
            #     train_item_1 = next(train_generator_1)
            # except StopIteration as e:
            #     print("The end of val set is reached. Refreshing...")
            #     train_generator_1 = iter(train_loader_1)
            #     train_item_1 = next(train_generator_1)
            # #### training
            # model.feed_data_router(batch=train_data, mode=args.mode)

            # val
            model.feed_data_router(batch=val_item, mode=args.mode)

            logs, debug_logs, did_val = model.optimize_parameters_router(mode=args.mode, step=current_step)
            if did_val:
                try:
                    val_item = next(val_generator)
                except StopIteration as e:
                    print("The end of val set is reached. Refreshing...")

                    info_str = f'valid_idx:{valid_idx} '
                    for key in running_list:
                        info_str += f'{key}: {running_list[key] / valid_idx:.4f} '
                    with open('./test_result_2.txt', 'a') as f:
                        f.write(info_str + '\n')
                    f.close()
                    # raise StopIteration()
                    opt['inference_benign_attack_begin_idx'] = opt['inference_benign_attack_begin_idx'] + 1
                    if opt['inference_benign_attack_begin_idx'] >= 24:
                        raise StopIteration()
                    current_step = 0
                    idx = 0
                    running_list = {} #[0.0] * len(variables_list)
                    valid_idx = 0
                    val_generator = iter(val_loader)
                    val_item = next(val_generator)
                model.feed_data_val_router(batch=val_item, mode=args.mode)

            # if variables_list[0] in logs or variables_list[1] in logs or variables_list[2] in logs:
            for key in logs:
                if key not in running_list:
                    running_list[key] = 0
                running_list[key] += logs[key]
            valid_idx += 1
            # else:
            #     ## which is kind of abnormal, print
            #     print(variables_list)

            if valid_idx > 0 and (
                    idx < 10 or valid_idx % print_step == print_step - 1):  # print every 2000 mini-batches
                # print(f'[{epoch + 1}, {valid_idx + 1} {rank}] '
                #       f'running_CE_MVSS: {running_CE_MVSS / print_step:.2f} '
                #       f'running_CE_mantra: {running_CE_mantra / print_step:.2f} '
                #       f'running_CE_resfcn: {running_CE_resfcn / print_step:.2f} '
                #       )
                end = time.time()
                lr = logs['lr']
                info_str = f'[{epoch + 1}, {valid_idx + 1} {idx * model.real_H.shape[0]} {rank} {lr}] '
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
            idx += 1

        # ## todo: inference 100 images
        # from inference_RR_IFA import inference_script_RR_IFA
        # inference_script_RR_IFA(opt=opt, args=args, rank=rank, model=model,
        #                         train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler,
        #                         num_images=100)