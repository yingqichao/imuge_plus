import time

def training_script(*, opt, args, rank, model, train_loader, val_loader, train_sampler):
    # total = len(train_set)
    ####################################################################################################
    # todo: TRAINING FUNCTIONALITIES
    ####################################################################################################

    if rank <= 0:
        print('Start training ...')
    # latest_values = None

    current_step, valid_idx = 0, 0
    running_list, print_step, restart_step = {}, 100, opt['restart_step']
    start = time.time()

    epochs = 200 if not opt['only_test'] else 1
    for epoch in range(epochs):
        current_step = 0
        if not opt['only_test']:
            print(f"Staring epoch {epoch}...")
            for idx, batch in enumerate(train_loader):
                model.feed_data_router(batch=batch, mode=args.mode)

                logs = model.train_tianchi(epoch=epoch, step=current_step)

                #### 对变量求平均
                for key in logs:
                    if key not in running_list:
                        running_list[key] = 0
                    running_list[key] += logs[key]
                valid_idx += 1

                if valid_idx > 0 and (idx < 10 or valid_idx % print_step == print_step - 1):
                    end = time.time()
                    lr = logs['lr']
                    info_str = f'[{epoch + 1}, {valid_idx + 1} {idx * model.img.shape[0]} {rank} {lr}] '
                    for key in running_list:
                        info_str += f'{key}: {running_list[key] / valid_idx:.4f} '
                    info_str += f'time per sample {(end - start) / print_step / model.img.shape[0]:.4f} s'
                    print(info_str)
                    start = time.time()
                    ## refresh the counter to see if the models behaves abnormaly.
                    if valid_idx >= restart_step:
                        running_list = {}  # [0.0] * len(variables_list)
                        valid_idx = 0

                current_step += 1

        # todo: testing tianchi
        if opt['conduct_test']:
            print(f"Staring testing epoch {epoch}...")
            for idx, batch in enumerate(val_loader):
                model.feed_data_val_router(batch=batch, mode=args.mode)
                model.test_tianchi(epoch=epoch, step=idx)