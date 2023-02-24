

def scripts_router(*, which_model, opt, args, rank, model,
                              train_loader, val_loader, train_sampler):
    ####################################################################################################
    # todo: TRAINING FUNCTIONALITIES
    ####################################################################################################
    start_epoch, current_step = 0, 0
    if 'ISP' in which_model or \
            ('PAMI' in which_model and args.mode in [0.0, 3.0]):
        ## todo: general training with a validation iterator for PAMI/DRAW
        from train_and_test_scripts.train_ISP import training_script_ISP
        training_script_ISP(opt=opt, args=args, rank=rank, model=model,
                            train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler)

    elif 'IFA' in which_model:
        ## todo: training of RR-IFA
        from train_and_test_scripts.train_IFA import training_script_IFA
        training_script_IFA(opt=opt, args=args, rank=rank, model=model,
                            train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler)

    elif which_model == 'PAMI' and args.mode == 2.0:
        ## todo: kd-jpeg training
        from train_and_test_scripts.train_kdjpeg import training_script_kdjpeg
        training_script_kdjpeg(opt=opt, args=args, rank=rank, model=model,
                               train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler)
    elif which_model == 'tianchi':
        ## todo: tianchi training
        from train_and_test_scripts.train_tianchi import training_script
        training_script(opt=opt, args=args, rank=rank, model=model,
                               train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler)


    ####################################################################################################
    # todo: TESTING FUNCTIONALITIES
    ####################################################################################################
    elif which_model == 'PAMI' and args.mode == 1.0:
        ## todo: PAMI inference
        from train_and_test_scripts.inference_PAMI import inference_script_PAMI
        inference_script_PAMI(opt=opt, args=args, rank=rank, model=model,
                              train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler)


    elif 'IFA' in which_model and args.mode in [2.0]:
        from train_and_test_scripts.inference_RR_IFA import inference_script_RR_IFA
        inference_script_RR_IFA(opt=opt, args=args, rank=rank, model=model,
                                train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler)

    else:
        raise NotImplementedError('大神是不是搞错了？')