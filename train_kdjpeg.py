import time

def training_script_kdjpeg(*, opt, args, rank, model, train_loader, val_loader, train_sampler):
    # total = len(train_set)
    ####################################################################################################
    # todo: TRAINING FUNCTIONALITIES
    ####################################################################################################
    current_step = 0
    if rank <= 0:
        print('Start training...')
    latest_values = None

    for epoch in range(50):
        # stateful_metrics = ['CK','RELOAD','ID','CEv_now','CEp_now','CE_now','STATE','LOCAL','lr','APEXGT','empty',
        #                     'SIMUL','RECON',
        #                     'exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
        #                     'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']
        # if rank <= 0:
        #     progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for idx, train_data in enumerate(train_loader):
            current_step += 1
            #### training
            model.feed_data(train_data)

            logs, debug_logs = model.KD_JPEG_Generator_training(current_step, latest_values)
            # if rank <= 0:
            #     progbar.add(len(model.real_H), values=logs)
