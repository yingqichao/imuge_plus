import time

def inference_script_RR_IFA(*, opt, args, rank, model, train_loader, val_loader, train_sampler):
    # total = len(train_set)
    ####################################################################################################
    # todo: Evaluating RR-IFA
    ####################################################################################################

    if rank <= 0:
        print('Start evaluating ...')
    # latest_values = None

    print_step, restart_step = 40, 10000000
    start = time.time()

    current_step = 0

    running_list = {}  # [0.0]*len(variables_list)
    valid_idx = 0
    # running_CE_MVSS, running_CE_mantra, running_CE_resfcn, valid_idx = 0.0, 0.0, 0.0, 0.0
    if opt['dist']:
        train_sampler.set_epoch(0)

    model.inference_RR_IFA(val_loader=val_loader)

