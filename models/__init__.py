# import logging
# logger = logging.getLogger('base')

print("package data initialized")

def create_models(*, opt, args, train_set=None, val_set=None):
    which_model = opt['model']

    # if which_model == 'CVPR':
    #     from models.ISP.Modified_invISP import IRNModel as M
    #     model = M(opt, args)
    # elif which_model == 'ICASSP_NOWAY':
    #     from models.CLRNet.IRNcrop_model import IRNcropModel as M
    #     model = M(opt, args)
    # elif which_model == 'CLRNet':
    #     from models.CLRNet.IRNclrNew_model import IRNclrModel as M
    #     model = M(opt, args)
    # else:
    #     raise NotImplementedError('大神是不是搞错了？')

    if 'ISP' in which_model:
        from models.ISP import create_training_scripts_and_print_variables
    elif 'IFA' in which_model:
        from models.IFA import create_training_scripts_and_print_variables
    elif 'PAMI' in which_model:
        from models.PAMI import create_training_scripts_and_print_variables
    elif 'tianchi' in which_model:
        from models.tianchi import create_training_scripts_and_print_variables
    elif 'SAM_forgery_detection' in which_model:
        from models.SAM_forgery_detection import create_training_scripts_and_print_variables
    elif 'detection_large_model' in which_model:
        from models.detection_large_model import create_training_scripts_and_print_variables
    else:
        raise NotImplementedError("大神，模式是不是搞错了？现在只支持ISP IFA PAMI")

    model, variables_list = create_training_scripts_and_print_variables(opt=opt, args=args, train_set=train_set,
                                                                        val_set=val_set)


    print('Model [{:s}] is created.'.format(model.__class__.__name__))

    ######### get variable lists ##############
    # variables_list = {} #[]

    # ############  PAMI imuge ###############
    # if 'PAMI' in which_model:
    #     if args.mode in [0,1,2]:
    #         variables_list = ['local','loss','null','lF','lB','canny','mask_rate','CE', 'CE_ema', 'ERROR', 'SIMU', 'PF', 'PB', 'SF', 'SB',
    #                           'DIS', 'DIS_A']
    #     elif args.mode in [3]:
    #         variables_list = ['file_generated']
    #
    #
    # ############  ISP fnctionalities ###############
    # elif 'ISP' in which_model:
    #     if 'ISP' in which_model and args.mode == 0:
    #         variables_list = ['RAW_L1', 'RAW_PSNR', 'loss', 'ERROR', 'CE', 'CEL1', 'F1', 'F1_1', 'RECALL', 'RECALL_1',
    #                           'RGB_PSNR_0', 'RGB_PSNR_1', 'RGB_PSNR_2']
    #     elif 'ISP' in which_model and args.mode == 4:
    #         variables_list = ['ERROR', 'CE', 'F1', 'RECALL', 'RAW_PSNR', 'RGB_PSNR', 'AUC', 'IoU']
    #     elif 'ISP' in which_model and args.mode in [2, 3, 4, 7, 8]:
    #         variables_list = ['ISP_PSNR', 'ISP_L1', 'CE', 'CE_ema', 'CEL1', 'l1_ema', 'Mean', 'Std', 'CYCLE_PSNR',
    #                           'CYCLE_L1', 'PIPE_PSNR', 'PIPE_L1', 'loss',
    #                           'RAW_L1', 'RAW_PSNR', 'PSNR_DIFF', 'ISP_PSNR_NOW', 'ISP_SSIM_NOW', 'Percept', 'Gray', 'Style',
    #                           'ERROR', 'inpaint', 'inpaintPSNR'
    #                           ]
    #     elif 'ISP' in which_model and args.mode == 5:
    #         variables_list = ['CYCLE_PSNR', 'CYCLE_L1', 'loss']
    #     elif 'ISP' in which_model and args.mode == 6:
    #         variables_list = ['CE', 'CE_MVSS', 'CE_Mantra']
    #
    # ############  crop recovery ###############
    # elif 'CLRNet' in which_model:
    #     variables_list = ['loss', 'PF', 'PB', 'CE', 'SSFW', 'SSBK', 'lF', 'local']
    # else:
    #     raise NotImplementedError("需要给你的任务加上要打印得的变量！在models/__init__.py里面")
    #
    # print(f"variables_list: {variables_list}")




    return model
