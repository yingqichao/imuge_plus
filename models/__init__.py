import logging
logger = logging.getLogger('base')


def create_training_scripts_and_print_variables(*, opt, args, train_set=None, val_set=None):
    which_model = opt['model']

    if which_model == 'CVPR':
        from models.ISP.Modified_invISP import IRNModel as M
        model = M(opt, args)
    elif which_model == 'PAMI':
        from models.PAMI.IRNp_model import IRNpModel as M
        model = M(opt, args)
    elif which_model == 'ICASSP_NOWAY':
        from models.CLRNet.IRNcrop_model import IRNcropModel as M
    elif which_model == 'CLRNet':
        from models.CLRNet.IRNclrNew_model import IRNclrModel as M
        model = M(opt, args)
    elif which_model == 'ISP':
        if args.mode==2:
            from models.ISP.ISP_Pipeline_Training import ISP_Pipeline_Training as M
        elif args.mode==3:
            from models.ISP.Ablation_RGB_Protection import Ablation_RGB_Protection as M
        elif args.mode==4:
            from models.ISP.Performance_Test import Performance_Test as M
        elif args.mode==5:
            from models.ISP.Train_One_Single_ISP import Train_One_Single_ISP as M
        elif args.mode==6:
            from models.ISP.Train_One_Single_Detector import Train_One_Single_Detector as M
        ### original version
        # from models.ISP.Modified_invISP import Modified_invISP as M
        model = M(opt, args, train_set)
    else:
        raise NotImplementedError('大神是不是搞错了？')

    print('Model [{:s}] is created.'.format(model.__class__.__name__))

    ######### get variable lists ##############
    variables_list = []

    ############  PAMI imuge ###############
    if 'PAMI' in which_model:
        if args.mode in [0,1,2]:
            variables_list = ['local','loss','null','lF','lB','canny','mask_rate','CE', 'CE_ema', 'ERROR', 'SIMU', 'PF', 'PB', 'SF', 'SB',
                              'DIS', 'DIS_A']
        elif args.mode in [3]:
            variables_list = ['file_generated']


    ############  ISP fnctionalities ###############
    elif 'ISP' in which_model:
        if 'ISP' in which_model and args.mode == 0:
            variables_list = ['RAW_L1', 'RAW_PSNR', 'loss', 'ERROR', 'CE', 'CEL1', 'F1', 'F1_1', 'RECALL', 'RECALL_1',
                              'RGB_PSNR_0', 'RGB_PSNR_1', 'RGB_PSNR_2']
        elif 'ISP' in which_model and args.mode == 4:
            variables_list = ['ERROR', 'CE', 'F1', 'RECALL', 'RAW_PSNR', 'RGB_PSNR', 'AUC', 'IoU']
        elif 'ISP' in which_model and args.mode == 0:
            variables_list = ['RAW_L1', 'RAW_PSNR', 'loss', 'ERROR', 'CE', 'CEL1', 'F1', 'F1_1', 'RECALL', 'RECALL_1',
                              'RGB_PSNR_0', 'RGB_PSNR_1', 'RGB_PSNR_2']
        elif 'ISP' in which_model and args.mode in [2, 3, 4]:
            variables_list = ['ISP_PSNR', 'ISP_L1', 'CE', 'CE_ema', 'CEL1', 'l1_ema', 'Mean', 'Std', 'CYCLE_PSNR',
                              'CYCLE_L1', 'PIPE_PSNR', 'PIPE_L1', 'loss',
                              'RAW_L1', 'RAW_PSNR', 'PSNR_DIFF', 'ISP_PSNR_NOW', 'ISP_SSIM_NOW', 'Percept', 'Gray', 'Style',
                              'ERROR', 'inpaint', 'inpaintPSNR'
                              ]
        elif 'ISP' in which_model and args.mode == 5:
            variables_list = ['CYCLE_PSNR', 'CYCLE_L1', 'loss']
        elif 'ISP' in which_model and args.mode == 6:
            variables_list = ['CE', 'CE_MVSS', 'CE_Mantra']

    ############  crop recovery ###############
    elif 'CLRNet' in which_model:
        variables_list = ['loss', 'PF', 'PB', 'CE', 'SSFW', 'SSBK', 'lF', 'local']
    else:
        raise NotImplementedError("需要给你的任务加上要打印得的变量！在models/__init__.py里面")

    print(f"variables_list: {variables_list}")




    return model, variables_list
