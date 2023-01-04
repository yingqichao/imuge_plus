### 无参函数放在这里

print("package models.ISP initialized")

def create_training_scripts_and_print_variables(*, opt, args, train_set=None, val_set=None):
    which_model = opt['model']

    if args.mode == 0:
        from models.ISP.Protected_Image_Generation import Protected_Image_Generation as M
    elif args.mode == 1:
        from models.ISP.Localization_Mode_One import Localization_Mode_One as M
    elif args.mode == 2:
        from models.ISP.ISP_Pipeline_Training import ISP_Pipeline_Training as M
    elif args.mode == 3:
        from models.ISP.Ablation_RGB_Protection import Ablation_RGB_Protection as M
    elif args.mode == 4:
        from models.ISP.Performance_Test import Performance_Test as M
    elif args.mode == 5:
        from models.ISP.Train_One_Single_ISP import Train_One_Single_ISP as M
    elif args.mode == 6:
        from models.ISP.Train_One_Single_Detector import Train_One_Single_Detector as M
    elif args.mode == 7:
        from models.ISP.Test import Test as M
    elif args.mode == 8:
        from models.ISP.Invert_RGB_to_RAW import Invert_RGB_to_RAW as M
    elif args.mode == 9:
        from models.ISP.CASIA_RAW_Protection import CASIA_RAW_Protection as M
    else:
        raise NotImplementedError('大神ISP的模式是不是搞错了？')
    ### original version
    # from models.ISP.Modified_invISP import Modified_invISP as M
    model = M(opt, args, train_set, val_set)

    return model, {}