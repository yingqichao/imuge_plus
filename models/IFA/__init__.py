### 无参函数放在这里

print("package models.ISP initialized")

def create_training_scripts_and_print_variables(*, opt, args, train_set=None, val_set=None):
    which_model = opt['model']
    if args.mode == 0:
        from models.IFA.RR_IFA import RR_IFA as M
    elif args.mode == 1:
        from models.IFA.IFA_loss import IFA_loss as M
    else:
        raise NotImplementedError('大神ISP的模式是不是搞错了？')
    model = M(opt, args, train_set, val_set)

    return model, {}