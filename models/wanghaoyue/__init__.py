### 无参函数放在这里

print("package models.IFA initialized")

def create_training_scripts_and_print_variables(*, opt, args, train_set=None, val_set=None):
    which_model = opt['model']
    if args.mode in [0,4]:
        from models.wanghaoyue.TrainMain import TrainMain as M

        raise NotImplementedError(f'大神ISP的模式是不是搞错了？{args.mode}')
    model = M(opt, args, train_set, val_set)

    return model, {}