### 无参函数放在这里

print("package models.PAMI initialized")

def create_training_scripts_and_print_variables(*, opt, args, train_set=None, val_set=None):
    which_model = opt['model']

    from models.PAMI.IRNp_model import IRNpModel as M
    model = M(opt, args)

    return model, {}