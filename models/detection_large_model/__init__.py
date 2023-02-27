### 无参函数放在这里

print("package models.detection_large_model initialized")

def create_training_scripts_and_print_variables(*, opt, args, train_set=None, val_set=None):
    which_model = opt['model']
    from models.detection_large_model.baseline_LLD import baseline_LLD as M

    model = M(opt, args, train_set, val_set)

    return model, {}