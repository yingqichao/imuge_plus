import torch
# path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
# if path not in sys.path:
#     sys.path.insert(0, path)
# sys.path.append('../..')
# import CATNet
# print(os.getcwd())
from detection_methods.CATNet.lib.config import update_config
from detection_methods.CATNet.lib.config import config
import argparse
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def get_model(NUM_CLASSES=1,use_SRM=0,load_pretrained_cat_model=True):
    default_config = './detection_methods/CATNet/experiments/CAT_full.yaml'
    opts = ['TEST.MODEL_FILE', '/groupshare/CATNet/RGB_only_v2.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0']
    args = argparse.Namespace(cfg=default_config, opts=opts)
    update_config(config, args)

    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.deterministic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED

    from detection_methods.CATNet.lib.models.network_RGB import get_seg_model
    model = get_seg_model(config, NUM_CLASSES, use_SRM, load_pretrained_cat_model)
    if load_pretrained_cat_model:
        model_state_file = config.TEST.MODEL_FILE
        checkpoint = torch.load(model_state_file, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    # model = nn.DataParallel(model).cuda()
    return model


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    x_rand = torch.randn((3, 3, 512, 512)).cuda()
    x_height, x_width = x_rand.shape[-2:]
    model = get_model()
    pred = model(x_rand, torch.zeros(1).cuda())
    pred = F.interpolate(pred, size=(x_height, x_width), mode='bilinear')
    print(pred.shape)
    pred = F.softmax(pred, dim=1)
    _, pred = torch.split(pred, 1, dim=1)
    # print(pred)
    print(pred.shape)