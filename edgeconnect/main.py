import os
import cv2
import random

import imageio
import numpy as np
import torch
import argparse
from shutil import copyfile
from edgeconnect.src.config import Config
from edgeconnect.src.edge_connect import EdgeConnect, EdgeConnectTest
from skimage.feature import canny
from PIL import Image
import torchvision.transforms.functional as F


def get_model():
    config = load_config(mode=2)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, '
    model = EdgeConnectTest(config)
    model.load()
    return model


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = EdgeConnect(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints/places2', help='model checkpoints path (default: ./checkpoints)')
    # parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], default=3, help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')
    #
    # # test mode
    # if mode == 2:
    #     parser.add_argument('--input', type=str, default='', help='path to the input images directory or an input image')
    #     parser.add_argument('--mask', type=str, default='', help='path to the masks directory or a mask file')
    #     parser.add_argument('--edge', type=str, default='', help='path to the edges directory or an edge file')
    #     parser.add_argument('--output', type=str, default='', help='path to the output directory')
    #
    # args = parser.parse_args()
    args = {
        'path': '/groupshare/checkpoints/places2',
        'model': 3,
    }
    config_path = os.path.join(args['path'], 'config.yml')

    # # create checkpoints path if does't exist
    # if not os.path.exists(weights_path):
    #     os.makedirs(weights_path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./edgeconnect/config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1
        if args['model']:
            config.MODEL = args['model']

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args['model'] if args['model'] is not None else 3
        config.INPUT_SIZE = 512
        #
        # if args.input is not None:
        #     config.TEST_FLIST = args.input
        #
        # if args.mask is not None:
        #     config.TEST_MASK_FLIST = args.mask
        #
        # if args.edge is not None:
        #     config.TEST_EDGE_FLIST = args.edge
        #
        # if args.output is not None:
        #     config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args['model'] if args['model'] is not None else 3

    return config

def to_tensor(img):
    img = Image.fromarray(img)
    img_tensor = F.to_tensor(img).unsqueeze(0).float()
    return img_tensor.cuda()


def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

if __name__ == "__main__":
    # main()
    # 测试方式示例
    model = get_model()
    src_image = './examples/places2/images/00020_0_0.png'
    mask_image = './examples/places2/masks/00020_0_0.png'
    img = imageio.imread(src_image)
    from skimage.color import rgb2gray
    image_gray = rgb2gray(img)
    """
    rgb = _prepare_colorarray(rgb)
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=rgb.dtype)
    return rgb @ coeffs
    以上是原本实现过程 可以通过1*1卷积实现
    """
    mask = imageio.imread(mask_image)
    # 在计算canny结果的时候，需要把篡改区域罩住，防止获取这部分的边缘信息
    canny_used_mask = (1 - mask / 255).astype(np.bool)
    edge = canny(image_gray, sigma=2, mask=canny_used_mask)
    items = (to_tensor(img), to_tensor(image_gray), to_tensor(edge), to_tensor(mask))
    result = model.test(items)
    result = postprocess(result).cpu().numpy()
    result = result[0].astype(np.uint8)
    Image.fromarray(result).save('./test.png')

