import sys
import os
import numpy as np
from common.transforms import direct_val
from common.utils import Progbar, read_annotations
import torch.backends.cudnn as cudnn
from MVSS.models.mvssnet import get_mvss
from MVSS.models.resfcn import ResFCN
import torch.utils.data
from MVSS.common.tools import inference_single
import cv2
# from apex import amp
import argparse
from noise_layers.combined import Combined
from noise_layers import *
from noise_layers.resize import Resize
from noise_layers.identity import Identity
from noise_layers.gaussian_blur import GaussianBlur
from PIL import Image
import torchvision.transforms.functional as F
from albumentations.pytorch.functional import img_to_tensor

jpeg = Combined(
            [JpegMask(80), Jpeg(80), JpegMask(90), Jpeg(90), JpegMask(70), Jpeg(70), JpegMask(60), Jpeg(60)]).cuda()
resize = Resize().cuda()
gaussian_blur = GaussianBlur().cuda()
identity = Identity().cuda()


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def getLabels(img, gt_img):
    height = img.shape[0]
    width = img.shape[1]
    #TN, TP, FN, FP
    result = [0, 0, 0 ,0]
    for row in range(height):
        for column in range(width):
            pixel = img[row, column]
            gt_pixel = gt_img[row, column]
            if pixel == gt_pixel:
                result[(pixel // 255)] += 1
            else:
                index = 2 if pixel == 0 else 3
                result[index] += 1
    return result



def getACC(TN, TP, FN, FP):
    return (TP+TN)/(TP+FP+FN+TN)
def getFPR(TN, FP):
    return FP / (FP + TN)
def getTPR(TP, FN):
    return TP/ (TP+ FN)
def getTNR(FP, TN):
    return TN/ (FP+ TN)
def getFNR(FN, TP):
    return FN / (TP + FN)
def getF1(TP, FP, FN):
    return (2*TP)/(2*TP+FP+FN)
def getBER(TN, TP, FN, FP):
    return 1/2*(getFPR(TN, FP)+FN/(FN+TP))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                # img_path = os.path.join(dirpath, fname)
                images.append((path, dirpath[len(path) + 1:], fname))
    assert images, '{:s} has no valid image file'.format(path)
    return images

def get_opt():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--model_path", type=str, help="Path to the pretrained model", default="ckpt/mvssnet_casia.pt")
    parser.add_argument("--test_file", type=str, default="pred_dir/test.txt")
    parser.add_argument("--save_dir", type=str, default="result")
    parser.add_argument("--resize", type=int, default=512)
    opt = parser.parse_args()

    return opt


def F1score(predicted_binary, gt_image, thresh=0.2):
    # gt_image = cv2.imread(src_image, 0)
    # predict_image = cv2.imread(dst_image, 0)
    # ret, gt_image = cv2.threshold(gt_image[0], int(255 * thresh), 255, cv2.THRESH_BINARY)
    # ret, predicted_binary = cv2.threshold(predict_image[0], int(255*thresh), 255, cv2.THRESH_BINARY)
    # predicted_binary = tensor_to_image(predict_image[0])
    # print(predicted_binary.shape)
    # print(gt_image.shape)
    ret, predicted_binary = cv2.threshold(predicted_binary, int(255 * thresh), 255, cv2.THRESH_BINARY)
    gt_image =gt_image[:,:,0]
    ret, gt_image = cv2.threshold(gt_image, int(255 * thresh), 255, cv2.THRESH_BINARY)



    [TN, TP, FN, FP] = getLabels(predicted_binary, gt_image)
    # print("{} {} {} {}".format(TN,TP,FN,FP))
    F1 = getF1(TP, FP, FN)
    # cv2.imwrite(save_path, predicted_binary)
    return F1, TP

def tensor_to_image(tensor):

    tensor = tensor * 255.0
    image = tensor.permute(1, 2, 0).detach().cpu().numpy()
    # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
    return np.clip(image, 0, 255).astype(np.uint8)

def image_to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(np.asarray(img)).float()
    return img_t

if __name__ == '__main__':
    opt = get_opt()
    print("in the head of inference:", opt)
    cudnn.benchmark = True

    # read test data
    test_file = opt.test_file
    dataset_name = os.path.basename(test_file).split('.')[0]
    model_type = os.path.basename(opt.model_path).split('.')[0]
    if not os.path.exists(test_file):
        print("%s not exists,quit" % test_file)
        sys.exit()
    test_data = get_paths_from_images('/home/qcying/real_world_test_images/copy-move/tamper_COCO_0114')
    ori_data = get_paths_from_images('/home/qcying/real_world_test_images/copy-move/ori_COCO_0114')
    mask_data = get_paths_from_images('/home/qcying/real_world_test_images/copy-move/binary_masks_COCO_0114')
    # test_data = read_annotations(test_file)
    new_size = opt.resize

    # load model
    model_path = opt.model_path
    if "mvssnet" in model_path:
        model = get_mvss(backbone='resnet50',
                         pretrained_base=True,
                         nclass=1,
                         sobel=True,
                         constrain=True,
                         n_input=3,
                         )
    elif "fcn" in model_path:
        model = ResFCN()
    else:
        print("model not found ", model_path)
        sys.exit()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        print("load %s finish" % (os.path.basename(model_path)))
    else:
        print("%s not exist" % model_path)
        sys.exit()
    model.cuda()
    # amp.register_float_function(torch, 'sigmoid')
    # model = amp.initialize(models=model, opt_level='O1', loss_scale='dynamic')
    model.eval()

    save_path = os.path.join(opt.save_dir, dataset_name, model_type)
    print("predicted maps will be saved in :%s" % save_path)
    os.makedirs(save_path, exist_ok=True)

    normalize = {"mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]}

    F1_sum, valid = 0, 0

    with torch.no_grad():
        # progbar = Progbar(len(test_data), stateful_metrics=['path'])
        pd_img_lab = []
        lab_all = []
        scores = []

        # for ix, (img_path, _, _) in enumerate(test_data):
        for idx in range(len(test_data)):
            p, q, r = test_data[idx]
            img_path = os.path.join(p, q, r)
            print(img_path)
            img = cv2.imread(img_path)
            ori_size = img.shape
            img = cv2.resize(img, (new_size, new_size))
            img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))

            p, q, r = ori_data[idx]
            img_path = os.path.join(p, q, r)
            ori_img = cv2.imread(img_path)
            ori_img = cv2.resize(ori_img, (new_size, new_size))
            ori_img = ori_img.reshape((-1, ori_img.shape[-3], ori_img.shape[-2], ori_img.shape[-1]))

            p, q, r = mask_data[idx]
            img_path = os.path.join(p, q, r)
            mask_img = cv2.imread(img_path)

            mask_img = cv2.resize(mask_img, (new_size, new_size))
            ret, mask_img = cv2.threshold(mask_img, int(255 * 0.5), 255, cv2.THRESH_BINARY)
            mask_img_reshape = mask_img.reshape((-1, mask_img.shape[-3], mask_img.shape[-2], mask_img.shape[-1]))
            # img = direct_val(img)
            # print(img.shape)
            # print(mask_img.shape)
            img = image_to_tensor(img[0]).unsqueeze(0).cuda()
            mask_img_reshape = image_to_tensor(mask_img_reshape[0]).unsqueeze(0).cuda()
            ori_img = image_to_tensor(ori_img[0]).unsqueeze(0).cuda()

            img = img* mask_img_reshape + ori_img*(1-mask_img_reshape)

            img_tensor_real = resize(img)
            img = tensor_to_image(img_tensor_real[0])

            img_tensor = img_to_tensor(img, normalize).unsqueeze(0)
            img_tensor = img_tensor.cuda()
            img_tensor = 0.8*img_tensor+0.2*resize(img_tensor)
            seg, _ = inference_single(img=img_tensor, model=model, th=0)
            save_seg_path = os.path.join(save_path, 'pred', 'copy-move','resize', os.path.split(img_path)[-1].split('.')[0] + '.png')
            os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
            seg = cv2.resize(seg, (new_size, new_size))
            cv2.imwrite(save_seg_path, seg.astype(np.uint8))

            save_seg_path = os.path.join(save_path, 'tamper', os.path.split(img_path)[-1].split('.')[0] + '.png')
            os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
            img = tensor_to_image(img_tensor_real[0])

            img = cv2.resize(img, (new_size, new_size))
            cv2.imwrite(save_seg_path, img.astype(np.uint8))

            F1, TP = F1score(seg, mask_img, thresh=0.5)
            if F1>0.4:
                F1_sum += F1
                valid += 1
            print("F1_sum {:3f} F1 {:3f}".format(F1_sum / (valid+1e-3), F1))
            # progbar.add(1, values=[('path', save_seg_path), ])

