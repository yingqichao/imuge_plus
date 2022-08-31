import os

import matplotlib.pyplot as plt
import gc
from noise_layers.combined import Combined
from noise_layers import *
from noise_layers.resize import Resize
from noise_layers.gaussian_blur import GaussianBlur
from MantraNet.mantranet import *

from pytorch_lightning import Trainer

device='cuda' #to change if you have a GPU with at least 12Go RAM (it will save you a lot of time !)
model=pre_trained_model(weight_path='./MantraNetv4.pt',device=device)

model.eval()
dist_folder = '/home/qcying/real_world_test_images/inpainting/tamper_COCO_0114/'
# for image in os.listdir(dist_folder):
#     print(f'{dist_folder}/{image}')
#     # plt.figure(figsize=(20,20))
check_forgery(model,fold_path=dist_folder, img_path=None,device=device)