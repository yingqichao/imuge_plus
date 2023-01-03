from detection_methods.MantraNet.mantranet import *

device='cuda' #to change if you have a GPU with at least 12Go RAM (it will save you a lot of time !)
model=pre_trained_model(weight_path='./MantraNetv4.pt',device=device)

model.eval()
dist_folder = '/home/qcying/real_world_test_images/inpainting/tamper_COCO_0114/'
# for image in os.listdir(dist_folder):
#     print(f'{dist_folder}/{image}')
#     # plt.figure(figsize=(20,20))
check_forgery(model,fold_path=dist_folder, img_path=None,device=device)