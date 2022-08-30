import os

import cv2
import torchvision

from noise_layers import *


def load_image(path, grayscale):
    image_c = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1] if not grayscale else cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_c = cv2.resize(image_c, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
    img = image_c.copy().astype(np.float32)
    img /= 255.0
    if not grayscale:
        img = img.transpose(2, 0, 1)
    tensor_c = torch.from_numpy(img).unsqueeze(0).cuda()
    if grayscale:
        tensor_c = tensor_c.unsqueeze(0)

    return tensor_c



tensor_c = load_image(path='/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/target3.png',grayscale=False)
# watermark_c = load_image(path='/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/images/images/1.png',grayscale=True)
source_tamper = load_image(path='/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/images/images/14.png',grayscale=False)
mask_tamper = load_image(path='/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real/14_GT.png',grayscale=True)
save_path = '/home/qichaoying/Downloads/Invertible-Image-Rescaling-master/icassp_real'


cropped = (1-mask_tamper)*tensor_c+mask_tamper*source_tamper


name = os.path.join(
    save_path,
    "for_experiment.png")
for image_no in range(cropped.shape[0]):
    camera_ready = cropped[image_no].unsqueeze(0)
    print("Save to: {}".format(name))
    torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                 name,
                                 nrow=1, padding=0, normalize=False)