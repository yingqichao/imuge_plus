import cv2
import os.path
import numpy as np

def getMaskImage(before_path, after_path, save_path):
    imgafter = cv2.imread(after_path)
    imgbefore = cv2.imread(before_path)
    imgmask = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
    height = imgmask.shape[0]
    width = imgmask.shape[1]
    debug = []
    for j in range(width):
        tmp = []
        for i in range(height):
            tmp.append(0)
        debug.append(tmp)
    for row in range(height):
        for column in range(width):
            pixel1 = imgbefore[row, column]
            pixel2 = imgafter[row, column]
            pixeldis = pixel1 - pixel2
            pixel1_avg = sum(pixel1) / 3
            pixel2_avg = sum(pixel2) / 3
            d = abs(pixel2_avg - pixel1_avg)
            debug[row][column] = d
            # distance = 0
            # for i in pixeldis:
            #     distance += pow(i, 2)
            # distance = pow(distance, 0.5)

            imgmask[row, column] = 0 if d <= 5 else 255
            # # 设置卷积核
            # kernel = np.ones((5, 5), np.uint8)
            # # 图像腐蚀
            # imgmask = cv2.erode(imgmask, kernel)
            # # 图像膨胀
            # imgmask = cv2.dilate(imgmask, kernel)
    cv2.imwrite(save_path, imgmask)

def rebuild_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print("Removed {}".format(path))
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        os.mkdir(path)

import shutil
ori_root_path = 'D://modified_immunized_image//ori_COCO_0114//'
src_root_path = 'D://modified_immunized_image//tamper//Zhang//splicing-add//'
dst_root_path = 'D://modified_immunized_image//immu_COCO_0114//'
save_root_path = 'D://temp'
rebuild_dir(save_root_path)
if not os.path.exists('D://temp//binary_masks_COCO_0114'): os.mkdir('D://temp//binary_masks_COCO_0114')
files = os.listdir(src_root_path)
print("found {} images".format(len(files)))
for name in files:
    print(name)
    ori_path = os.path.join(ori_root_path,name)
    before_path = os.path.join(src_root_path,name)
    after_path = os.path.join(dst_root_path,name)
    save_path = os.path.join(save_root_path,'masks_COCO_0114')
    if not os.path.exists(save_path): os.mkdir(save_path)
    getMaskImage(before_path, after_path, os.path.join(save_path,name))
    ####### other images ########
    save_path = os.path.join(save_root_path,'ori_COCO_0114')
    if not os.path.exists(save_path): os.mkdir(save_path)
    img = cv2.imread(ori_path)
    cv2.imwrite(os.path.join(save_path,name), img)

    save_path = os.path.join(save_root_path, 'tamper_COCO_0114')
    if not os.path.exists(save_path): os.mkdir(save_path)
    img = cv2.imread(before_path)
    cv2.imwrite(os.path.join(save_path,name), img)

    save_path = os.path.join(save_root_path ,'immu_COCO_0114')
    if not os.path.exists(save_path): os.mkdir(save_path)
    img = cv2.imread(after_path)
    cv2.imwrite(os.path.join(save_path,name), img)
