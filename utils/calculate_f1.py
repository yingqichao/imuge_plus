import cv2
import os.path


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



def F1score(src_image, dst_image, save_path, thresh=0.2):
    gt_image = cv2.imread(src_image, 0)
    predict_image = cv2.imread(dst_image, 0)
    ret, gt_image = cv2.threshold(gt_image, int(255 * thresh), 255, cv2.THRESH_BINARY)
    ret, predicted_binary = cv2.threshold(predict_image, int(255*thresh), 255, cv2.THRESH_BINARY)

    [TN, TP, FN, FP] = getLabels(predicted_binary, gt_image)
    F1 = getF1(TP, FP, FN)
    cv2.imwrite(save_path, predicted_binary)
    return F1, TP

src_root_path = 'D://modified_immunized_image//results//jpeg//gt_masks//'
dst_root_path = 'D://modified_immunized_image//results//jpeg//predicted_masks//'
save_root_path = 'D://modified_immunized_image//results//jpeg//predicted_binary//'
files = os.listdir(src_root_path)
print("found {} images".format(len(files)))
thresh=0.1
while thresh<0.95:
    save_thresh_path = save_root_path + "{}//".format(int(thresh*10))
    if not os.path.exists(save_thresh_path): os.mkdir(save_thresh_path)
    sum_F1 = 0
    for i in range(len(files)-1):
        name = files[i]
        before_path = src_root_path + name
        after_path = dst_root_path + name
        save_path = save_thresh_path + name
        F1, TP = F1score(before_path, after_path, save_path, thresh=thresh)
        print("{} : {}".format(name,F1))
        sum_F1 += F1
    sum_F1 /= (len(files)-1)
    print("thresh {} : average F1 {}".format(thresh,sum_F1))
    thresh += 0.1