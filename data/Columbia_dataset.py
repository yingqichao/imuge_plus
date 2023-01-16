import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import os
import albumentations as A
import copy

class Columbia_dataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, dataset_opt, is_train=True, with_mask=True, splic_only=False):
        super(Columbia_dataset, self).__init__()
        self.is_train = is_train
        self.dataset_opt = dataset_opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.GT_size = self.dataset_opt['GT_size']
        self.with_mask = with_mask
        self.splic_only = splic_only

        print(f"Using Columbia_dataset with_au False(not supported) with_mask {with_mask} is_train {is_train}")

        self.transform_just_resize = A.Compose(
            [
                A.Resize(always_apply=True, height=self.GT_size, width=self.GT_size)
            ]
        )

        self.path_root = '/groupshare/Columbia_Uncompressed_Image_Splicing_Detection/'
        self.au_folder = '4cam_auth'
        self.sp_folder = '4cam_splc'
        GT_path = []

        fake_path = self._list_cur_images(os.path.join(self.path_root, self.sp_folder))
        GT_path += fake_path
        if not self.splic_only:
            real_path = self._list_cur_images(os.path.join(self.path_root, self.au_folder))
            GT_path += real_path
        random.shuffle(GT_path)
        dataset_len = len(GT_path)
        print(f"len image {dataset_len}")
        num_train_val_split = int(dataset_len * 0.85)
        self.paths_GT = (GT_path[:num_train_val_split] if self.is_train else GT_path[num_train_val_split:])

        assert self.paths_GT, 'Error: GT path is empty.'


    def __getitem__(self, index):
        return_list = []
        # scale = self.dataset_opt['scale']

        # get GT image
        GT_path = self.paths_GT[index]
        print(GT_path)

        # img_GT = util.read_img(GT_path)
        img_GT = cv2.imread(GT_path, cv2.IMREAD_COLOR)
        img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]
        img_GT = self.transform_just_resize(image=copy.deepcopy(img_GT))["image"]
        img_GT = img_GT.astype(np.float32) / 255.
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        # some images have 4 channels
        if img_GT.shape[2] > 3:
            img_GT = img_GT[:, :, :3]
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        return_list.append(img_GT)
        # todo: 0 means fake 1 means true
        label = 0 if self.sp_folder in GT_path else 1
        label = torch.tensor(label).float()

        if self.with_mask and self.sp_folder in GT_path:
            # 只有splicing图像需要生成mask
            dir_name = os.path.dirname(GT_path)
            file_name = str(os.path.basename(GT_path)).split('.')[0]
            mask_path = os.path.join(dir_name, 'edgemask', f'{file_name}_edgemask.jpg')
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            mask = util.channel_convert(mask.shape[2], self.dataset_opt['color'], [mask])[0]
            # mask = self.transform_just_resize(image=copy.deepcopy(mask))["image"]
            mask = mask[:, :, [2, 1, 0]]

            # BRIGHT_GREEN = np.array([0, 255, 0])
            # REGULAR_GREEN = np.array([0, 200, 0])
            # 绿色区域即是拼接上去的区域

            mask_init = np.zeros((mask.shape[:2]), dtype=np.uint8)
            mask_green = (mask[:, :, 1] == 200)
            mask_init[mask_green] = 255
            # mask = util.channel_convert(mask.shape[2], self.dataset_opt['color'], [mask])[0]
            mask = self.transform_just_resize(image=copy.deepcopy(mask_init))["image"]
            mask = mask.astype(np.float32) / 255.
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = torch.from_numpy(np.ascontiguousarray(np.transpose(mask, (2, 0, 1)))).float()
            return_list.append(mask)
        elif self.with_mask and self.au_folder in GT_path:
            mask = torch.zeros_like(img_GT)
            return_list.append(mask)
        return_list.append(label)

        return return_list[:]

    def _list_cur_images(self, data_root):
        files = os.listdir(data_root)
        image_list = [os.path.join(data_root, i) for i in files if util.is_image_file(i)]
        return image_list

    def __len__(self):
        return len(self.paths_GT)

    # def to_tensor(self, img):
    #     img = Image.fromarray(img)
    #     img_t = F.to_tensor(img).float()
    #     return img_t

if __name__ == '__main__':
    import torchvision
    def print_this_image(image, filename):
        '''
            the input should be sized [C,H,W], not [N,C,H,W]
        '''
        camera_ready = image.unsqueeze(0)
        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                     filename, nrow=1,
                                     padding=0, normalize=False)
    dataset_opt = {'color':'RGB', 'GT_size':256}
    # 单元测试
    train_set = Columbia_dataset(dataset_opt=dataset_opt)
    test_image, test_mask, label = train_set[0]
    print(label)
    print_this_image(test_image, './image.png')
    print_this_image(test_mask, './mask.png')