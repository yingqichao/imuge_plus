import torch

from torch.fft import fft as rfft
from torch.fft import ifft2
################## OLD ################################################### NOW ##########
# torch.rfft(input, signal_ndim=2, normalized=False, onesided=False)	torch.fft.fft()
# torch.rfft(input, signal_ndim=2, normalized=False, onesided=True)	torch.fft.rfft()
# torch.irfft(input, signal_ndim=2, normalized=False, onesided=False)	torch.fft.ifft2()
# torch.irfft(input, signal_ndim=2, normalized=False, onesided=True)	torch.fft.irfft2()

# try:
#     from torch import irfft
#     from torch import rfft
# except ImportError:
#     from torch.fft import irfft2
#     from torch.fft import rfft2
#
#
#     def rfft(x, d):
#         t = rfft2(x, dim=(-d))
#         return torch.stack((t.real, t.imag), -1)
#
#
#     def irfft(x, d, signal_sizes):
#         return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d))


def calc_fft(image):
    '''image is tensor, N*C*H*W'''
    fft = rfft(image, 2)
    fft_mag = torch.log(1 + torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2 + 1e-8))
    return fft_mag


def fft_L1_loss(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
    real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114

    fake_fft = calc_fft(fake_image_gray)
    real_fft = calc_fft(real_image_gray)
    loss = criterion_L1(fake_fft, real_fft)
    return loss


def fft_L1_loss_mask(fake_image, real_image, mask):
    criterion_L1 = torch.nn.L1Loss()

    fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
    real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114

    fake_fft = calc_fft(fake_image_gray)
    real_fft = calc_fft(real_image_gray)
    print(fake_fft.shape)
    print(mask.shape)
    loss = criterion_L1(fake_fft * mask, real_fft * mask)
    return loss


def fft_L1_loss_color(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_fft = calc_fft(fake_image)
    real_fft = calc_fft(real_image)
    loss = criterion_L1(fake_fft, real_fft)
    return loss


def decide_circle(N=4, L=256, r=96, size=256):
    x = torch.ones((N, L, L))
    for i in range(L):
        for j in range(L):
            if (i - L / 2 + 0.5) ** 2 + (j - L / 2 + 0.5) ** 2 < r ** 2:
                x[:, i, j] = 0
    return x, torch.ones((N, L, L)) - x


if __name__ == '__main__':
    import cv2
    import numpy as np

    path1 = 'D:\CVPR_images\qian.jpg'
    path2 = 'D:\CVPR_images\zhangxinpeng.jpg'
    img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)

    img1 = img1.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.

    img1 = cv2.resize(np.copy(img1), (256, 256), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(np.copy(img2), (256, 256), interpolation=cv2.INTER_LINEAR)

    img1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img1, (2, 0, 1)))).float()
    img1 = img1.unsqueeze(0).cuda()

    img2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img2, (2, 0, 1)))).float()
    img2 = img2.unsqueeze(0).cuda()

    fft_loss = fft_L1_loss_color(img1, img2)
    print(fft_loss)