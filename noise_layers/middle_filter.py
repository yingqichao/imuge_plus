import torch.nn as nn
from kornia.filters import MedianBlur
from utils.metrics import PSNR

class MiddleBlur(nn.Module):

	def __init__(self, kernel=5):
		super(MiddleBlur, self).__init__()
		self.psnr = PSNR(255.0).cuda()
		self.middle_filters = [MedianBlur((3, 3)),
							   MedianBlur((5, 5)),
							   MedianBlur((7, 7))
							]

	def forward(self, tensor, kernel=5):
		# image, cover_image = image_and_cover
		result = tensor
		for idx, kernel in enumerate([3, 5, 7]):
			blur_result = self.middle_filters[idx](tensor)
			psnr = self.psnr(self.postprocess(blur_result), self.postprocess(tensor)).item()
			if psnr >= 28:
				return blur_result, kernel
		## if none of the above satisfy psnr>30, we abandon the attack
		# print("abandoned median blur, we cannot find a suitable kernel that satisfy PSNR>=25")
		return result, 0

	def postprocess(self, img):
		# [0, 1] => [0, 255]
		img = img * 255.0
		img = img.permute(0, 2, 3, 1)
		return img.int()


		# if kernel==3:
		# 	return self.middle_filter_3(image)
		# if kernel==5:
		# 	return self.middle_filter_5(image)
		# if kernel==7:
		# 	return self.middle_filter_7(image)
		# else:
		# 	raise NotImplementedError("Middleblur只支持3,5,7")

