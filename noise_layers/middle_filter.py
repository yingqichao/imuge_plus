import torch.nn as nn
from kornia.filters import MedianBlur


class MiddleBlur(nn.Module):

	def __init__(self, kernel=5):
		super(MiddleBlur, self).__init__()
		self.middle_filter_3 = MedianBlur((3, 3))
		self.middle_filter_5 = MedianBlur((5, 5))
		self.middle_filter_7 = MedianBlur((7, 7))

	def forward(self, image, kernel=5):
		# image, cover_image = image_and_cover
		if kernel==3:
			return self.middle_filter_3(image)
		if kernel==5:
			return self.middle_filter_5(image)
		if kernel==7:
			return self.middle_filter_7(image)
		else:
			raise NotImplementedError("Middleblur只支持3,5,7")

