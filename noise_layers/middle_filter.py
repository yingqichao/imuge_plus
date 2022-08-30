import torch.nn as nn
from kornia.filters import MedianBlur


class MiddleBlur(nn.Module):

	def __init__(self, kernel=5):
		super(MiddleBlur, self).__init__()
		self.middle_filter = MedianBlur((kernel, kernel))

	def forward(self, image):
		# image, cover_image = image_and_cover
		return self.middle_filter(image)

