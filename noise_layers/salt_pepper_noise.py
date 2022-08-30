import torch
import torch.nn as nn


class SaltPepper(nn.Module):

	def __init__(self, prob):
		super(SaltPepper, self).__init__()
		self.prob = prob

	def sp_noise(self, image, prob):
		prob_zero = prob / 2
		prob_one = 1 - prob_zero
		rdn = torch.rand(image.shape).to(image.device)

		output = torch.where(rdn > prob_one, torch.zeros_like(image).to(image.device), image)
		output = torch.where(rdn < prob_zero, torch.ones_like(output).to(output.device), output)

		return output

	def forward(self, image):
		# image, cover_image = image_and_cover
		return self.sp_noise(image, self.prob)
