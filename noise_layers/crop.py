import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class Crop(nn.Module):

	def __init__(self):
		super(Crop, self).__init__()

	def get_random_rectangle_inside(self, image_shape, height_ratio, width_ratio):
		image_height = image_shape[2]
		image_width = image_shape[3]

		remaining_height = int(height_ratio * image_height)
		remaining_width = int(width_ratio * image_width)

		if remaining_height == image_height:
			height_start = 0
		else:
			height_start = np.random.randint(0, image_height - remaining_height)

		if remaining_width == image_width:
			width_start = 0
		else:
			width_start = np.random.randint(0, image_width - remaining_width)

		return height_start, height_start + remaining_height, width_start, width_start + remaining_width

	def forward(self, image, apex=None, min_rate=0.5, max_rate=1.0):
		if min_rate:
			self.height_ratio = min_rate + (max_rate-min_rate) * np.random.rand()
			self.width_ratio = min_rate + (max_rate-min_rate) * np.random.rand()
		else:
			self.height_ratio = 0.3 + 0.7 * np.random.rand()
			self.width_ratio = 0.3 + 0.7 * np.random.rand()
		self.height_ratio = min(self.height_ratio,self.width_ratio+0.2)
		self.width_ratio = min(self.width_ratio,self.height_ratio+0.2)
		# image, cover_image = image_and_cover
		if apex is not None:
			h_start, h_end, w_start, w_end = apex
		else:
			h_start, h_end, w_start, w_end = self.get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)

		new_images = image[:, :, h_start: h_end, w_start: w_end]

		scaled_images = F.interpolate(
			new_images,
			size=[image.shape[2], image.shape[3]],
			mode='bilinear')

		return scaled_images, (h_start, h_end, w_start, w_end) #, scaled_images #(h_start/image.shape[2], h_end/image.shape[2], w_start/image.shape[3], w_end/image.shape[3])   #mask[:,0,:,:].unsqueeze(1)

	def cropped_for_outpainting(self, image, real_H, min_rate=0.5, min_rate_2=0.7, apex=None):

		self.height_ratio = min_rate + (1 - min_rate) * np.random.rand()
		self.width_ratio = min_rate + (1 - min_rate) * np.random.rand()
		self.height_ratio = min(self.height_ratio, self.width_ratio + 0.3)
		self.width_ratio = min(self.width_ratio, self.height_ratio + 0.3)
		h_start, h_end, w_start, w_end = self.get_random_rectangle_inside(image.shape, self.height_ratio,self.width_ratio)
		new_images = image[:, :, h_start: h_end, w_start: w_end]

		zero_images = torch.zeros_like(new_images)
		self.height_ratio = min_rate_2 + (1 - min_rate_2) * np.random.rand()
		self.width_ratio = min_rate_2 + (1 - min_rate_2) * np.random.rand()
		self.height_ratio = min(self.height_ratio, self.width_ratio + 0.3)
		self.width_ratio = min(self.width_ratio, self.height_ratio + 0.3)
		h_start_1, h_end_1, w_start_1, w_end_1 = self.get_random_rectangle_inside(new_images.shape, self.height_ratio,self.width_ratio)
		zero_images[:, :, h_start_1: h_end_1, w_start_1: w_end_1] = image[:, :, h_start_1: h_end_1, w_start_1: w_end_1]

		GT = real_H[:, :, h_start: h_end, w_start: w_end]

		return new_images, zero_images, GT  # , scaled_images #(h_start/image.shape[2], h_end/image.shape[2], w_start/image.shape[3], w_end/image.shape[3])   #mask[:,0,:,:].unsqueeze(1)

	def cropped_out(self, image, apex=None, min_rate=None, max_rate=1.0):
		mask = torch.ones_like(image)
		if min_rate:
			self.height_ratio = min_rate + (max_rate-min_rate) * np.random.rand()
			self.width_ratio = min_rate + (max_rate-min_rate) * np.random.rand()
		else:
			self.height_ratio = 0.3 + 0.7 * np.random.rand()
			self.width_ratio = 0.3 + 0.7 * np.random.rand()
		self.height_ratio = min(self.height_ratio,self.width_ratio+0.3)
		self.width_ratio = min(self.width_ratio,self.height_ratio+0.3)
		# image, cover_image = image_and_cover
		if apex is not None:
			h_start, h_end, w_start, w_end = apex
			h_start, h_end, w_start, w_end = int(h_start*image.shape[2]), int(h_end*image.shape[2]), int(w_start*image.shape[3]), int(w_end*image.shape[3])
		else:
			h_start, h_end, w_start, w_end = self.get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)

		new_images = image[:, :, h_start: h_end, w_start: w_end]
		# zero_images[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
		mask[:, :, h_start: h_end, w_start: w_end] = 0
		zero_images = image * (1-mask)
		scaled_images = F.interpolate(
			new_images,
			size=[image.shape[2], image.shape[3]],
			mode='bilinear')
		scaled_images = torch.clamp(scaled_images,0,1)

		###### dual reshape loss ############
		scaled_back = F.interpolate(
			scaled_images,
			size=[h_end-h_start, w_end-w_start],
			mode='bilinear')
		scaled_back = torch.clamp(scaled_back, 0, 1)
		zero_images_GT = torch.zeros_like(image)
		zero_images_GT[:, :, h_start: h_end, w_start: w_end] = scaled_back
		dual_reshape_diff = (zero_images_GT-zero_images).clone().detach()

		zero_images = zero_images + dual_reshape_diff

		return scaled_images, zero_images, mask, (h_start/image.shape[2], h_end/image.shape[2], w_start/image.shape[3], w_end/image.shape[3]), new_images  #mask[:,0,:,:].unsqueeze(1)


class Cropout(nn.Module):

	def __init__(self, height_ratio, width_ratio):
		super(Cropout, self).__init__()
		self.height_ratio = height_ratio
		self.width_ratio = width_ratio

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = self.get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		cover_image[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
		return cover_image

class Dropout(nn.Module):

	def __init__(self, prob=0.5):
		super(Dropout, self).__init__()
		self.prob = prob

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		rdn = torch.rand(image.shape).to(image.device)
		output = torch.where(rdn > self.prob * 1., cover_image, image)
		return output
