import random
import torch

def get_random_float(float_range: [float]):
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])


from .identity import Identity
from .crop import Crop, Cropout, Dropout
from .gaussian_noise import GN
from .middle_filter import MiddleBlur
from .gaussian_filter import GF
from .salt_pepper_noise import SaltPepper
from .jpeg import Jpeg, JpegSS, JpegMask, JpegTest
from .combined import Combined

import numpy as np


# Morphology Dilate
def Morphology_Dilate(img, Dil_time=1):
	H, W = img.shape

	# kernel
	MF = np.array(((0, 1, 0),
				   (1, 0, 1),
				   (0, 1, 0)), dtype=np.int)

	# each dilate time
	out = img.copy()
	for i in range(Dil_time):
		tmp = np.pad(out, (1, 1), 'edge')
		for y in range(1, H):
			for x in range(1, W):
				if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) >= 255:
					out[y, x] = 255

	return out


# Morphology Erode
def Morphology_Erode(img, Erode_time=1):
	H, W = img.shape
	out = img.copy()

	# kernel
	MF = np.array(((0, 1, 0),
				   (1, 0, 1),
				   (0, 1, 0)), dtype=np.int)

	# each erode
	for i in range(Erode_time):
		tmp = np.pad(out, (1, 1), 'edge')
		# erode
		for y in range(1, H):
			for x in range(1, W):
				if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) < 255 * 4:
					out[y, x] = 0

	return out


# Morphology Closing
def Morphology_Closing(img, time=1):
	out = Morphology_Dilate(img, Dil_time=time)
	out = Morphology_Erode(out, Erode_time=time)

	return out


# Opening morphology
def Morphology_Opening(img, time=1):
	out = Morphology_Erode(img, Erode_time=time)
	out = Morphology_Dilate(out, Dil_time=time)

	return out

