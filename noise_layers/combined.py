from . import Identity
import torch.nn as nn
from . import get_random_int


class Combined(nn.Module):

	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Identity()]
		self.list = list
		self.name = "NotChosenYet"

	def forward(self, image_and_cover, id=None):
		if id is None or id>=len(self.list):
			id = get_random_int([0, len(self.list) - 1])
		selected = self.list[id]
		self.name = selected.name
		return selected(image_and_cover)
