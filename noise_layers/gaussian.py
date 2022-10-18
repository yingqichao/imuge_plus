import torch
import torch.nn as nn

class Gaussian(nn.Module):
    '''Adds random noise to a tensor.'''

    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, tensor, cover_image=None, mean=0, stddev=1.0):

        self.name="Gaussian"
        noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).cuda(), mean, stddev)
        mixed = tensor + noise
        mixed = torch.clamp(mixed, 0, 1)

        return mixed

