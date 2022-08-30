import numpy as np
import torch
import torch.nn as nn

import math

def diff_round(input_tensor):
    # input_tensor must be within [0,1]
    input_tensor = input_tensor*255.
    test = 0
    for n in range(1, 10):
        test += math.pow(-1, n+1) / n * torch.sin(2 * math.pi * n * input_tensor)
    final_tensor = input_tensor - 1 / math.pi * test
    return final_tensor/255.


class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)
