import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
# from layers import bn, VarianceLayer, CovarianceLayer, GrayscaleLayer
# from downsampler import *
from torch.nn import functional


class LossTracer(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(LossTracer, self).__init__()
        self.loss_pool = []

        self.ce_loss = nn.CrossEntropyLoss().cuda()
        self.bce_loss = nn.BCELoss().cuda()
        self.bce_with_logit_loss = nn.BCEWithLogitsLoss().cuda()
        self.l1_loss = nn.SmoothL1Loss(beta=0.5).cuda()  # reduction="sum"
        self.consine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.hard_l1_loss = nn.L1Loss().cuda()  # reduction="sum"
        self.l2_loss = nn.MSELoss().cuda()  # reduction="sum"
        # self.perceptual_loss = PerceptualLoss().cuda()
        # self.style_loss = StyleLoss().cuda()

    def forward(self, x, y):
        pass