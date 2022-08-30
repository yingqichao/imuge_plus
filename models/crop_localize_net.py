import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision

class CropLocalizeNet(nn.Module):
    def __init__(self,backbone_patch,patch_embedding_size,location_classes):
        super(CropLocalizeNet, self).__init__()
        self._backbone_patch = self._create_resnet(backbone_patch, 3, patch_embedding_size)
        self._location_net = torch.nn.Linear(
            patch_embedding_size, location_classes)

    def forward(self, patch):
        embedding = self._backbone_patch(patch)  # (B, 64)
        location = self._location_net(embedding)  # (B, 16)
        return embedding, location


    def _create_resnet(self, name, input_channels, output_size):
        name = name.lower()
        if name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = torch.nn.Conv2d(
                input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.fc = torch.nn.Linear(512, output_size)
        elif name == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False)
            model.conv1 = torch.nn.Conv2d(
                input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.fc = torch.nn.Linear(512, output_size)
        elif name == 'none':
            model = None
        else:
            raise ValueError('Unknown resnet backbone.')
        return model

