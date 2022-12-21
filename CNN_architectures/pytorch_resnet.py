# -*- coding: utf-8 -*-
"""
From scratch implementation of the famous ResNet models.
The intuition for ResNet is simple and clear, but to code
it didn't feel super clear at first, even when reading Pytorch own
implementation. 

Video explanation: 
Got any questions leave a comment on youtube :)

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-12 Initial coding
"""

import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

import numpy as np

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, use_SRM=False, feat_concat=False,
                 just_feat_extract=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.use_SRM = use_SRM
        self.just_feat_extract = just_feat_extract
        self.feat_concat = feat_concat # num of external features: 256
        if self.use_SRM:
            ## bayar conv
            self.BayarConv2D = nn.Conv2d(image_channels, 3, 5, 1, padding=2, bias=False)
            self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
            self.bayar_mask[2, 2] = 0
            self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
            self.bayar_final[2, 2] = -1

            # ## srm conv
            # self.SRMConv2D = nn.Conv2d(image_channels, 9, 5, 1, padding=2, bias=False)
            # self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
            #
            # ##SRM filters (fixed)
            # for param in self.SRMConv2D.parameters():
            #     param.requires_grad = False
            #
            self.relu = nn.ELU()
            # image_channels = 12

        self.conv1 = nn.Conv2d(image_channels*2 if self.use_SRM else image_channels,
                               64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1)
        self.fc = nn.Linear(512 * 4 if not self.feat_concat else 512 * 4 + 256, num_classes)

        if self.just_feat_extract:
            self.fc_feat_extract = nn.Linear(512*4, 1024)
            self.embedding = nn.Embedding(1,1024)

    def forward(self, x, mid_feats_from_recovery=None):
        if self.use_SRM:
            self.BayarConv2D.weight.data *= self.bayar_mask
            self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1),-1)
            self.BayarConv2D.weight.data += self.bayar_final
            conv_bayar = self.BayarConv2D(x)
            # conv_srm = self.SRMConv2D(x)

            # x = torch.cat((conv_srm, conv_bayar), dim=1)
            x = torch.cat([x,self.relu(x)],dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        if self.just_feat_extract:
            return self.fc_feat_extract(x)

        if mid_feats_from_recovery is not None:
            x = torch.cat([x,mid_feats_from_recovery], dim=1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000, use_SRM=False, feat_concat=False, just_feat_extract=False):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes, use_SRM, feat_concat, just_feat_extract=just_feat_extract)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


def test():
    net = ResNet101(img_channel=3, num_classes=1000)
    y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    print(y.size())


# test()
