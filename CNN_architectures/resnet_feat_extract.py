import torch
import torch.nn as nn
from CNN_architectures.pytorch_resnet import block
import numpy as np

class ResNet_feat_extract(nn.Module):
    def __init__(self, block=block, layers=[3, 4, 6, 3], image_channels=3):
        super(ResNet_feat_extract, self).__init__()
        self.in_channels = 64



        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


        self.fc_feat_extract = nn.Linear(512*4, 1024)
        self.fc_classification = nn.Linear(1024, 1)
        self.embedding = nn.Parameter(torch.ones(1, 1024))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1)
        self.l1_loss = nn.SmoothL1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def feat_extract(self, x):

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

        return self.fc_feat_extract(x)

    def forward(self, attacked_positive, attacked_tampered_negative):

        anchor = self.embedding.data.repeat(attacked_positive.shape[0], 1).detach()
        feat_positive = self.feat_extract(attacked_positive)
        feat_negative = self.feat_extract(attacked_tampered_negative)

        loss_triplet = self.triplet_loss(anchor, feat_positive, feat_negative)
        loss_l1 = self.l1_loss(feat_positive, anchor)
        loss = loss_triplet+loss_l1

        ### classification loss
        label = torch.cat([torch.zeros((feat_negative.shape[0],1)), torch.ones((feat_positive.shape[0],1))],dim=0).cuda()
        predicted_label = torch.cat([self.fc_classification(feat_negative),self.fc_classification(feat_positive)],dim=0)
        loss_class = self.bce_loss(predicted_label, label)
        loss += loss_class

        rate = 0.9
        self.embedding.data.mul_(rate).add_(torch.mean(feat_positive,dim=0,keepdim=True), alpha=1 - rate)
        return (loss, loss_triplet, loss_class), (anchor,feat_positive,feat_negative)



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
