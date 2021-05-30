import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.vgg import VGG
import warnings

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)

class Single_vgg_Unet(nn.Module):
    def __init__(self, num_classes):
        warnings.filterwarnings('ignore')
        super(Single_vgg_Unet, self).__init__()
        vgg = models.vgg16(pretrained=True)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        self.features1 = nn.Sequential(*features[: 5])
        self.features2 = nn.Sequential(*features[5: 10])
        self.features3 = nn.Sequential(*features[10: 17])
        self.features4 = nn.Sequential(*features[17:])
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
    def forward(self, x):
        # pool1 = (64, 184, 212)
        # pool2 = (128, 92, 106)
        # pool3 = (256, 46, 53)
        # pool4 = (512, 11, 13)
        x = x.squeeze(dim = 1)
        x_size = x.size()
        pool1 = self.features1(x)
        pool2 = self.features2(pool1)
        pool3 = self.features3(pool2)
        pool4 = self.features4(pool3)
        center = self.center(pool4)
        dec4 = self.dec4(torch.cat([center, F.upsample(pool4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(pool3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(pool2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(pool1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        predict = F.upsample(final, x.size()[2:], mode='bilinear')
        return predict