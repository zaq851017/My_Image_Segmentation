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
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)
class Single_Res_Unet(nn.Module):
    def __init__(self, num_classes):
        super(Single_Res_Unet, self).__init__()
        warnings.filterwarnings('ignore')
        res = models.resnet50(pretrained=True)
        res_feature = nn.Sequential(*list(res.children())[:-1])
        self.features1 = nn.Sequential(*res_feature[:5])
        self.features2 = nn.Sequential(*res_feature[5:6])
        self.features3 = nn.Sequential(*res_feature[6:7])
        self.features4 = nn.Sequential(*res_feature[7:8])
        self.center = _DecoderBlock(2048, 4096, 2048)
        self.dec4 = _DecoderBlock(4096, 2048, 1024)
        self.dec3 = _DecoderBlock(2048, 1024, 512)
        self.dec2 = _DecoderBlock(1024, 512, 256)
        self.dec1 = _DecoderBlock(512, 256, 128)
        self.final = nn.Conv2d(128, num_classes, kernel_size=1)
    def forward(self, x):
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
class Temporal_Res_Unet(nn.Module):
    def __init__(self, num_classes):
        super(Temporal_Res_Unet, self).__init__()
        warnings.filterwarnings('ignore')
        res = models.resnet50(pretrained=True)
        res_feature = nn.Sequential(*list(res.children())[:-1])
        self.features1 = nn.Sequential(*res_feature[:5])
        self.features2 = nn.Sequential(*res_feature[5:6])
        self.features3 = nn.Sequential(*res_feature[6:7])
        self.features4 = nn.Sequential(*res_feature[7:8])
        self.center = _DecoderBlock(2048, 4096, 2048)
        self.dec4 = _DecoderBlock(4096, 2048, 1024)
        self.dec3 = _DecoderBlock(2048, 1024, 512)
        self.dec2 = _DecoderBlock(1024, 512, 256)
        self.dec1 = _DecoderBlock(512, 256, 128)
        self.final = nn.Conv2d(128, num_classes, kernel_size=1)
    def forward(self, x, other_frame):
        image_num = other_frame.shape[1]
        output1 = torch.tensor([]).cuda()
        output2 = torch.tensor([]).cuda()
        output3 = torch.tensor([]).cuda()
        output4 = torch.tensor([]).cuda()
        for i in range(image_num):
            temp = self.features1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            temp = temp.view(-1, 1, 256, 92, 106)
            output1  = torch.cat((output1, temp), dim = 1)
        for i in range(image_num):
            temp = self.features2(output1[:,i:i+1,:,:,:].squeeze(dim = 1))
            temp = temp.view(-1, 1, 512, 46, 53)
            output2  = torch.cat((output2, temp), dim = 1)
        for i in range(image_num):
            temp = self.features3(output2[:,i:i+1,:,:,:].squeeze(dim = 1))
            temp = temp.view(-1, 1, 1024, 23, 27)
            output3  = torch.cat((output3, temp), dim = 1)
        for i in range(image_num):
            temp = self.features4(output3[:,i:i+1,:,:,:].squeeze(dim = 1))
            temp = temp.view(-1, 1, 2048, 12, 14)
            output4  = torch.cat((output4, temp), dim = 1)
        x = x.squeeze(dim = 1)
        x_size = x.size()
        pool1 = self.features1(x)
        pool2 = self.features2(pool1)
        pool3 = self.features3(pool2)
        pool4 = self.features4(pool3)
        merge_pool1 = torch.cat((output1, pool1.unsqueeze(dim = 1)), dim = 1)
        merge_pool2 = torch.cat((output2, pool2.unsqueeze(dim = 1)), dim = 1)
        merge_pool3 = torch.cat((output3, pool3.unsqueeze(dim = 1)), dim = 1)
        merge_pool4 = torch.cat((output4, pool4.unsqueeze(dim = 1)), dim = 1)
        merge_pool1 = torch.mean(merge_pool1, dim = 1)
        merge_pool2 = torch.mean(merge_pool2, dim = 1)
        merge_pool3 = torch.mean(merge_pool3, dim = 1)
        merge_pool4 = torch.mean(merge_pool4, dim = 1)
        center = self.center(merge_pool4)
        dec4 = self.dec4(torch.cat([center, F.upsample(merge_pool4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(merge_pool3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(merge_pool2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(merge_pool1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        predict = F.upsample(final, x.size()[2:], mode='bilinear')
        return predict