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

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()
        vgg = models.vgg16(pretrained=True)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        self.features5 = nn.Sequential(*features)
    def forward(self, x):
        image_num = x.shape[1]
        output = torch.tensor([]).cuda()
        with torch.no_grad():
            for i in range(image_num):
                temp = self.features5(x[:,0:1,:,:,:].squeeze())
                temp = temp.view(-1, 1, 512, 16, 16)
                output = torch.cat((output, temp), dim = 1)
            return output
class Temporal_vgg_FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(Temporal_vgg_FCN8s, self).__init__()
        vgg = models.vgg16(pretrained=True)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        '''
        100 padding for 2 reasons:
            1) support very small input size
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        '''
        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()
        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore_pool4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

    def forward(self, x, other_frame):
        # pool3 = (256, 88, 88)
        # pool4 = (512, 44, 44)
        # pool5 = (512, 22, 22)
        image_num = other_frame.shape[1]
        output3 = torch.tensor([]).cuda()
        output4 = torch.tensor([]).cuda()
        output5 = torch.tensor([]).cuda()
        for i in range(image_num):
            temp = self.features3(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            temp = temp.view(-1, 1, 256, 70, 77)
            output3  = torch.cat((output3, temp), dim = 1)
        for i in range(image_num):
            temp = self.features4(output3[:,i:i+1,:,:,:].squeeze(dim = 1))
            temp = temp.view(-1, 1, 512, 35, 38)
            output4  = torch.cat((output4, temp), dim = 1)
        for i in range(image_num):
            temp = self.features5(output4[:,i:i+1,:,:,:].squeeze(dim = 1))
            temp = temp.view(-1, 1, 512, 17, 19)
            output5  = torch.cat((output5, temp), dim = 1)
        x = x.squeeze(dim = 1)
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)
        merge_pool3 = torch.cat((output3, pool3.unsqueeze(dim = 1)), dim = 1)
        merge_pool4 = torch.cat((output4, pool4.unsqueeze(dim = 1)), dim = 1)
        merge_pool5 = torch.cat((output5, pool5.unsqueeze(dim = 1)), dim = 1)
        merge_pool3 = torch.mean(merge_pool3, dim = 1)
        merge_pool4 = torch.mean(merge_pool4, dim = 1)
        merge_pool5 = torch.mean(merge_pool5, dim = 1)
        score_fr = self.score_fr(merge_pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * merge_pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)

        score_pool3 = self.score_pool3(0.0001 * merge_pool3)
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                 + upscore_pool4)
        predict = F.upsample(upscore8, x.size()[2:], mode='bilinear')
        return predict