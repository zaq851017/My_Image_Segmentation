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

class T_Res_FCN(nn.Module):
    def __init__(self, num_classes):
        super(T_Res_FCN, self).__init__()
        res = models.resnet50(pretrained=True)
        res_feature = nn.Sequential(*list(res.children())[:-1])
        res_feature[0].padding = (100, 100)
        self.features5 = nn.Sequential(*res_feature)
    def forward(self, x, other_frame):
        # pool3 = (256, 88, 88)
        # pool4 = (512, 44, 44)
        # pool5 = (512, 22, 22)
        image_num = other_frame.shape[1]
        output5 = torch.tensor([]).cuda()
        for i in range(image_num):
            temp = self.features5(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            temp = temp.view(-1, 1, 2048, 1, 1)
            output5 = torch.cat((output5, temp), dim = 1)
        import ipdb; ipdb.set_trace()
        """
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
        return (upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous())
        """