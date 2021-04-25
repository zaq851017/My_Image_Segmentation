import torch
import torch.nn.functional as F
from torch import nn
import warnings
from network.Unet3D import UNet_3D
from network.Res_Unet import Temporal_Res_Unet, Single_Res_Unet
from network.Nested_Unet import Single_Nested_Unet
from torchvision import models
class _Temporal_Module(nn.Module):
    def __init__(self, num_classes, Unet_3D_channel = 64):
        super(_Temporal_Module, self).__init__()
        self.n_classes = num_classes
        self.ED = UNet_3D(1, Unet_3D_channel)
    def forward(self, x, other_frame):
        other_frame = other_frame.transpose(1, 2).contiguous()
        temporal_result = self.ED(other_frame)
        return temporal_result
class Two_Level_Nested_Unet(nn.Module):
    def __init__(self, num_classes, Unet_3D_channel = 64):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.Temporal_Module = _Temporal_Module(num_classes, Unet_3D_channel)
        self.Segmentation_Module = Single_Nested_Unet(num_classes, input_channels = 14)
    def forward(self, input, other_frame):
        temporal_mask = self.Temporal_Module(input, other_frame).squeeze(dim = 1)
        predict = self.Segmentation_Module(temporal_mask)
        return temporal_mask, predict
class Two_Level_Res_Unet(nn.Module):
    def __init__(self, num_classes, Unet_3D_channel = 64):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.Temporal_Module = _Temporal_Module(num_classes, Unet_3D_channel)
        self.down = nn.Conv2d(in_channels = 16, out_channels = 3, kernel_size=3, padding = 1)
        #res = models.resnet34(pretrained=True)
        #self.feature_extractor = nn.Sequential(*list(res.children())[:-2])
        self.Segmentation_Module = Single_Res_Unet(num_classes)
    def forward(self, input, other_frame):
        temporal_mask = self.Temporal_Module(input, other_frame).squeeze(dim = 1)
        #frame_feature = self.feature_extractor(input.squeeze(dim = 1))
        down = self.down(temporal_mask)
        predict = self.Segmentation_Module(down)
        return temporal_mask, predict