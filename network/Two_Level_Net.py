import torch
import torch.nn.functional as F
from torch import nn
import warnings
from network.Unet3D import UNet_3D
from network.Res_Unet import Single_Res_Unet
from network.Nested_Unet import Single_Nested_Unet
class _Temporal_Module(nn.Module):
    def __init__(self, num_classes):
        super(_Temporal_Module, self).__init__()
        self.n_classes = num_classes
        self.ED = UNet_3D(1)
    def forward(self, x, other_frame):
        other_frame = other_frame.transpose(1, 2).contiguous()
        temporal_result = self.ED(other_frame)
        return temporal_result
class Two_Level_Nested_Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.Temporal_Module = _Temporal_Module(num_classes)
        self.Segmentation_Module = Single_Nested_Unet(num_classes, input_channels = 8)
    def forward(self, input, other_frame):
        temporal_mask = self.Temporal_Module(input, other_frame).squeeze(dim = 1)
        predict = self.Segmentation_Module(temporal_mask)
        return temporal_mask, predict
class Two_Level_Res_Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.Temporal_Module = _Temporal_Module(num_classes)
        self.down = nn.Conv2d(in_channels = 8, out_channels = 3, kernel_size=3, padding = 1)
        self.Segmentation_Module = Single_Res_Unet(num_classes)
    def forward(self, input, other_frame):
        temporal_mask = self.Temporal_Module(input, other_frame).squeeze(dim = 1)
        down = self.down(temporal_mask)
        predict = self.Segmentation_Module(down)
        return temporal_mask, predict