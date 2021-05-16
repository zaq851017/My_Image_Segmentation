import torch
import torch.nn.functional as F
from torch import nn
import warnings
from network.Unet3D import UNet_3D
from network.Res_Unet import Single_Res_Unet_with_backbone, Single_Res_Unet
from network.Nested_Unet import Single_Nested_Unet
from network.DeepLab import DeepLab
from network.CLSTM import BDCLSTM
from torchvision import models
from network.Unet import UNetSmall
import segmentation_models_pytorch as smp
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
    def __init__(self, num_classes, Unet_3D_channel = 64, continue_num = 8):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.Temporal_Module = _Temporal_Module(num_classes, Unet_3D_channel)
        self.Segmentation_Module = Single_Nested_Unet(num_classes, input_channels = continue_num )
    def forward(self, input, other_frame):
        temporal_mask = self.Temporal_Module(input, other_frame).squeeze(dim = 1)
        predict = self.Segmentation_Module(temporal_mask)
        return temporal_mask, predict
class Two_Level_Res_Unet(nn.Module):
    def __init__(self, num_classes, Unet_3D_channel = 64, continue_num = 8):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.Temporal_Module = _Temporal_Module(num_classes, Unet_3D_channel)
        self.down =  nn.Conv3d(in_channels = continue_num, out_channels = 1, kernel_size=3, padding = 1)
        self.Segmentation_Module = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
    def forward(self, input, other_frame):
        temporal_mask = self.Temporal_Module(input, other_frame).squeeze(dim = 1)
        down = self.down(temporal_mask.unsqueeze(dim = 2)).squeeze(dim = 2)
        predict = self.Segmentation_Module(down)
        return temporal_mask, predict
class Two_Level_Deeplab(nn.Module):
    def __init__(self, num_classes, Unet_3D_channel = 64, continue_num = 8):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.Temporal_Module = _Temporal_Module(num_classes, Unet_3D_channel)
        self.down = nn.Conv2d(in_channels = continue_num, out_channels = 3, kernel_size=3, padding = 1)
        self.Segmentation_Module = DeepLab()
    def forward(self, input, other_frame):
        temporal_mask = self.Temporal_Module(input, other_frame).squeeze(dim = 1)
        down = self.down(temporal_mask)
        predict = self.Segmentation_Module(down)
        return temporal_mask, predict

class Two_Level_Res_Unet_with_backbone(nn.Module):
    def __init__(self, num_classes, Unet_3D_channel = 64, continue_num = 8):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.Temporal_Module = _Temporal_Module(num_classes, Unet_3D_channel)
        self.down =  nn.Conv3d(in_channels = continue_num, out_channels = 3, kernel_size=3, padding = 1)
        res = models.resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(res.children())[:-2])
        self.Segmentation_Module = Single_Res_Unet_with_backbone(num_classes)
    def forward(self, input, other_frame):
        temporal_mask = self.Temporal_Module(input, other_frame).squeeze(dim = 1)
        frame_feature = self.feature_extractor(input.squeeze(dim = 1))
        down = self.down(temporal_mask.unsqueeze(dim = 2)).squeeze(dim = 2)
        predict = self.Segmentation_Module(frame_feature, down)
        return temporal_mask, predict
class Unet_LSTM(nn.Module):
    def __init__(self, num_classes, continue_num = 8):
        super().__init__()
        #self.unet1 = UNetSmall(num_channels = 3, num_classes = 1)
        self.unet1 = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        #self.unet1.segmentation_head = nn.Sequential(*[self.unet1.segmentation_head[i] for i in range(0)])
        self.len = continue_num
        self.lstm = BDCLSTM(input_channels = 1, hidden_channels=[8])
    def forward(self, input, other_frame):
        predict_pre = []
        predict_next = []
        temporal_mask = torch.tensor([]).cuda()
        for i in range(int(self.len / 2)):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_pre.append(temp)
        for i in range(int(self.len / 2+1), self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_next.append(temp)
        predict_now = self.unet1(other_frame[:,self.len // 2:self.len // 2+1,:,:,:].squeeze(dim = 1))
        final_predict = self.lstm(predict_pre, predict_now, predict_next)
        for p_p in (predict_pre):
            temporal_mask = torch.cat((temporal_mask, p_p), dim = 1)
        temporal_mask = torch.cat((temporal_mask, predict_now), dim = 1)
        for p_n in (predict_next):
            temporal_mask = torch.cat((temporal_mask, p_n), dim = 1)
        # predict_pre = self.unet1(other_frame[:,0:1,:,:,:].squeeze(dim = 1))
        # predict_next = self.unet1(other_frame[:,1:2,:,:,:].squeeze(dim = 1))
        # predict_now = self.unet1(input.squeeze(dim = 1))
        # final_predict = self.lstm(predict_pre, predict_now, predict_next)
        return temporal_mask, final_predict
        