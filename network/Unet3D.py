import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)

class UNet_3D(nn.Module):
    def __init__(self, num_class, Unet_3D_channel = 16):
        super(UNet_3D, self).__init__()
        warnings.filterwarnings('ignore')
        self.in_dim = 3
        self.out_dim = num_class
        model_size_num = Unet_3D_channel
        print("Unet_3D_channel", Unet_3D_channel)
        activation = nn.ReLU()
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, model_size_num, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(model_size_num, 2*model_size_num, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(2*model_size_num, 4*model_size_num, activation)
        self.pool_3 = max_pooling_3d()
        self.bridge = conv_block_2_3d(4*model_size_num, 8*model_size_num, activation)
        # Up sampling
        self.trans_1 = conv_trans_block_3d(8*model_size_num, 8*model_size_num, activation)
        self.up_1 = conv_block_2_3d(12*model_size_num, 4*model_size_num, activation)
        self.trans_2 = conv_trans_block_3d(4*model_size_num, 4*model_size_num, activation)
        self.up_2 = conv_block_2_3d(6*model_size_num, 2*model_size_num, activation)
        self.trans_3 = conv_trans_block_3d(2*model_size_num, 2*model_size_num, activation)
        self.up_3 = conv_block_2_3d(3*model_size_num, 1*model_size_num, activation)
        self.out = nn.Sequential(
        nn.Conv3d(1*model_size_num, 1, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(1))
 
    def forward(self, other_frame):
        down1 = self.down_1(other_frame)
        pool1 = self.pool_1(down1)
        down2 = self.down_2(pool1)
        pool2 = self.pool_2(down2)
        down3 = self.down_3(pool2)
        pool3 = self.pool_3(down3)
        bridge = self.bridge(pool3)
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, F.upsample(down3, trans_1.size()[2:])], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, F.upsample(down2, trans_2.size()[2:])], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, F.upsample(down1, trans_3.size()[2:])], dim=1)
        up_3 = self.up_3(concat_3)
        predict = self.out(up_3)
        result = F.upsample(predict, other_frame.size()[2:])
        return result
class UNet_3D_Seg(nn.Module):
    def __init__(self, num_class, Unet_3D_channel = 64, continue_num = 8):
        super(UNet_3D_Seg, self).__init__()
        warnings.filterwarnings('ignore')
        self.in_dim = 3
        self.out_dim = num_class
        model_size_num = Unet_3D_channel
        print("Unet_3D_channel", Unet_3D_channel)
        activation = nn.ReLU()
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, model_size_num, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(model_size_num, 2*model_size_num, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(2*model_size_num, 4*model_size_num, activation)
        self.pool_3 = max_pooling_3d()
        self.bridge = conv_block_2_3d(4*model_size_num, 8*model_size_num, activation)
        # Up sampling
        self.trans_1 = conv_trans_block_3d(8*model_size_num, 8*model_size_num, activation)
        self.up_1 = conv_block_2_3d(12*model_size_num, 4*model_size_num, activation)
        self.trans_2 = conv_trans_block_3d(4*model_size_num, 4*model_size_num, activation)
        self.up_2 = conv_block_2_3d(6*model_size_num, 2*model_size_num, activation)
        self.trans_3 = conv_trans_block_3d(2*model_size_num, 2*model_size_num, activation)
        self.up_3 = conv_block_2_3d(3*model_size_num, 1*model_size_num, activation)
        self.out = nn.Sequential(
        nn.Conv3d(1*model_size_num, 1, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(1))
        self.out2 = nn.Sequential(
        nn.Conv3d(1*model_size_num, 1, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(1))
        self.OUT = nn.Conv2d(8, 1, kernel_size=1)
        #self.OUT = nn.Conv3d(8, 1, kernel_size=1)
 
    def forward(self, input, other_frame):
        other_frame = other_frame.transpose(1, 2).contiguous()
        down1 = self.down_1(other_frame)
        pool1 = self.pool_1(down1)
        down2 = self.down_2(pool1)
        pool2 = self.pool_2(down2)
        down3 = self.down_3(pool2)
        pool3 = self.pool_3(down3)
        bridge = self.bridge(pool3)
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down3], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down2], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down1], dim=1)
        up_3 = self.up_3(concat_3)
        temporal_mask = self.out(up_3).squeeze(dim = 1)
        import ipdb; ipdb.set_trace()
        output = self.OUT(temporal_mask)
        return output