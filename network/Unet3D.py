import torch
import torch.nn as nn
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
    def __init__(self, num_class):
        super(UNet_3D, self).__init__()
        
        self.in_dim = 3
        self.out_dim = num_class
        activation = nn.ReLU()
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, 64, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(64, 128, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(128, 256, activation)
        self.pool_3 = max_pooling_3d()
        self.bridge = conv_block_2_3d(256, 512, activation)
        # Up sampling
        self.trans_1 = conv_trans_block_3d(512, 512, activation)
        self.up_1 = conv_block_2_3d(512+256, 256, activation)
        self.trans_2 = conv_trans_block_3d(256, 256, activation)
        self.up_2 = conv_block_2_3d(384, 128, activation)
        self.trans_3 = conv_trans_block_3d(128, 128, activation)
        self.up_3 = conv_block_2_3d(128+64, 64, activation)
        self.out = conv_block_3d(64, 1, activation)
        """
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        # Output
        self.out = conv_block_3d(self.num_filters, self.out_dim, activation)
        """
    
    def forward(self, other_frame):
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
        predict = self.out(up_3)
        result = torch.mean(predict, dim = 2)
        return result