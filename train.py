import numpy as np
import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import imageio
## need to remove before submit
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
##net work
from network.Vgg_FCN8s import Single_vgg_FCN8s
from network.Vgg_Unet import Single_vgg_Unet
from network.Res_Unet import Single_Res_Unet
from network.Nested_Unet import Single_Nested_Unet
from network.Double_Unet import Single_Double_Unet
from network.Unet3D import UNet_3D_Seg
from network.Two_Level_Net import Two_Level_Nested_Unet, Two_Level_Res_Unet
from train_src.train_code import train_single, train_continuous
from train_src.dataloader import get_loader, get_continuous_loader
import segmentation_models_pytorch as smp
  
def main(config):
    # parameter setting
    LR = config.learning_rate
    EPOCH = config.epoch
    BATCH_SIZE = config.batch_size
    if config.which_model == 1:
        net = Single_vgg_FCN8s(1)
        model_name = "Single_vgg__FCN8s"
        print("Model Single_vgg__FCN8s")
    elif config.which_model == 2:
        net = Single_vgg_Unet(1)
        model_name = "Single_vgg_Unet"
        print("Model Single_vgg_Unet")
    elif config.which_model == 3:
        net = Single_Res_Unet(1)
        model_name = "Single_Res_Unet"
        print("Model Single_Res_Unet")
    elif config.which_model == 4:
        net = Single_Nested_Unet(1)
        model_name = "Single_Nested_Unet"
        print("Model Single_Nested_Unet")
    elif config.which_model == 5:
        net = Single_Double_Unet(1)
        model_name = "Single_Double_Unet"
        print("Model Single_Double_Unet")
    elif config.which_model == 11:
        net = Two_Level_Res_Unet(1)
        model_name = "Two_Level_Res_Unet"
        print("Model Two_Level_Res_Unet")
    elif config.which_model == 12:
        net = Two_Level_Nested_Unet(1)
        model_name = "Two_Level_Nested_Unet"
        print("Model Two_Level_Nested_Unet")
    elif config.which_model == 13:
        net = UNet_3D_Seg(1)
        model_name = "UNet_3D_Seg"
        print("Model UNet_3D_Seg")
    elif config.which_model == 0:
        print("No assign which model!")
    if config.pretrain_model != "":
        net.load_state_dict(torch.load(config.pretrain_model))
        print("pretrain model loaded!")
    #net = nn.DataParallel(net)
    net = net.cuda()
    threshold = config.threshold
    best_score = config.best_score
    train_weight = torch.FloatTensor([10 / 1]).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight = train_weight)
    OPTIMIZER = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = LR)
    if config.resize_image == 0:
        crop_range_num = [150, 574, 70, 438]
    elif config.resize_image == 1:
        crop_range_num = [150, 574, 70, 282]
    if config.continuous == 0:
        print("Single image version")
        train_loader = get_loader(image_path = "Medical_data/train/",
                                batch_size = BATCH_SIZE,
                                mode = 'train',
                                augmentation_prob = config.augmentation_prob,
                                shffule_yn = True,
                                crop_range = crop_range_num)
        valid_loader = get_loader(image_path = "Medical_data/valid/",
                                batch_size = 1,
                                mode = 'valid',
                                augmentation_prob = 0.,
                                shffule_yn = False,
                                crop_range = crop_range_num)
        test_loader = get_loader(image_path = "Medical_data/valid/",
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False,
                                crop_range = crop_range_num)
        train_single(config, net, model_name, threshold, best_score, criterion, OPTIMIZER, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR)
    elif config.continuous == 1:
        print("Continuous image version (1,10,20,30)")
        train_loader = get_continuous_loader(image_path = "Medical_data/train/", 
                            batch_size = BATCH_SIZE,
                            mode = 'train',
                            augmentation_prob = config.augmentation_prob,
                            shffule_yn = True,
                            crop_range = crop_range_num)
        valid_loader = get_continuous_loader(image_path = "Medical_data/valid/",
                                batch_size = 1,
                                mode = 'valid',
                                augmentation_prob = 0.,
                                shffule_yn = False,
                                crop_range = crop_range_num)
        test_loader = get_continuous_loader(image_path = "Medical_data/test/",
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False,
                                crop_range = crop_range_num)
        train_continuous(config, net,model_name, threshold, best_score, criterion, OPTIMIZER, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model', type=str, default="")
    parser.add_argument('--which_model', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_model_path', type=str, default="./My_Image_Segmentation/models/")
    parser.add_argument('--best_score', type=float, default=0.7)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--augmentation_prob', type=float, default=0.0)
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--draw_image', type=int, default=0)
    parser.add_argument('--draw_image_path', type=str, default="")
    parser.add_argument('--resize_image', type=int, default=0)
    config = parser.parse_args()
    main(config)