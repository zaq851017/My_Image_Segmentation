import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import imageio
import imageio
import cv2
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time
from matplotlib import cm as CM
import copy
from train_src.dataloader import get_loader, get_continuous_loader
from predict_src.postprocess_src import test_wo_postprocess, test_w_postprocess
from network.Vgg_FCN8s import Single_vgg_FCN8s
from network.Vgg_Unet import Single_vgg_Unet
from network.Res_Unet import Single_Res_Unet
from network.Nested_Unet import Single_Nested_Unet
from network.Double_Unet import Single_Double_Unet
from network.Unet3D import UNet_3D_Seg
from network.Two_Level_Net import Two_Level_Nested_Unet, Two_Level_Res_Unet

def main(config):
    if config.continuous == 0:
        test_loader = get_loader(image_path = config.input_path,
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False)
    elif config.continuous == 1:
        test_loader = get_continuous_loader(image_path = config.input_path,
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False)
    if config.which_model == 1:
        net = Single_vgg_FCN8s(1)
        model_name = "Single_vgg__FCN8s"
        print("Model Single_vgg_FCN8s")
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
    net.load_state_dict(torch.load(config.model_path))
    net = net.cuda()
    if config.w_postprocess == 0 :
        test_wo_postprocess(config, test_loader, net)
    elif config.w_postprocess == 1 :
        test_w_postprocess(config, test_loader, net)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--which_model', type=int, default=0)
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--keep_image', type= int, default=1)
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--w_postprocess', type=int, default=0)
    parser.add_argument('--resize_image', type=int, default=0)
    parser.add_argument('--draw_temporal', type=int, default=0)
    config = parser.parse_args()
    main(config)