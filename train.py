import numpy as np
import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
from eval import *
from PIL import Image
import imageio
from mean_iou_evaluate import *
from loss_func import *
## need to remove before submit
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
##net work
from network.FCN32s import *
from network.HDC import *
from network.FCN8s import *
from network.Pspnet import *
from network.GCN import *
from network.T_FCN8s import *
from network.T_Res_Unet import *
from network.Vgg_Unet import *
from loss_func import *
from train_src.train_code import train_single, train_continuous
from train_src.dataloader import get_loader, get_continuous_loader
import segmentation_models_pytorch as smp
  
def main(config):
    print(config)
    # parameter setting
    LR = config.learning_rate
    EPOCH = config.epoch
    BATCH_SIZE = config.batch_size
    if config.which_model == 1:
        net = FCN32s(2)
        model_name = "FCN32S"
        print("Base model FCN32S")
    elif config.which_model == 2:
        net = HDC(2)
        model_name = "HDC"
        print("HDC model")
    elif config.which_model == 3:
        net = FCN8s(1)
        model_name = "FCN8S"
        print("Model FCN8S")
    elif config.which_model == 4:
        net = GCN(2)
        model_name = "GCN"
        print("GCN net")
    elif config.which_model == 5:
        net = smp.Unet('vgg16', encoder_weights='imagenet', classes=2)
        print("Unet Vgg16")
    elif config.which_model == 6:
        net = smp.PSPNet('vgg16', encoder_weights='imagenet', classes=2)
        print("PSPNet Vgg16")
    elif config.which_model == 7:
        net = T_FCN8s(1)
        model_name = "vgg_temporal_FCN"
        print("Model vgg-temporal Temporal_FCN8S")
    elif config.which_model == 8:
        net = T_Res_Unet(1)
        model_name = "res_temporal_Unet"
        print("Model res-temporal T_Res_Unet")
    elif config.which_model == 9:
        net = Vgg_Unet(1)
        model_name = "vgg_unet"
        print("Model vgg-unet Vgg_Unet")
    elif config.which_model == 0:
        print("No assign which model!")
    if config.pretrain_model != "":
        net.load_state_dict(torch.load(config.pretrain_model))
        print("pretrain model loaded!")
    net = net.cuda()
    threshold = config.threshold
    best_score = config.best_score
    train_weight = torch.FloatTensor([10 / 1]).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight = train_weight)
    OPTIMIZER = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = LR)
    if config.continuous == 0:
        print("Single image version")
        train_loader = get_loader(image_path = "Medical_data/train/",
                                batch_size = BATCH_SIZE,
                                mode = 'train',
                                augmentation_prob = 0.,
                                shffule_yn = True)
        valid_loader = get_loader(image_path = "Medical_data/valid/",
                                batch_size = 1,
                                mode = 'valid',
                                augmentation_prob = 0.,
                                shffule_yn = False)
        test_loader = get_loader(image_path = "Medical_data/valid/",
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False)
        train_single(config, net, threshold, best_score, criterion, OPTIMIZER, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR)
    elif config.continuous == 1:
        print("Continuous image version (1,10,20,30)")
        train_loader = get_continuous_loader(image_path = "Medical_data/train/", 
                            batch_size = BATCH_SIZE,
                            mode = 'train',
                            augmentation_prob = 0.,
                            shffule_yn = True)
        valid_loader = get_continuous_loader(image_path = "Medical_data/valid/",
                                batch_size = 1,
                                mode = 'valid',
                                augmentation_prob = 0.,
                                shffule_yn = False)
        test_loader = get_continuous_loader(image_path = "Medical_data/test/",
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False)
        train_continuous(config, net, threshold, best_score, criterion, OPTIMIZER, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model', type=str, default="")
    parser.add_argument('--which_model', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_model_path', type=str, default="./models/")
    parser.add_argument('--best_score', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--continuous', type=int, default=0)
    config = parser.parse_args()
    main(config)