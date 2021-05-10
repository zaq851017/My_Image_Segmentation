import numpy as np
import os
import torch
import cv2
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import imageio
## need to remove before submit
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import logging
##net work
import segmentation_models_pytorch as smp
from network.Vgg_FCN8s import Single_vgg_FCN8s
from network.Vgg_Unet import Single_vgg_Unet
from network.Res_Unet import Single_Res_Unet
from network.Nested_Unet import Single_Nested_Unet
from network.DeepLab import DeepLab
from network.Unet3D import UNet_3D_Seg
from network.PraNet import PraNet
from network.Two_Level_Net import Two_Level_Nested_Unet, Two_Level_Res_Unet, Two_Level_Deeplab, Two_Level_Res_Unet_with_backbone, _Temporal_Module
from train_src.train_code import train_single, train_continuous, train_temporal
from train_src.dataloader import get_loader, get_continuous_loader
## loss
from train_src.loss_func import DiceBCELoss
  
def main(config):
    # parameter setting
    LR = config.learning_rate
    EPOCH = config.epoch
    BATCH_SIZE = config.batch_size
    if config.continuous == 0:
        frame_continue_num = 0
    else:
        frame_continue_num = list(map(int, config.continue_num))
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
        net = PraNet(1)
        model_name = "Single_PraNet"
        print("Model Single_PraNet")
    elif config.which_model == 5:
        net = DeepLab()
        model_name = "Single_DeepLab"
        print("Model Single_DeepLab")
    elif config.which_model == 11:
        net = Two_Level_Res_Unet(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Res_Unet"
        print("Model Two_Level_Res_Unet")
    elif config.which_model == 12:
        net = Two_Level_Nested_Unet(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Nested_Unet"
        print("Model Two_Level_Nested_Unet")
    elif config.which_model == 13:
        net = UNet_3D_Seg(1)
        model_name = "UNet_3D_Seg"
        print("Model UNet_3D_Seg")
    elif config.which_model == 14:
        net = Two_Level_Deeplab(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Deeplab"
        print("Two_Level_Deeplab")
    elif config.which_model == 15:
        net = Two_Level_Res_Unet_with_backbone(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Res_Unet_with_backbone"
        print("Two_Level_Res_Unet_with_backbone")
    elif config.which_model == -1:
        net = _Temporal_Module(1, config.Unet_3D_channel)
        model_name = "_Temporal_Module"
        print("_Temporal_Module")
    elif config.which_model == 0:
        print("No assign which model!")
    if config.pretrain_model != "":
        net.load_state_dict(torch.load(config.pretrain_model))
        print("pretrain model loaded!")
    if config.data_parallel == 1:
        net = nn.DataParallel(net)
    now_time = datetime.now().strftime("%Y_%m_%d_%I:%M:%S_")
    log_name = os.path.join('My_Image_Segmentation', 'log', now_time+"_"+model_name+"_"+str(frame_continue_num)+".log")
    print("log_name ", log_name)
    logging.basicConfig(level=logging.DEBUG,
                        handlers = [logging.FileHandler(log_name, 'w', 'utf-8'),logging.StreamHandler()])
    logging.info(sys.argv)
    logging.info(config)
    net = net.cuda()
    threshold = config.threshold
    best_score = config.best_score
    if config.loss_func == 0:
        train_weight = torch.FloatTensor([10 / 1]).cuda()
        criterion_single = nn.BCEWithLogitsLoss(pos_weight = train_weight)
        criterion_temporal = nn.BCEWithLogitsLoss(pos_weight = train_weight)
        logging.info("train weight = "+str(train_weight))
        logging.info("criterion_single = nn.BCEWithLogitsLoss()")
        logging.info("criterion_temporal = nn.BCEWithLogitsLoss()")
    elif config.loss_func == 1:
        train_weight = torch.FloatTensor([10 / 1]).cuda()
        criterion_single = DiceBCELoss(weight = train_weight)
        criterion_temporal = nn.BCEWithLogitsLoss(pos_weight = train_weight)
        logging.info("train weight = "+str(train_weight))
        logging.info("criterion_single = DiceBCELoss()")
        logging.info("criterion_temporal = nn.BCEWithLogitsLoss()")
    OPTIMIZER = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = LR)
    if config.continuous == 0:
        logging.info("Single image version")
        train_loader = get_loader(image_path = config.train_data_path,
                                batch_size = BATCH_SIZE,
                                mode = 'train',
                                augmentation_prob = config.augmentation_prob,
                                shffule_yn = True)
        valid_loader = get_loader(image_path = config.valid_data_path,
                                batch_size = 1,
                                mode = 'valid',
                                augmentation_prob = 0.,
                                shffule_yn = False)
        test_loader = get_loader(image_path = config.test_data_path,
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False)
        train_single(config, logging, net, model_name, threshold, best_score, criterion_single, OPTIMIZER, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR, now_time)
    elif config.continuous == 1:
        logging.info("Continuous image version")
        train_loader, continue_num = get_continuous_loader(image_path = config.train_data_path, 
                            batch_size = BATCH_SIZE,
                            mode = 'train',
                            augmentation_prob = config.augmentation_prob,
                            shffule_yn = True,
                            continue_num = frame_continue_num)
        valid_loader, continue_num = get_continuous_loader(image_path = config.valid_data_path,
                                batch_size = 1,
                                mode = 'valid',
                                augmentation_prob = 0.,
                                shffule_yn = False,
                                continue_num = frame_continue_num)
        test_loader, continue_num = get_continuous_loader(image_path = config.test_data_path,
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False,
                                continue_num = frame_continue_num)
        logging.info("temporal frame: "+str(continue_num))
        if config.which_model != -1:
            train_continuous(config, logging, net,model_name, threshold, best_score, criterion_single, criterion_temporal, OPTIMIZER, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR, continue_num, now_time)
        else:
            train_temporal(config, logging, net,model_name, threshold, best_score, criterion_single, criterion_temporal, OPTIMIZER, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR, continue_num, now_time)



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
    parser.add_argument('--train_data_path', type=str, default="Medical_data/train/")
    parser.add_argument('--valid_data_path', type=str, default="Medical_data/valid/")
    parser.add_argument('--test_data_path', type=str, default="Medical_data/test/")
    parser.add_argument('--augmentation_prob', type=float, default=0.0)
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--draw_temporal', type=int, default=0)
    parser.add_argument('--draw_image_path', type=str, default="Medical_data/test_image_output/")
    parser.add_argument('--Unet_3D_channel', type=int, default=64)
    parser.add_argument('--loss_func', type=int, default=0)
    parser.add_argument('--continue_num', nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_parallel', type=int, default=0)
    parser.add_argument('--random_train', type=int, default=0)
    config = parser.parse_args()
    main(config)
