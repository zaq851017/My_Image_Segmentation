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
import warnings
import logging
##net work
from train_src.train_code import train_single, train_continuous, train_3D
from train_src.dataloader import get_loader, get_continuous_loader
from all_model import WHICH_MODEL
import random
## loss
from train_src.loss_func import DiceBCELoss, IOUBCELoss, Temporal_Loss
  
def main(config):
    warnings.filterwarnings('ignore')
    seed = 1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    # parameter setting
    LR = config.learning_rate
    EPOCH = config.epoch
    BATCH_SIZE = config.batch_size
    if config.continuous == 0:
        frame_continue_num = 0
    else:
        frame_continue_num = list(map(int, config.continue_num))
    net, model_name = WHICH_MODEL(config, frame_continue_num)
    if config.data_parallel == 1:
        net = nn.DataParallel(net)
    now_time = datetime.now().strftime("%Y_%m_%d_%I:%M:%S_")
    if not os.path.isdir(config.save_log_path):
        os.makedirs(config.save_log_path)
    log_name = os.path.join(config.save_log_path, now_time+"_"+model_name+"_"+str(frame_continue_num)+"_gamma="+str(config.gamma)+".log")
    print("log_name ", log_name)
    logging.basicConfig(level=logging.DEBUG,
                        handlers = [logging.FileHandler(log_name, 'w', 'utf-8'),logging.StreamHandler()])
    logging.info(sys.argv)
    logging.info(config)
    net = net.cuda()
    threshold = config.threshold
    best_score = config.best_score
    OPTIMIZER = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = LR)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, mode='max', factor=0.1, patience = 3)
    #scheduler = optim.lr_scheduler.StepLR(OPTIMIZER, step_size = 10, gamma = 0.3)
    scheduler = optim.lr_scheduler.MultiStepLR(OPTIMIZER, milestones=[21], gamma = 0.1)
    if config.loss_func == 0:
        train_weight = torch.FloatTensor([10 / 1]).cuda()
        criterion_single = IOUBCELoss(weight = train_weight)
        criterion_temporal = Temporal_Loss(weight = train_weight, gamma = config.gamma, distance = frame_continue_num)
        logging.info("train weight = "+str(train_weight))
        logging.info("criterion_single = "+str(criterion_single))
        logging.info("criterion_temporal = "+str(criterion_temporal))
    elif config.loss_func == 1:
        train_weight = torch.FloatTensor([10 / 1]).cuda()
        criterion_single = DiceBCELoss(weight = train_weight)
        criterion_temporal = DiceBCELoss(weight = train_weight)
        logging.info("train weight = "+str(train_weight))
        logging.info("criterion_single = "+str(criterion_single))
        logging.info("criterion_temporal = "+str(criterion_temporal))
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
        train_single(config, logging, net, model_name, threshold, best_score, criterion_single, OPTIMIZER,scheduler, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR, now_time)
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
        if config.which_model == -1 or config.which_model == -2:
            train_3D(config, logging, net,model_name, threshold, best_score, criterion_single, criterion_temporal, OPTIMIZER,scheduler, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR, continue_num, now_time)
        else:          
            train_continuous(config, logging, net,model_name, threshold, best_score, criterion_single, criterion_temporal, OPTIMIZER,scheduler, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR, continue_num, now_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--which_model', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_model_path', type=str, default="./My_Image_Segmentation/models/")
    parser.add_argument('--save_log_path', type=str, default="./My_Image_Segmentation/log/")
    parser.add_argument('--best_score', type=float, default=0.7)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--train_data_path', type=str, default="Medical_data/train/")
    parser.add_argument('--valid_data_path', type=str, default="Medical_data/valid/")
    parser.add_argument('--test_data_path', type=str, default="Medical_data/test/")
    parser.add_argument('--backbone', type=str, default="resnet34")
    parser.add_argument('--augmentation_prob', type=float, default=0.0)
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--draw_temporal', type=int, default=0)
    parser.add_argument('--draw_image_path', type=str, default="Medical_data/test_image_output/")
    parser.add_argument('--Unet_3D_channel', type=int, default= 8)
    parser.add_argument('--loss_func', type=int, default=0)
    parser.add_argument('--continue_num', nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_parallel', type=int, default=0)
    parser.add_argument('--random_train', type=int, default=0)
    parser.add_argument('--w_T_LOSS', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.0)
    config = parser.parse_args()
    main(config)
