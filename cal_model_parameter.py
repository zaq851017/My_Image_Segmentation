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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_model', type=int, default=0)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--continue_num', nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--backbone', type=str, default="resnet34")
    parser.add_argument('--w_T_LOSS', type=int, default=1)
    parser.add_argument('--Unet_3D_channel', type=int, default= 8)
    config = parser.parse_args()
    if config.continuous == 0:
        frame_continue_num = 0
    else:
        frame_continue_num = list(map(int, config.continue_num))
    net, model_name = WHICH_MODEL(config, frame_continue_num)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))