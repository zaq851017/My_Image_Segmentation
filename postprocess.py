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
from all_model import WHICH_MODEL
import random
def main(config):
    seed = 1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    with torch.no_grad():
        frame_continue_num = list(map(int, config.continue_num))
        print(len(frame_continue_num))
        if config.continuous == 0:
            test_loader = get_loader(image_path = config.input_path,
                                    batch_size = 1,
                                    mode = 'test',
                                    augmentation_prob = 0.,
                                    shffule_yn = False)
        elif config.continuous == 1:
            test_loader, continue_num = get_continuous_loader(image_path = config.input_path,
                                    batch_size = 1,
                                    mode = 'test',
                                    augmentation_prob = 0.,
                                    shffule_yn = False,
                                    continue_num = frame_continue_num)
        net, model_name = WHICH_MODEL(config, frame_continue_num)
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
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--keep_image', type= int, default=1)
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--w_postprocess', type=int, default=0)
    parser.add_argument('--resize_image', type=int, default=0)
    parser.add_argument('--draw_temporal', type=int, default=0)
    parser.add_argument('--Unet_3D_channel', type=int, default=64)
    parser.add_argument('--continue_num', nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--w_T_LOSS', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="resnet34")
    config = parser.parse_args()
    main(config)