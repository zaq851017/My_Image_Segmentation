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
    if config.w_postprocess == 0 :
        test_wo_postprocess(config, test_loader)
    elif config.w_postprocess == 1 :
        test_w_postprocess(config, test_loader)

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
    config = parser.parse_args()
    main(config)