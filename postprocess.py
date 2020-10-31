import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
from dataloader import get_loader
from eval import *
from PIL import Image
import imageio
from mean_iou_evaluate import *
from loss_func import *
import imageio
import cv2
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time
##net work
from FCN32s import *
from HDC import *
from FCN8s import *

def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def frame2video(path):
    video_path = (path[:-6])
    videoWriter = cv2.VideoWriter(os.path.join(video_path,"video.avi"), cv2.VideoWriter_fourcc(*'DIVX'), 12, (1024, 512))
    for frame_files in  LISTDIR(path):
        if frame_files[-3:] == "jpg":
            full_path = os.path.join(path, frame_files)
            frame = cv2.imread(full_path)
            videoWriter.write(frame)
    videoWriter.release()

def test(config, test_loader):
    print(config)
    if config.which_model == 1:
        net = FCN32s(2)
        print("FCN32s load!")
    elif config.which_model == 2:
        net = HDC(2)
        print("HDC load")
    elif config.which_model == 3:
        net = FCN8s(2)
        print("FCN 8S load")
    net.load_state_dict(torch.load(config.model_path))
    net = net.cuda()
    net.eval()
    if not os.path.isdir(config.output_path):
        os.makedirs(config.output_path)
    with torch.no_grad():
        
        tStart = time.time()
        for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
            image = image.cuda()
            output = net(image)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            SR = torch.argmax(output, dim = 1).squeeze().cpu().data.numpy().astype("uint8")
            contours, hierarchy = cv2.findContours(SR, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            maege_image = np.zeros( (crop_image.shape[0],crop_image.shape[1]*2,3),dtype=np.uint8)
            dir_file = file_name[0][:-13]
            write_path = os.path.join(os.path.join(config.output_path, dir_file.split("_")[0]))
            write_path = os.path.join(os.path.join(write_path, dir_file))
            if not os.path.isdir(write_path):
                os.makedirs(write_path+"/merge")
                os.makedirs(write_path+"/origin")
                os.makedirs(write_path+"/forfilm")
            if contours ==[]:
                maege_image = np.concatenate( (origin_crop_image, crop_image), axis = 1)
                imageio.imwrite(os.path.join(write_path+"/merge", file_name[0]), maege_image)
                imageio.imwrite(os.path.join(write_path+"/origin", file_name[0]), origin_crop_image)
                imageio.imwrite(os.path.join(write_path+"/forfilm", file_name[0]), crop_image)
            else:
                cv2.drawContours(np.uint8(crop_image), contours, -1, (0,255,0), 3)
                maege_image = np.concatenate( (origin_crop_image, np.uint8(crop_image)), axis = 1)
                imageio.imwrite(os.path.join(write_path+"/merge", file_name[0]), maege_image)
                imageio.imwrite(os.path.join(write_path+"/origin", file_name[0]), origin_crop_image)
                imageio.imwrite(os.path.join(write_path+"/forfilm", file_name[0]), crop_image)
        tEnd = time.time()
        print("Cost time(seconds)= "+str(tEnd-tStart))
        
        for dir_files in (LISTDIR(config.output_path)):
            full_path = os.path.join(config.output_path, dir_files)
            for num_files in tqdm(LISTDIR(full_path)):
                full_path_2 = os.path.join(full_path, num_files+"/merge")
                frame2video(full_path_2)
def main(config):
    # parameter setting
    test_loader = get_loader(image_path = config.input_path,
                            batch_size = 1,
                            mode = 'test',
                            augmentation_prob = 0.,
                            shffule_yn = False)
    test(config, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--which_model', type=int, default=3)
    parser.add_argument('--output_path', type=str, default="./all_test_results/")
    parser.add_argument('--input_path', type=str, default="./all_test_dataset/")
    config = parser.parse_args()
    main(config)