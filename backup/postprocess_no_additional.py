import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
import imageio
from mean_iou_evaluate import *
import imageio
import cv2
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time
from matplotlib import cm as CM
import copy
from network.Single_vgg_FCN8s import Single_vgg_FCN8s
from network.Single_vgg_Unet import Single_vgg_Unet
from network.Single_Res_Unet import Single_Res_Unet
from network.Single_Nested_Unet import Single_Nested_Unet
from network.Single_Double_Unet import Single_Double_Unet
from network.Temporal_vgg_FCN8s import Temporal_vgg_FCN8s
from network.Temporal_vgg_Unet import Temporal_vgg_Unet
from network.Temporal_Res_Unet import Temporal_Res_Unet
from train_src.train_code import train_single, train_continuous
from train_src.dataloader import get_loader, get_continuous_loader
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def frame2video(path):
    video_path = (path[:-6])
    videoWriter = cv2.VideoWriter(os.path.join(video_path,"video.avi"), cv2.VideoWriter_fourcc(*'DIVX'), 12, (848, 368))
    for frame_files in  LISTDIR(path):
        if frame_files[-3:] == "jpg":
            full_path = os.path.join(path, frame_files)
            frame = cv2.imread(full_path)
            videoWriter.write(frame)
    videoWriter.release()
def test(config, test_loader):
    Sigmoid_func = nn.Sigmoid()
    threshold = config.threshold
    Sigmoid_func = nn.Sigmoid()
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
    elif config.which_model == 6:
        net = Temporal_vgg_FCN8s(1)
        model_name = "Temporal_vgg_FCN8s"
        print("Model Temporal_vgg_FCN8s")
    elif config.which_model == 7:
        net = Temporal_vgg_Unet(1)
        model_name = "Temporal_vgg_Unet"
        print("Model Temporal_vgg_Unet")
    elif config.which_model == 8:
        net = Temporal_Res_Unet(1)
        model_name = "Temporal_Res_Unet"
        print("Model Temporal_Res_Unet")
    net.load_state_dict(torch.load(config.model_path))
    net = net.cuda()
    net.eval()
    if not os.path.isdir(config.output_path):
        os.makedirs(config.output_path)
    with torch.no_grad():
        tStart = time.time()
        for i, (crop_image ,file_name, image_list) in tqdm(enumerate(test_loader)):
            pn_frame = image_list[:,1:,:,:,:]
            frame = image_list[:,:1,:,:,:]
            output = net(frame, pn_frame).squeeze(dim = 1)
            output = Sigmoid_func(output)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy().astype("uint8")
            heatmap = np.uint8(255 * SR)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heat_img = heatmap*0.9+origin_crop_image
            temp = [config.output_path] + file_name[0].split("/")[2:-2]
            write_path = "/".join(temp)
            img_name = file_name[0].split("/")[-1]
            if not os.path.isdir(write_path+"/original"):
                os.makedirs(write_path+"/original")
            if not os.path.isdir(write_path+"/forfilm"):
                os.makedirs(write_path+"/forfilm")
            if not os.path.isdir(write_path+"/merge"):
                os.makedirs(write_path+"/merge")
            if not os.path.isdir(write_path+"/vol_mask"):
                os.makedirs(write_path+"/vol_mask")
            merge_img = np.hstack([origin_crop_image, heat_img])
            cv2.imwrite(os.path.join(write_path+"/merge", img_name), merge_img)
            imageio.imwrite(os.path.join(write_path+"/original", img_name), origin_crop_image)
            cv2.imwrite(os.path.join(write_path+"/forfilm", img_name), heat_img) 
            cv2.imwrite(os.path.join(write_path+"/vol_mask", img_name), SR*255)
        tEnd = time.time()
        print("Cost time(seconds)= "+str(tEnd-tStart))
        for dir_files in (LISTDIR(config.output_path)):
            full_path = os.path.join(config.output_path, dir_files)
            o_full_path = os.path.join(config.input_path, dir_files)
            for num_files in tqdm(LISTDIR(full_path)):
                full_path_2 = os.path.join(full_path, num_files+"/merge")
                height_path = os.path.join(o_full_path, num_files, "height.txt")
                s_height_path = os.path.join(full_path, num_files)
                os.system("cp "+height_path+" "+s_height_path)
                print("cp "+height_path+" "+s_height_path)
                frame2video(full_path_2)
                if config.keep_image == 0:
                    full_path_3 = os.path.join(full_path, num_files+"/original")
                    full_path_4 = os.path.join(full_path, num_files+"/forfilm")
                    os.system("rm -r "+full_path_3)
                    os.system("rm -r "+full_path_4)
    
def main(config):
    # parameter setting
    if config.which_model == 1 or config.which_model == 2 or config.which_model == 3:
        test_loader = get_loader(image_path = config.input_path,
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False)
    elif config.which_model == 7:
        test_loader = get_continuous_loader(image_path = config.input_path,
                            batch_size = 1,
                            mode = 'test',
                            augmentation_prob = 0.,
                            shffule_yn = False)
    test(config, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--which_model', type=int, default=3)
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--keep_image', type= int, default=1)
    config = parser.parse_args()
    main(config)