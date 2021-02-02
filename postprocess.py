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
from matplotlib import cm as CM
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
def postprocess_img(o_img, final_mask_exist):
    int8_o_img = np.array(o_img, dtype=np.uint8)
    if np.sum(int8_o_img != 0) == 0 or final_mask_exist == 0:
        return np.zeros((o_img.shape[0],o_img.shape[1]), dtype = np.uint8)
    else:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(int8_o_img, connectivity=8)
        if stats.shape[0] > 2:
            if stats[1][4] <= stats[2][4]:
                labels[labels == 1] = 0
            elif stats[1][4] > stats[2][4]:
                labels[labels == 2] = 0
        return np.array(labels, dtype=np.uint8)
def test(config, test_loader):
    threshold = config.threshold
    if config.which_model == 1:
        net = FCN32s(1)
        print("FCN32s load!")
    elif config.which_model == 2:
        net = HDC(1)
        print("HDC load")
    elif config.which_model == 3:
        net = FCN8s(1)
        print("FCN 8S load")
    net.load_state_dict(torch.load(config.model_path))
    net = net.cuda()
    net.eval()
    if not os.path.isdir(config.output_path):
        os.makedirs(config.output_path)
    with torch.no_grad():
        tStart = time.time()
        final_mask_exist = []
        mask_img = {}
        temp_mask_exist =[1] * len(test_loader)
        for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
            image = image.cuda()
            output = net(image)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
            SR = postprocess_img(SR, temp_mask_exist)
            dict_path = file_name[0].split("/")[-3]
            if dict_path not in mask_img:
                mask_img[dict_path] = []
                mask_img[dict_path].append(SR)
            else:
                mask_img[dict_path].append(SR)
        middle_list = {}
        for key in mask_img:
            middle_list[key] = []
            for img_index in range(len(mask_img[key])):
                img = mask_img[key][img_index]
                if np.sum(img) != 0:
                    mean_x = np.mean(img.nonzero()[0])
                    mean_y = np.mean(img.nonzero()[1])
                else:
                    mean_x = 0
                    mean_y = 0
                middle_list[key].append([mean_x, mean_y])
        mean_list = {}
        for key in middle_list:
            previous_total = 0
            previous_x = 0
            previous_y = 0
            next_total = 0
            next_x = 0
            next_y = 0
            for i, (x,y) in enumerate(middle_list[key]):
                if x != 0 and y != 0:
                    if i < len(middle_list[key]) / 2:
                        previous_x += x
                        previous_y += y
                        previous_total += 1
                    else:
                        next_x += x
                        next_y += y
                        next_total += 1
            if previous_total ==0 or next_total == 0:
                print(key)
                previous_total += 1
                next_total += 1
            mean_list[key] = [ [previous_x/previous_total, previous_y/previous_total], [next_x/next_total, next_y/next_total]]
        for key in middle_list:
            for i, (x, y) in enumerate(middle_list[key]):
                if x == 0 and y == 0:
                    final_mask_exist.append(1)
                elif i < len(middle_list[key]) / 2:
                    abs_x = abs(x - mean_list[key][0][0])
                    abs_y = abs(y - mean_list[key][0][1])
                    if abs_x >= 75 or abs_y >= 75:
                        final_mask_exist.append(0)
                    else:
                        final_mask_exist.append(1)
                else:
                    abs_x = abs(x - mean_list[key][1][0])
                    abs_y = abs(y - mean_list[key][1][1])
                    if abs_x >= 75 or abs_y >= 75:
                        final_mask_exist.append(0)
                    else:
                        final_mask_exist.append(1)
        for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
            image = image.cuda()
            output = net(image)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
            SR = postprocess_img(SR, final_mask_exist[i])
            heatmap = np.uint8(255 * SR)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heat_img = heatmap*0.9+origin_crop_image
            temp = [config.output_path] + file_name[0].split("/")[2:-2]
            write_path = "/".join(temp)
            img_name = file_name[0].split("/")[-1]
            if not os.path.isdir(write_path):
                os.makedirs(write_path+"/merge")
                os.makedirs(write_path+"/original")
                os.makedirs(write_path+"/forfilm")
            merge_img = np.hstack([origin_crop_image, heat_img])
            cv2.imwrite(os.path.join(write_path+"/merge", img_name), merge_img)
            imageio.imwrite(os.path.join(write_path+"/original", img_name), origin_crop_image)
            cv2.imwrite(os.path.join(write_path+"/forfilm", img_name), heat_img)
        tEnd = time.time()
        print("Cost time(seconds)= "+str(tEnd-tStart))
        for dir_files in (LISTDIR(config.output_path)):
            full_path = os.path.join(config.output_path, dir_files)
            for num_files in tqdm(LISTDIR(full_path)):
                full_path_2 = os.path.join(full_path, num_files+"/merge")
                frame2video(full_path_2)
                if config.keep_image == 0:
                    full_path_3 = os.path.join(full_path, num_files+"/original")
                    full_path_4 = os.path.join(full_path, num_files+"/forfilm")
                    os.system("rm -r "+full_path_3)
                    os.system("rm -r "+full_path_4)
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
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--keep_image', type= int, default=1)
    config = parser.parse_args()
    main(config)