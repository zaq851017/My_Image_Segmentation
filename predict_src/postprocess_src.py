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
from network.Vgg_FCN8s import Single_vgg_FCN8s, Temporal_vgg_FCN8s
from network.Vgg_Unet import Single_vgg_Unet, Temporal_vgg_Unet
from network.Res_Unet import Single_Res_Unet, Temporal_Res_Unet, Two_level_Res_Unet
from network.Nested_Unet import Single_Nested_Unet, Temporal_Nested_Unet, Two_Level_Nested_Unet
from network.Double_Unet import Single_Double_Unet, Temporal_Double_Unet
from train_src.train_code import train_single, train_continuous
from train_src.dataloader import get_loader, get_continuous_loader
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def frame2video(path):
    video_path = (path[:-6])
    videoWriter = cv2.VideoWriter(os.path.join(video_path,"video.mp4"), cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (848, 368))
    for frame_files in  LISTDIR(path):
        if frame_files[-3:] == "jpg":
            full_path = os.path.join(path, frame_files)
            frame = cv2.imread(full_path)
            videoWriter.write(frame)
    videoWriter.release()
def postprocess_img(o_img, final_mask_exist, continue_list):
    int8_o_img = np.array(o_img, dtype=np.uint8)
    if np.sum(int8_o_img != 0) == 0 or final_mask_exist == 0 or continue_list == 0:
        return np.zeros((o_img.shape[0],o_img.shape[1]), dtype = np.uint8)
    else:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(int8_o_img, connectivity=8)
        index_sort = np.argsort(-stats[:,4])
        if index_sort.shape[0] > 2:
            for ll in index_sort[2:]:
                labels[ labels == ll ] = 0
        return np.array(labels, dtype=np.uint8)
def Check_continue(continue_list, postprocess_continue_list, bound_list, distance = 30):
    start = 0
    end = 0
    check_start = False
    for i in range(len(continue_list)):
            if continue_list[i] == 1 and check_start == False and i < len(continue_list)-1:
                start = i
                check_start = True
                continue
            elif continue_list[i] == 1 and check_start == True and i < len(continue_list)-1 and i not in bound_list:
                end = i
                continue
            elif continue_list[i] == 0 and check_start == True:
                temp = (end+1) - start
                if temp < 0:
                    postprocess_continue_list[start: start+1] = [0]
                if temp <= distance:
                    postprocess_continue_list[start: end+1] = [0] * temp
                check_start = False
                continue
            elif continue_list[i] == 1 and i in  bound_list:
                end = i
                temp = (end+1) - start
                if temp < 0:
                    postprocess_continue_list[end: end+1] = [0]
                if temp <= 30:
                    postprocess_continue_list[start: end+1] = [0] * temp
                check_start = False
    return postprocess_continue_list
def Cal_mask_center(mask_img):
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
    return middle_list
def Cal_Local_Global_mean(middle_list, interval_num = 5):
    mean_list = {}
    global_mean_list = {}
    for key in middle_list:
        temp_total = [0] * interval_num
        temp_x = [0] * interval_num
        temp_y = [0] * interval_num
        temp_global_x = 0
        temp_global_y = 0
        temp_global_total = 0
        len_check_list = []
        for i in range(1,interval_num+1):
            len_check_list.append([(i-1), (i-1)*len(middle_list[key])/interval_num, i*len(middle_list[key])/interval_num ])
        for i, (x,y) in enumerate(middle_list[key]):
            if x!=0 and y != 0:
                temp_global_x += x
                temp_global_y += y
                temp_global_total += 1
                for j in range(len(len_check_list)):
                    if i >= len_check_list[j][1] and i< len_check_list[j][2]:
                        temp_x[len_check_list[j][0]] += x
                        temp_y[len_check_list[j][0]] += y
                        temp_total[len_check_list[j][0]] += 1
        if temp_global_total == 0:
            temp_global_total += 1
        for check_temp in range(len(temp_total)):
            if temp_total[check_temp] == 0:
                temp_total[check_temp] +=1
        temp_list = []
        for temp in range(len(temp_total)):
            temp_list.append([ temp_x[temp]/temp_total[temp], temp_y[temp]/temp_total[temp]])
        global_mean_list[key] = [temp_global_x/ temp_global_total, temp_global_y/ temp_global_total]
        mean_list[key] = temp_list
    return mean_list, global_mean_list
def Final_postprocess(middle_list, mean_list, global_mean_list, interval_num = 5, distance = 75):
    final_mask_exist = []
    for key in middle_list:
        len_check_list = []
        for i in range(1,interval_num+1):
            len_check_list.append([(i-1), (i-1)*len(middle_list[key])/interval_num, i*len(middle_list[key])/interval_num ])
        for i, (x, y) in enumerate(middle_list[key]):
            if x == 0 and y == 0:
                final_mask_exist.append(0)
            else:
                for j in range(len(len_check_list)):
                    if i >= len_check_list[j][1] and i< len_check_list[j][2]:
                        abs_x = min(abs(x-global_mean_list[key][0]),abs(x - mean_list[key][len_check_list[j][0]][0]))
                        abs_y = min(abs(y-global_mean_list[key][1]), abs(y - mean_list[key][len_check_list[j][0]][1]))
                        if abs_x >= distance or abs_y >= distance:
                            final_mask_exist.append(0)
                        else:
                            final_mask_exist.append(1)
    return final_mask_exist
def test_wo_postprocess(config, test_loader):
    Sigmoid_func = nn.Sigmoid()
    threshold = config.threshold
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
    elif config.which_model == 9:
        net = Temporal_Nested_Unet(1)
        model_name = "Temporal_Nested_Unet"
        print("Model Temporal_Nested_Unet")
    elif config.which_model == 10:
        net = Temporal_Double_Unet(1)
        model_name = "Temporal_Double_Unet"
        print("Model Temporal_Double_Unet")
    elif config.which_model == 11:
        net = Two_level_Res_Unet(1)
        model_name = "Two_level_Res_Unet"
        print("Model Two_level_Res_Unet")
    elif config.which_model == 12:
        net = Two_Level_Nested_Unet(1)
        model_name = "Two_Level_Nested_Unet"
        print("Model Two_Level_Nested_Unet")
    elif config.which_model == 0:
        print("No assign which model!")
    net.load_state_dict(torch.load(config.model_path))
    net = net.cuda()
    net.eval()
    if not os.path.isdir(config.output_path):
        os.makedirs(config.output_path)
    with torch.no_grad():
        tStart = time.time()
        for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
            if config.continuous == 0:
                image = image.cuda()
                output = net(image)
            elif config.continuous == 1:
                pn_frame = image[:,1:,:,:,:]
                frame = image[:,:1,:,:,:]
                temporal_mask, output = net(frame, pn_frame)
                output = output.squeeze(dim = 1)
            output = Sigmoid_func(output)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
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
def test_w_postprocess(config, test_loader):
    Sigmoid_func = nn.Sigmoid()
    threshold = config.threshold
    distance = 75
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
    elif config.which_model == 9:
        net = Temporal_Nested_Unet(1)
        model_name = "Temporal_Nested_Unet"
        print("Model Temporal_Nested_Unet")
    elif config.which_model == 10:
        net = Temporal_Double_Unet(1)
        model_name = "Temporal_Double_Unet"
        print("Model Temporal_Double_Unet") 
    elif config.which_model == 11:
        net = Two_level_Res_Unet(1)
        model_name = "Two_level_Res_Unet"
        print("Model Two_level_Res_Unet")
    elif config.which_model == 12:
        net = Two_Level_Nested_Unet(1)
        model_name = "Two_Level_Nested_Unet"
        print("Model Two_Level_Nested_Unet")
    elif config.which_model == 0:
        print("No assign which model!")
    net.load_state_dict(torch.load(config.model_path))
    net = net.cuda()
    net.eval()
    if not os.path.isdir(config.output_path):
        os.makedirs(config.output_path)
    with torch.no_grad():
        tStart = time.time()
        mask_img = {}
        temp_mask_exist = [1] * len(test_loader)
        temp_continue_list = [1] * len(test_loader)
        continue_list = []
        last_signal = 0
        start = 0
        end = -1
        last_film_name = ""
        bound_list = []
        for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
            if config.continuous == 0:
                image = image.cuda()
                output = net(image)
            elif config.continuous == 1:
                pn_frame = image[:,1:,:,:,:]
                frame = image[:,:1,:,:,:]
                temporal_mask, output = net(frame, pn_frame)
                output = output.squeeze(dim = 1)
            output = Sigmoid_func(output)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
            SR = postprocess_img(SR, temp_mask_exist[i], temp_continue_list[1])
            if np.sum(SR != 0) == 0:
                continue_list.append(0)
            else:
                continue_list.append(1)
            dict_path = file_name[0].split("/")[-3]
            if dict_path not in mask_img:
                if i != 0:
                    bound_list.append(i-1)
                mask_img[dict_path] = []
                mask_img[dict_path].append(SR)
            else:
                mask_img[dict_path].append(SR)
        bound_list.append(i)
        postprocess_continue_list = copy.deepcopy(continue_list)
        postprocess_continue_list = Check_continue(continue_list, postprocess_continue_list, bound_list, distance = 30)
        middle_list = Cal_mask_center(mask_img)
        mean_list, global_mean_list = Cal_Local_Global_mean(middle_list, interval_num = 5)
        final_mask_exist = Final_postprocess(middle_list, mean_list, global_mean_list)
        for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
            if config.continuous == 0:
                image = image.cuda()
                output = net(image)
            elif config.continuous == 1:
                pn_frame = image[:,1:,:,:,:]
                frame = image[:,:1,:,:,:]
                temporal_mask, output = net(frame, pn_frame)
                output = output.squeeze(dim = 1)
            output = Sigmoid_func(output)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
            SR = postprocess_img(SR, final_mask_exist[i], postprocess_continue_list[i])
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