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
import copy
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
        """
        import ipdb; ipdb.set_trace()
        if stats.shape[0] > 2:
            if stats[1][4] <= stats[2][4]:
                labels[labels == 1] = 0
            elif stats[1][4] > stats[2][4]:
                labels[labels == 2] = 0
        """
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
        temp_mask_exist = [1] * len(test_loader)
        temp_continue_list = [1] * len(test_loader)
        continue_list = []
        last_signal = 0
        range_dict = {}
        start = 0
        end = -1
        last_film_name = ""
        for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
            now_filename = file_name[0].split("/")[-3]
            if now_filename != last_film_name and last_film_name in range_dict:
                range_dict[last_film_name] = [start, end]
            if now_filename != last_film_name and now_filename not in range_dict:
                range_dict[now_filename] = [0, 0]
                start = end+1
            end += 1
            last_film_name = now_filename
            image = image.cuda()
            output = net(image)
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
                mask_img[dict_path] = []
                mask_img[dict_path].append(SR)
            else:
                mask_img[dict_path].append(SR)
        range_dict[last_film_name] = [start, end]
        postprocess_continue_list = copy.deepcopy(continue_list)
        start = 0
        end = 0
        check_start = False
        for i in range(len(continue_list)):
            if continue_list[i] == 1 and check_start == False and i < len(continue_list)-1:
                start = i
                check_start = True
                continue
            elif continue_list[i] == 1 and check_start == True and i < len(continue_list)-1:
                end = i
                continue
            elif continue_list[i] == 0 and check_start == True:
                temp = (end+1) - start
                if temp < 0:
                    postprocess_continue_list[start: start+1] = [0]
                if temp <= 30:
                    postprocess_continue_list[start: end+1] = [0] * temp
                check_start = False
                continue
            elif continue_list[i] == 1 and i == len(continue_list)-1:
                end = i
                temp = (end+1) - start
                if temp < 0:
                    postprocess_continue_list[end: end+1] = [0]
                if temp <= 30:
                    postprocess_continue_list[start: end+1] = [0] * temp
                check_start = False
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
        global_mean_list = {}
        for key in middle_list:
            temp_total = [0] * 5
            temp_x = [0] * 5
            temp_y = [0] * 5
            temp_global_x = 0
            temp_global_y = 0
            temp_global_total = 0
            for i, (x,y) in enumerate(middle_list[key]):
                if x != 0 and y != 0:
                    temp_global_x += x
                    temp_global_y += y
                    temp_global_total += 1
                    if i < len(middle_list[key]) / 5:
                        temp_x[0] += x
                        temp_y[0] += y
                        temp_total[0] += 1
                    elif i < 2*len(middle_list[key]) / 5:
                        temp_x[1] += x
                        temp_y[1] += y
                        temp_total[1] += 1
                    elif i < 3*len(middle_list[key]) / 5:
                        temp_x[2] += x
                        temp_y[2] += y
                        temp_total[2] += 1
                    elif i < 4*len(middle_list[key]) / 5:
                        temp_x[3] += x
                        temp_y[3] += y
                        temp_total[3] += 1
                    else:
                        temp_x[4] += x
                        temp_y[4] += y
                        temp_total[4] += 1
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
        for key in middle_list:
            for i, (x, y) in enumerate(middle_list[key]):
                if x == 0 and y == 0:
                    final_mask_exist.append(0)
                elif i < len(middle_list[key]) / 5:
                    abs_x = min(abs(x-global_mean_list[key][0]),abs(x - mean_list[key][0][0]))
                    abs_y = min(abs(y-global_mean_list[key][1]), abs(y - mean_list[key][0][1]))
                    if abs_x >= 75 or abs_y >= 75:
                        final_mask_exist.append(0)
                    else:
                        final_mask_exist.append(1)
                elif i < 2*len(middle_list[key]) / 5:
                    abs_x = min(abs(x-global_mean_list[key][0]), abs(x - mean_list[key][1][0]))
                    abs_y = min(abs(y-global_mean_list[key][1]), abs(y - mean_list[key][1][1]))
                    if abs_x >= 75 or abs_y >= 75:
                        final_mask_exist.append(0)
                    else:
                        final_mask_exist.append(1)
                elif i < 3*len(middle_list[key]) / 5:
                    abs_x = min(abs(x-global_mean_list[key][0]), abs(x - mean_list[key][2][0]))
                    abs_y = min(abs(y-global_mean_list[key][1]), abs(y - mean_list[key][2][1]))
                    if abs_x >= 75 or abs_y >= 75:
                        final_mask_exist.append(0)
                    else:
                        final_mask_exist.append(1)
                elif i < 4*len(middle_list[key]) / 5:
                    abs_x = min(abs(x-global_mean_list[key][0]), abs(x - mean_list[key][3][0]))
                    abs_y = min(abs(y-global_mean_list[key][1]), abs(y - mean_list[key][3][1]))
                    if abs_x >= 75 or abs_y >= 75:
                        final_mask_exist.append(0)
                    else:
                        final_mask_exist.append(1)
                else:
                    abs_x = min(abs(x-global_mean_list[key][0]), abs(x - mean_list[key][4][0]))
                    abs_y = min(abs(y-global_mean_list[key][1]), abs(y - mean_list[key][4][1]))
                    if abs_x >= 75 or abs_y >= 75:
                        final_mask_exist.append(0)
                    else:
                        final_mask_exist.append(1)
        range_list = [*range_dict.values()]
        for (x,y) in range_list:
            start = x
            end = y
            for i in range(start, end+1, 1):
                if postprocess_continue_list[i] == 0:
                    start +=1
                else:
                    break
            for i in range(end, start-1, -1):
                if postprocess_continue_list[i] == 0:
                    end -= 1
                else:
                    break
            temp = (end+1) - start
            postprocess_continue_list[start: end+1] = [1] *temp
        adjustment_list = [0] * len(final_mask_exist)
        for i in range(len(final_mask_exist)):
            if postprocess_continue_list[i] == 1 and final_mask_exist[i] == 0:
                adjustment_list[i] = 1
        check_start = False
        temp_record = 0
        save_list = []
        for i in range(len(adjustment_list)-1, -1, -1):
            if adjustment_list[i] == 0:
                temp_record = i
            elif adjustment_list[i] == 1:
                save_list.insert(0, temp_record)
                adjustment_list[i] = temp_record
        save_img_list = {}
        for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
            if i in save_list:
                image = image.cuda()
                output = net(image)
                SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
                SR = postprocess_img(SR, final_mask_exist[i], postprocess_continue_list[i])
                if np.sum(SR==1) <= 1000:
                    for j in range(len(save_list)):
                        if save_list[j] == i:
                            save_list[j] = i+1
                    for j in range(len(adjustment_list)):
                        if adjustment_list[j] == i:
                            adjustment_list[j] = i+1
                    last_num = [m for m,x in enumerate(adjustment_list) if x == i+1][-1]
                    adjustment_list[last_num: i+1] = [i+1]*(i+1-last_num)
                else:
                    save_img_list[i] = SR
        import ipdb; ipdb.set_trace()
        for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
            image = image.cuda()
            output = net(image)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            if adjustment_list[i] == 0 or  (adjustment_list[i] == 1 and final_mask_exist[i] == 0):
                SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
                SR = postprocess_img(SR, final_mask_exist[i], postprocess_continue_list[i])
            else:
                SR = save_img_list[adjustment_list[i]]
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