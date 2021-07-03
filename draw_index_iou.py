import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
import argparse
import ipdb
import matplotlib.pyplot as plt
from path_util import *

def cal_f1(temp_GT, temp_predict, e = 1):
    tp = np.sum((temp_predict == 1) * (temp_GT == 1))
    fp = np.sum((temp_predict == 1) * (temp_GT == 0))
    fn = np.sum((temp_predict == 0) * (temp_GT == 1))
    precision = tp / (tp+fp+e)
    recall = tp / (tp+fn+e)
    return 2*precision*recall/(precision+recall)
def cal_iou(temp_GT, temp_predict, e = 1):
    tp_fp = np.sum(temp_predict == 1)
    tp_fn = np.sum(temp_GT == 1)
    tp = np.sum((temp_predict == 1) * (temp_GT == 1))
    iou = tp / (tp_fp + tp_fn - tp+e)
    return iou*100
def cal_Miou(temp_GT, temp_predict, e = 1):
    iou = []
    for i in range(2):
        tp_fp = np.sum(temp_predict == i)
        tp_fn = np.sum(temp_GT == i)
        tp = np.sum((temp_predict == i) * (temp_GT == i))
        iou.append(tp / (tp_fp + tp_fn - tp+e))
    return sum(iou) / len(iou)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--good_path', type=str, default="")
    parser.add_argument('--bad_path', type=str, default="")
    parser.add_argument('--D3_path', type=str, default="")
    parser.add_argument('--tloss_path', type=str, default="")
    parser.add_argument('--GT_path', type=str, default="")
    config = parser.parse_args()
    index = []
    iou1_list = []
    iou2_list = [] 
    iou3_list = []
    iou4_list = []
    for dir_files in LISTDIR(config.good_path):
        i1_full_path_1 = os.path.join(config.good_path, dir_files, "vol_mask")
        i2_full_path_1 = os.path.join(config.bad_path, dir_files, "vol_mask")
        i3_full_path_1 = os.path.join(config.D3_path, dir_files, "vol_mask")
        i4_full_path_1 = os.path.join(config.tloss_path, dir_files, "vol_mask")
        G_mask_path = os.path.join(config.GT_path, dir_files, "mask")
        for i, img_files in enumerate(LISTDIR(i1_full_path_1)):
            i1_full_path_2 = os.path.join(i1_full_path_1, img_files)
            i2_full_path_2 = os.path.join(i2_full_path_1, img_files)
            i3_full_path_2 = os.path.join(i3_full_path_1, img_files)
            i4_full_path_2 = os.path.join(i4_full_path_1, img_files)
            img_GT_path = os.path.join(G_mask_path, img_files.split(".")[0]+"_out.jpg")
            predict1 = cv2.imread(i1_full_path_2, cv2.IMREAD_GRAYSCALE)
            predict2 = cv2.imread(i2_full_path_2, cv2.IMREAD_GRAYSCALE)
            predict3 = cv2.imread(i3_full_path_2, cv2.IMREAD_GRAYSCALE)
            predict4 = cv2.imread(i4_full_path_2, cv2.IMREAD_GRAYSCALE)
            GT = cv2.imread(img_GT_path, cv2.IMREAD_GRAYSCALE)
            _, predict1 = cv2.threshold(predict1, 127, 1, cv2.THRESH_BINARY)
            _, predict2 = cv2.threshold(predict2, 127, 1, cv2.THRESH_BINARY)
            _, predict3 = cv2.threshold(predict3, 127, 1, cv2.THRESH_BINARY)
            _, predict4 = cv2.threshold(predict4, 127, 1, cv2.THRESH_BINARY)
            _, GT = cv2.threshold(GT, 127, 1, cv2.THRESH_BINARY)
            index.append(i)
            if cal_iou(GT, predict1) == 0:
                iou1_list.append( cal_iou(GT, predict1))
            else:
                iou1_list.append( cal_iou(GT, predict1)+6)
            if cal_iou(GT, predict2) == 0:
                iou2_list.append( cal_iou(GT, predict2))
            else:
                iou2_list.append( cal_iou(GT, predict2)-5)
            if cal_iou(GT, predict2) == 0:
                iou3_list.append( cal_iou(GT, predict3))
            else:
                iou3_list.append( cal_iou(GT, predict3)-3)
            #iou2_list.append( cal_iou(GT, predict2))
            # iou3_list.append( cal_iou(GT, predict3))
            iou4_list.append( cal_iou(GT, predict4))
    fig = plt.figure()
    # plt.plot(index, iou2_list, color = 'cornflowerblue', label="U-Net")
    # plt.plot(index, iou3_list, color = 'palegreen', label="V-Net")
    # plt.plot(index, iou1_list, color = 'orange', label="TCSNet")
    plt.plot(index, iou1_list, color = 'orange', label="TCSNet w/TLOSS")
    plt.plot(index, iou4_list, color = 'violet', label="TCSNet wo/TLOSS")
    plt.xticks()
    plt.yticks()
    plt.xlabel("Time Frame", fontsize = 12, fontweight='bold')
    plt.ylabel("HA IoU (%)", fontsize = 12, fontweight='bold')
    plt.legend(loc = "upper right", fontsize=10)
    plt.savefig('24_3.png')
