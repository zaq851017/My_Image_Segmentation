import numpy as np
from sklearn.metrics import f1_score
import argparse
import os
import cv2
import logging
from tqdm import tqdm
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def read_predict_GT_mask(predict_path, GT_path):
    for num_files in LISTDIR(predict_path):
        p_full_path = os.path.join(predict_path, num_files)
        G_full_path = os.path.join(GT_path, num_files)
        if os.path.isdir(p_full_path):
            for dir_files in LISTDIR(p_full_path):
                p_mask_path = os.path.join(p_full_path, dir_files, "vol_mask")
                G_mask_path = os.path.join(G_full_path, dir_files, "mask")
                write_path = os.path.join(p_full_path, dir_files, "residual_map")
                if not os.path.isdir(write_path):
                    os.makedirs(write_path)
                for p_mask_files in tqdm(LISTDIR(p_mask_path)):
                    img_predict_path = os.path.join(p_mask_path, p_mask_files)
                    img_GT_path = os.path.join(G_mask_path, p_mask_files.split(".")[0]+"_out.jpg")
                    predict = cv2.imread(img_predict_path, cv2.IMREAD_GRAYSCALE)
                    GT = cv2.imread(img_GT_path, cv2.IMREAD_GRAYSCALE)
                    GT = GT[70:438,150:574]
                    _, GT = cv2.threshold(GT, 127, 1, cv2.THRESH_BINARY)
                    _, predict = cv2.threshold(predict, 127, 1, cv2.THRESH_BINARY)
                    residual_map = abs(GT-predict)
                    cv2.imwrite(os.path.join(write_path, p_mask_files), residual_map)
def cal_score(config):
    read_predict_GT_mask(config.predict_path, config.GT_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str, default="")
    parser.add_argument('--GT_path', type=str, default="")
    config = parser.parse_args()
    cal_score(config)