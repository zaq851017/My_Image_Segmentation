import numpy as np
from sklearn.metrics import f1_score
import argparse
import os
import cv2
from tqdm import tqdm
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def read_predict_GT_mask(predict_path, GT_path):
    temp_GT = np.zeros((1, 368, 424))
    temp_predict = np.zeros((1, 368, 424))
    for num_files in LISTDIR(predict_path):
        p_full_path = os.path.join(predict_path, num_files)
        G_full_path = os.path.join(GT_path, num_files)
        for dir_files in LISTDIR(p_full_path):
            p_mask_path = os.path.join(p_full_path, dir_files, "vol_mask")
            G_mask_path = os.path.join(G_full_path, dir_files, "mask")
            for p_mask_files in tqdm(LISTDIR(p_mask_path)):
                img_predict_path = os.path.join(p_mask_path, p_mask_files)
                img_GT_path = os.path.join(G_mask_path, p_mask_files.split(".")[0]+"_out.jpg")
                predict = cv2.imread(img_predict_path, cv2.IMREAD_GRAYSCALE)
                GT = cv2.imread(img_GT_path, cv2.IMREAD_GRAYSCALE)
                GT = GT[70:438,150:574]
                _, GT = cv2.threshold(GT, 127, 1, cv2.THRESH_BINARY)
                _, predict = cv2.threshold(predict, 127, 1, cv2.THRESH_BINARY)
                GT = np.expand_dims(GT, axis = 0)
                predict = np.expand_dims(predict, axis = 0)
                temp_GT = np.concatenate((temp_GT, GT), axis = 0)
                temp_predict = np.concatenate((temp_predict, predict), axis = 0)
    for i in range(2):
        tp_fp = np.sum(temp_predict[1:,:,:] == i)
        tp_fn = np.sum(temp_GT[1:,:,:] == i)
        tp = np.sum((temp_predict[1:,:,:] == i) * (temp_GT[1:,:,:] == i))
        iou = tp / (tp_fp + tp_fn - tp)
        if i == 0:
            background_iou = iou
        else:
            MA_iou = iou
    F1_score = f1_score(temp_GT[1:,:,:].flatten(), temp_predict[1:,:,:].flatten())
    print("backgroind iou", background_iou)
    print("MA iou",MA_iou)
    print("F1 score",F1_score)
    #score = f1_score(GT.flatten(), predict.flatten(), average='binary')

def cal_score(config):
    read_predict_GT_mask(config.predict_path, config.GT_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str, default="")
    parser.add_argument('--GT_path', type=str, default="")
    config = parser.parse_args()
    cal_score(config)