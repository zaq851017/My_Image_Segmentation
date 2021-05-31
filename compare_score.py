import numpy as np
from sklearn.metrics import f1_score
import argparse
import os
import cv2
import logging
from tqdm import tqdm
import ipdb
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
    return iou
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict1_path', type=str, default="")
    parser.add_argument('--predict2_path', type=str, default="")
    parser.add_argument('--GT_path', type=str, default="")
    config = parser.parse_args()
    predict1 = cv2.imread(config.predict1_path, cv2.IMREAD_GRAYSCALE)
    predict2 = cv2.imread(config.predict2_path, cv2.IMREAD_GRAYSCALE)
    GT = cv2.imread(config.GT_path, cv2.IMREAD_GRAYSCALE)
    _, predict1 = cv2.threshold(predict1, 127, 1, cv2.THRESH_BINARY)
    _, predict2 = cv2.threshold(predict2, 127, 1, cv2.THRESH_BINARY)
    _, GT = cv2.threshold(GT, 127, 1, cv2.THRESH_BINARY)
    log_name = "/".join(config.GT_path.split(".")[:-1])+"_score.txt"
    logging.basicConfig(level=logging.DEBUG,
                        handlers = [logging.FileHandler(log_name, 'w', 'utf-8'),logging.StreamHandler()])
    iou1 = cal_iou(GT, predict1)
    iou2 = cal_iou(GT, predict2)
    logging.info("predict1 iou "+str(iou1))
    logging.info("predict2 iou "+str(iou2))
