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
def read_predict_GT_mask(predict_path, GT_path):
    log_name = os.path.join(predict_path, "score.txt")
    logging.basicConfig(level=logging.DEBUG,
                        handlers = [logging.FileHandler(log_name, 'w', 'utf-8'),logging.StreamHandler()])
    logging.info("Predict_path:"+predict_path)
    total_GT = np.zeros((1, 368, 424))
    total_predict = np.zeros((1, 368, 424))
    for num_files in LISTDIR(predict_path):
        p_full_path = os.path.join(predict_path, num_files)
        G_full_path = os.path.join(GT_path, num_files)
        temp_GT = np.zeros((1, 368, 424))
        temp_predict = np.zeros((1, 368, 424))
        if os.path.isdir(p_full_path):
            for dir_files in LISTDIR(p_full_path):
                p_mask_path = os.path.join(p_full_path, dir_files, "vol_mask")
                G_mask_path = os.path.join(G_full_path, dir_files, "mask")
                for p_mask_files in tqdm(LISTDIR(p_mask_path)):
                    img_predict_path = os.path.join(p_mask_path, p_mask_files)
                    img_GT_path = os.path.join(G_mask_path, p_mask_files.split(".")[0]+"_out.jpg")
                    predict = cv2.imread(img_predict_path, cv2.IMREAD_GRAYSCALE)
                    GT = cv2.imread(img_GT_path, cv2.IMREAD_GRAYSCALE)
                    _, GT = cv2.threshold(GT, 127, 1, cv2.THRESH_BINARY)
                    _, predict = cv2.threshold(predict, 127, 1, cv2.THRESH_BINARY)
                    GT = np.expand_dims(GT, axis = 0)
                    predict = np.expand_dims(predict, axis = 0)
                    temp_GT = np.concatenate((temp_GT, GT), axis = 0)
                    temp_predict = np.concatenate((temp_predict, predict), axis = 0)
                    total_GT = np.concatenate((total_GT, GT), axis = 0)
                    total_predict = np.concatenate((total_predict, predict), axis = 0)
            MA_iou = cal_iou(temp_GT[1:,:,:].flatten(), temp_predict[1:,:,:].flatten())
            F1_score = cal_f1(temp_GT[1:,:,:].flatten(), temp_predict[1:,:,:].flatten())
            logging.info(p_full_path+"_MA iou: "+str(MA_iou))
            logging.info(p_full_path+"_F1 score: "+str(F1_score))
    MA_iou = cal_iou(total_GT[1:,:,:].flatten(), total_predict[1:,:,:].flatten())
    F1_score = cal_f1(total_GT[1:,:,:].flatten(), total_predict[1:,:,:].flatten())
    logging.info("Total_MA iou: "+str(MA_iou))
    logging.info("Total_F1 score: "+str(F1_score))
    return F1_score, MA_iou

def cal_score(config):
    F1_score, MA_iou = read_predict_GT_mask(config.predict_path, config.GT_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str, default="")
    parser.add_argument('--GT_path', type=str, default="")
    config = parser.parse_args()
    cal_score(config)