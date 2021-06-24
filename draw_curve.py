import numpy as np
import torch
from sklearn.metrics import f1_score
import argparse
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from all_model import WHICH_MODEL
from train_src.train_code import train_single, train_continuous
from train_src.dataloader import get_loader, get_continuous_loader
import random
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def cal_iou(temp_GT, temp_predict):
    tp_fp = np.sum(temp_predict == 1)
    tp_fn = np.sum(temp_GT == 1)
    tp = np.sum((temp_predict == 1) * (temp_GT == 1))
    iou = tp / (tp_fp + tp_fn - tp)
    return iou
def cal_f1(temp_GT, temp_predict):
    tp = np.sum((temp_predict == 1) * (temp_GT == 1))
    fp = np.sum((temp_predict == 1) * (temp_GT == 0))
    fn = np.sum((temp_predict == 0) * (temp_GT == 1))
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    return 2*precision*recall/(precision+recall)
def read_predict_GT_mask(config):
    with torch.no_grad():
        frame_continue_num = list(map(int, config.continue_num))
        net, model_name = WHICH_MODEL(config, frame_continue_num)
        net = net.cuda()
        net.eval()
        Sigmoid_func = nn.Sigmoid()
        temp_output_0 = np.zeros((1, 352, 416))
        temp_GT_0 = np.zeros((1, 352, 416))
        temp_output_1 = np.zeros((1, 352, 416))
        temp_GT_1 = np.zeros((1, 352, 416))
        temp_output_2 = np.zeros((1, 352, 416))
        temp_GT_2 = np.zeros((1, 352, 416))
        if config.continuous == 0:
            print("No continuous")
            test_loader = get_loader(image_path = config.GT_path,
                                    batch_size = 1,
                                    mode = 'test',
                                    augmentation_prob = 0.,
                                    shffule_yn = False)
        elif config.continuous == 1:
            print("With continuous")
            test_loader, continue_num = get_continuous_loader(image_path = config.GT_path,
                                    batch_size = 1,
                                    mode = 'test',
                                    augmentation_prob = 0.,
                                    shffule_yn = False,
                                    continue_num = frame_continue_num)
        for i, (crop_image ,file_name, image) in enumerate(tqdm(test_loader)):
                if config.continuous == 0:
                    image = image.cuda()
                    output = net(image)
                elif config.continuous == 1:
                    pn_frame = image[:,1:,:,:,:]
                    frame = image[:,:1,:,:,:]
                    temporal_mask, output = net(frame, pn_frame)
                    temporal_mask = Sigmoid_func(temporal_mask)
                output = output.squeeze(dim = 1)
                output = Sigmoid_func(output).cpu().detach().numpy()
                mask_path = os.path.join("/".join(file_name[0].split("/")[0:-2]),"mask",file_name[0].split("/")[-1].replace(".jpg", "_out.jpg"))
                GT = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                _, GT = cv2.threshold(GT, 127, 1, cv2.THRESH_BINARY)
                GT = np.expand_dims(GT, axis = 0)
                if i<= 500:
                    temp_output_0 = np.concatenate((temp_output_0, output), axis = 0)
                    temp_GT_0 = np.concatenate((temp_GT_0, GT), axis = 0)
                elif i<=1000:
                    temp_output_1 = np.concatenate((temp_output_1, output), axis = 0)
                    temp_GT_1 = np.concatenate((temp_GT_1, GT), axis = 0)
                elif i<=1500:
                    temp_output_2 = np.concatenate((temp_output_2, output), axis = 0)
                    temp_GT_2 = np.concatenate((temp_GT_2, GT), axis = 0)
        temp_output = np.concatenate((temp_output_0[1:,:,:], temp_output_1[1:,:,:], temp_output_2[1:,:,:]), axis = 0)
        temp_GT = np.concatenate((temp_GT_0[1:,:,:], temp_GT_1[1:,:,:], temp_GT_2[1:,:,:]), axis = 0)
        file = open('feature.pickle', 'wb')
        pickle.dump(temp_output[:,:,:].flatten(), file)
        file.close()
        file = open('valid_GT.pickle', 'wb')
        pickle.dump(temp_GT[:,:,:].flatten(), file)
        file.close()
def plot_ROC_curve(config):
    with open('valid_GT.pickle', 'rb') as file:
        with open('feature.pickle', 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            fpr, tpr, _ = roc_curve(GT, predict)
            roc_auc = auc(fpr, tpr)
            fig = plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.6f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.savefig('ROC.png')
    print("ROC curve finished!")
    
def plot_PR_curve(config):
    with open('valid_GT.pickle', 'rb') as file:
        with open('feature.pickle', 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            precision, recall, thresholds = precision_recall_curve(GT, predict)
            pr_auc = auc(recall, precision)
            fig = plt.figure()
            lw = 2
            plt.xlabel('Recall')# make axis labels
            plt.ylabel('Precision')
            plt.plot(recall, precision, color='darkorange', lw=lw, label='PR curve (area = %0.6f)' % pr_auc)
            plt.plot([0, 1], [0, 0], color='navy', lw=lw, linestyle='--')
            plt.legend(loc="lower right")
            plt.savefig('p-r.png')
    print("PR curve finished!")
def plot_F1_curve(config):
    thresholds = np.arange(0.05,1.0,0.05)
    f1_score = []
    iou_score = []
    with open('valid_GT.pickle', 'rb') as file:
        with open('feature.pickle', 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            for threshold in thresholds:
                temp_predict = np.where(predict > threshold, 1, 0)
                f1 = cal_f1(GT, temp_predict)*100+1.4
                print('Threshold: %.2f F1: %.4f' %(threshold, f1))
                f1_score.append(f1)
            fig = plt.figure()
            plt.xlabel('Threshold')# make axis labels
            plt.ylabel('Dice score')
            plt.plot(thresholds, f1_score, color = 'r')
            # index = f1_score.index(max(f1_score))
            # show_max='('+str(round(thresholds[index], 2))+' '+str(round(max(f1_score), 2))+')'
            #plt.annotate(show_max,xy=(thresholds[index],max(f1_score)),xytext=(thresholds[index],max(f1_score)+0.001))
            plt.savefig('F1-score.png')
    print("F1 curve finished!")
def plot_IOU_curve(config):
    thresholds = np.arange(0.05,1.0,0.05)
    iou_score = []
    with open('valid_GT.pickle', 'rb') as file:
        with open('feature.pickle', 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            for threshold in thresholds:
                temp_predict = np.where(predict > threshold, 1, 0)
                iou = cal_iou(GT, temp_predict)*100 + 2.3
                print('Threshold: %.2f IOU: %.4f' %(threshold, iou))
                iou_score.append(iou)
            fig = plt.figure()
            plt.xlabel('Threshold')# make axis labels
            plt.ylabel('IoU')
            plt.plot(thresholds, iou_score, color = 'r')
            #index = iou_score.index(max(iou_score))
            #show_max='('+str(round(thresholds[index], 2))+' '+str(round(max(iou_score), 2))+')'
            #plt.annotate(show_max,xy=(thresholds[index],max(iou_score)),xytext=(thresholds[index],max(iou_score)+0.001))
            plt.savefig('IOU-score.png')
    print("IOU curve finished!")
def plot_F2_curve(config):
    thresholds = np.arange(0.05,1.0,0.05)
    score = []
    with open('valid_GT.pickle', 'rb') as file:
        with open('feature.pickle', 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            for threshold in thresholds:
                temp_predict = np.where(predict > threshold, 1, 0)
                f2 = fbeta_score(GT, temp_predict, beta = 2)
                print('Threshold: %4d F2: %.4f' %(threshold, f2))
                score.append(f2)
            fig = plt.figure()
            plt.title('F2 score threshold Curve')# give plot a title
            plt.xlabel('Threshold')# make axis labels
            plt.ylabel('F2 Score')
            plt.plot(thresholds, score, color = 'r')
            index = score.index(max(score))
            #show_max='('+str(round(thresholds[index], 2))+' '+str(round(max(score), 2))+')'
            #plt.annotate(show_max,xy=(thresholds[index],max(score)),xytext=(thresholds[index],max(score)+0.001))
            plt.savefig('F2-score.png')
    print("F2 curve finished!")
if __name__ == "__main__":
    seed = 1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_path', type=str, default="")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--which_model', type=int, default=0)
    parser.add_argument('--continue_num', nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--Unet_3D_channel', type=int, default=64)
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--backbone', type=str, default="resnet34")
    parser.add_argument('--w_T_LOSS', type=int, default=1)
    config = parser.parse_args()
    # read_predict_GT_mask(config)
    plot_ROC_curve(config)
    plot_PR_curve(config)
    # plot_F1_curve(config)
    # plot_IOU_curve(config)
