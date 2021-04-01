import numpy as np
from sklearn.metrics import f1_score
import argparse
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from dataloader import get_loader, get_continuous_loader
from FCN32s import *
from HDC import *
from FCN8s import *
from T_FCN8s import *
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def read_predict_GT_mask(config):
    if config.which_model == 1:
        net = FCN32s(1)
        print("FCN32s load!")
    elif config.which_model == 2:
        net = HDC(1)
        print("HDC load")
    elif config.which_model == 3:
        net = FCN8s(1)
        print("FCN 8S load")
    elif config.which_model == 7:
        net = T_FCN8s(1)
        print("Model temporal Temporal_FCN8S")
    net.load_state_dict(torch.load(config.model_path))
    net = net.cuda()
    net.eval()
    Sigmoid_func = nn.Sigmoid()
    temp_output = np.zeros((1, 368, 424))
    temp_GT = np.zeros((1, 368, 424))
    test_loader = get_continuous_loader(image_path = config.GT_path,
                            batch_size = 1,
                            mode = 'test',
                            augmentation_prob = 0.,
                            shffule_yn = False)
    for i, (crop_image ,file_name, image_list) in tqdm(enumerate(test_loader)):
            pn_frame = image_list[:,1:,:,:,:]
            frame = image_list[:,:1,:,:,:]
            output = net(frame, pn_frame).squeeze(dim = 1)
            output = Sigmoid_func(output).cpu().detach().numpy()
            mask_path = os.path.join("/".join(file_name[0].split("/")[0:-2]),"mask",file_name[0].split("/")[-1].replace(".jpg", "_out.jpg"))
            GT = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            GT = GT[70:438,150:574]
            _, GT = cv2.threshold(GT, 127, 1, cv2.THRESH_BINARY)
            GT = np.expand_dims(GT, axis = 0)
            temp_output = np.concatenate((temp_output, output), axis = 0)
            temp_GT = np.concatenate((temp_GT, GT), axis = 0)
    file = open(config.model_path.split("/")[-1]+'_predict.pickle', 'wb')
    pickle.dump(temp_output[1:,:,:].flatten(), file)
    file.close()
    file = open('valid_GT.pickle', 'wb')
    pickle.dump(temp_GT[1:,:,:].flatten(), file)
    file.close()
def plot_ROC_curve(config):
    with open('valid_GT.pickle', 'rb') as file:
        with open('Epoch=6_Score=1.304_predict.pickle', 'rb') as file2:
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
            plt.title('ROC curve')
            plt.legend(loc="lower right")
            plt.savefig('ROC.png')
    
def plot_PR_curve(config):
    with open('valid_GT.pickle', 'rb') as file:
        with open('Epoch=6_Score=1.304_predict.pickle', 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            precision, recall, thresholds = precision_recall_curve(GT, predict)
            pr_auc = auc(recall, precision)
            fig = plt.figure()
            lw = 2
            plt.title('PR Curve')# give plot a title
            plt.xlabel('Recall')# make axis labels
            plt.ylabel('Precision')
            plt.plot(recall, precision, color='darkorange', lw=lw, label='PR curve (area = %0.6f)' % pr_auc)
            plt.plot([0, 1], [0, 0], color='navy', lw=lw, linestyle='--')
            plt.legend(loc="lower right")
            plt.savefig('p-r.png')

def plot_F1_curve(config):
    thresholds = np.arange(0.05,1.0,0.05)
    score = []
    with open('valid_GT.pickle', 'rb') as file:
        with open('Epoch=6_Score=1.304_predict.pickle', 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            for threshold in thresholds:
                temp_predict = np.where(predict > threshold, 1, 0)
                f1 = fbeta_score(GT, temp_predict, beta = 1)
                print("f1",f1)
                score.append(f1)
            fig = plt.figure()
            plt.title('F1 score threshold Curve')# give plot a title
            plt.xlabel('Threshold')# make axis labels
            plt.ylabel('F1 Score')
            plt.plot(thresholds, score, color = 'r')
            index = score.index(max(score))
            show_max='('+str(round(thresholds[index], 2))+' '+str(round(max(score), 2))+')'
            plt.annotate(show_max,xy=(thresholds[index],max(score)),xytext=(thresholds[index],max(score)+0.001))
            plt.savefig('F1-score.png')
def plot_F2_curve(config):
    thresholds = np.arange(0.05,1.0,0.05)
    score = []
    with open('valid_GT.pickle', 'rb') as file:
        with open('Epoch=6_Score=1.304_predict.pickle', 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            for threshold in thresholds:
                temp_predict = np.where(predict > threshold, 1, 0)
                f2 = fbeta_score(GT, temp_predict, beta = 2)
                print("f2",f2)
                score.append(f2)
            fig = plt.figure()
            plt.title('F2 score threshold Curve')# give plot a title
            plt.xlabel('Threshold')# make axis labels
            plt.ylabel('F2 Score')
            plt.plot(thresholds, score, color = 'r')
            index = score.index(max(score))
            show_max='('+str(round(thresholds[index], 2))+' '+str(round(max(score), 2))+')'
            plt.annotate(show_max,xy=(thresholds[index],max(score)),xytext=(thresholds[index],max(score)+0.001))
            plt.savefig('F2-score.png')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_path', type=str, default="")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--which_model', type=int, default=3)
    config = parser.parse_args()
    #read_predict_GT_mask(config)
    #plot_ROC_curve(config)
    #plot_PR_curve(config)
    plot_F1_curve(config)
    plot_F2_curve(config)