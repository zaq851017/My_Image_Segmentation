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
from network.Vgg_FCN8s import Single_vgg_FCN8s
from network.Vgg_Unet import Single_vgg_Unet
from network.Res_Unet import Single_Res_Unet
from network.Nested_Unet import Single_Nested_Unet
from network.DeepLab import DeepLab
from network.Unet3D import UNet_3D_Seg
from network.Two_Level_Net import Two_Level_Nested_Unet, Two_Level_Res_Unet, Two_Level_Deeplab, Two_Level_Res_Unet_with_backbone
from train_src.train_code import train_single, train_continuous
from train_src.dataloader import get_loader, get_continuous_loader
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
    frame_continue_num = list(map(int, config.continue_num))
    if config.which_model == 1:
        net = Single_vgg_FCN8s(1)
        model_name = "Single_vgg__FCN8s"
        print("Model Single_vgg__FCN8s")
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
        net = DeepLab()
        model_name = "Single_DeepLab"
        print("Model Single_DeepLab")
    elif config.which_model == 11:
        net = Two_Level_Res_Unet(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Res_Unet"
        print("Model Two_Level_Res_Unet")
    elif config.which_model == 12:
        net = Two_Level_Nested_Unet(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Nested_Unet"
        print("Model Two_Level_Nested_Unet")
    elif config.which_model == 13:
        net = UNet_3D_Seg(1)
        model_name = "UNet_3D_Seg"
        print("Model UNet_3D_Seg")
    elif config.which_model == 14:
        net = Two_Level_Deeplab(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Deeplab"
        print("Two_Level_Deeplab")
    elif config.which_model == 15:
        net = Two_Level_Res_Unet_with_backbone(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Res_Unet_with_backbone"
        print("Two_Level_Res_Unet_with_backbone")
    elif config.which_model == 0:
        print("No assign which model!")
    net.load_state_dict(torch.load(config.model_path))
    net = net.cuda()
    net.eval()
    Sigmoid_func = nn.Sigmoid()
    temp_output = np.zeros((1, 368, 424))
    temp_GT = np.zeros((1, 368, 424))
    if config.continuous == 0:
        test_loader = get_loader(image_path = config.GT_path,
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False)
    elif config.continuous == 1:
        test_loader = get_continuous_loader(image_path = config.GT_path,
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False,
                                continue_num = frame_continue_num)
    for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
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
            temp_output = np.concatenate((temp_output, output), axis = 0)
            temp_GT = np.concatenate((temp_GT, GT), axis = 0)
    file = open(model_name +"_"+config.model_path.split("/")[-1]+'_predict.pickle')
    pickle.dump(temp_output[1:,:,:].flatten(), file)
    file.close()
    file = open('valid_GT.pickle', 'wb')
    pickle.dump(temp_GT[1:,:,:].flatten(), file)
    file.close()
def plot_ROC_curve(config):
    with open('valid_GT.pickle', 'rb') as file:
        with open(config.feature_model_path, 'rb') as file2:
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
    print("ROC curve finished!")
    
def plot_PR_curve(config):
    with open('valid_GT.pickle', 'rb') as file:
        with open(config.feature_model_path, 'rb') as file2:
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
    print("PR curve finished!")
def plot_F1_curve(config):
    thresholds = np.arange(0.05,1.0,0.05)
    f1_score = []
    iou_score = []
    with open('valid_GT.pickle', 'rb') as file:
        with open(config.feature_model_path, 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            for threshold in thresholds:
                temp_predict = np.where(predict > threshold, 1, 0)
                f1 = cal_f1(GT, temp_predict)
                print('Threshold: %.2f F1: %.4f' %(threshold, f1))
                f1_score.append(f1)
            fig = plt.figure()
            plt.plot(thresholds, f1_score, color = 'r')
            index = f1_score.index(max(f1_score))
            show_max='('+str(round(thresholds[index], 2))+' '+str(round(max(f1_score), 2))+')'
            plt.annotate(show_max,xy=(thresholds[index],max(f1_score)),xytext=(thresholds[index],max(f1_score)+0.001))
            plt.savefig('F1-score.png')
    print("F1 curve finished!")
def plot_IOU_curve(config):
    thresholds = np.arange(0.05,1.0,0.05)
    iou_score = []
    with open('valid_GT.pickle', 'rb') as file:
        with open(config.feature_model_path, 'rb') as file2:
            GT = pickle.load(file)
            predict = pickle.load(file2)
            for threshold in thresholds:
                temp_predict = np.where(predict > threshold, 1, 0)
                iou = cal_iou(GT, temp_predict)
                print('Threshold: %.2f IOU: %.4f' %(threshold, iou))
                iou_score.append(iou)
            fig = plt.figure()
            plt.plot(thresholds, iou_score, color = 'r')
            index = iou_score.index(max(iou_score))
            show_max='('+str(round(thresholds[index], 2))+' '+str(round(max(iou_score), 2))+')'
            plt.annotate(show_max,xy=(thresholds[index],max(iou_score)),xytext=(thresholds[index],max(iou_score)+0.001))
            plt.savefig('IOU-score.png')
    print("IOU curve finished!")
def plot_F2_curve(config):
    thresholds = np.arange(0.05,1.0,0.05)
    score = []
    with open('valid_GT.pickle', 'rb') as file:
        with open(config.feature_model_path, 'rb') as file2:
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
            show_max='('+str(round(thresholds[index], 2))+' '+str(round(max(score), 2))+')'
            plt.annotate(show_max,xy=(thresholds[index],max(score)),xytext=(thresholds[index],max(score)+0.001))
            plt.savefig('F2-score.png')
    print("F2 curve finished!")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_path', type=str, default="")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--which_model', type=int, default=0)
    parser.add_argument('--continue_num', nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--Unet_3D_channel', type=int, default=64)
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--feature_model_path', type=str, default="")
    config = parser.parse_args()
    #read_predict_GT_mask(config)
    #plot_ROC_curve(config)
    #plot_PR_curve(config)2
    #plot_F1_curve(config)
    #plot_IOU_curve(config)
