import numpy as np
import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
from eval import *
from PIL import Image
import imageio
from mean_iou_evaluate import *
## need to remove before submit
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
##net work
from network.FCN32s import *
from network.HDC import *
from network.FCN8s import *
from network.Pspnet import *
from network.GCN import *
from network.T_FCN8s import *
from network.T_Res_Unet import *
from network.Vgg_Unet import *
from loss_func import *
import segmentation_models_pytorch as smp
def train_continuous(config, net, threshold, best_score, criterion, OPTIMIZER, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR):
    Sigmoid_func = nn.Sigmoid()
    for epoch in range(EPOCH):
        train_loss = 0
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        net.train()
        for i, (file_name, image_list, mask_list) in tqdm(enumerate(train_loader)):
            pn_frame = image_list[:,1:,:,:,:]
            frame = image_list[:,:1,:,:,:]
            mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
            pn_mask = mask_list[:,1:,:,:]
            output = net(frame, pn_frame).squeeze(dim = 1)
            loss = criterion(output, mask.float())
            OPTIMIZER.zero_grad() 
            loss.backward()
            OPTIMIZER.step()
            train_loss += loss.item()
            SR = torch.where(output > threshold, 1, 0).cpu()
            GT = mask.cpu()
            acc += get_accuracy(SR,GT, threshlod)
            SE += get_sensitivity(SR,GT, threshlod)
            SP += get_specificity(SR,GT, threshlod)
            PC += get_precision(SR,GT, threshlod)
            F1 += get_F1(SR,GT, threshlod)
            JS += get_JS(SR,GT, threshlod)
            DC += get_DC(SR,GT, threshlod)
            if i % 50 == 0:
                print('[Training] Loss: %.4f Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (train_loss/(i+1),acc /(i+1), SE/(i+1), SP/(i+1), PC/(i+1), F1 /(i+1), JS/(i+1), DC/(i+1)) )
        length = len(train_loader)
        train_loss / length
        acc = acc/length
        SE = SE/length
        SP = SP/length
        PC = PC/length
        F1 = F1/length
        JS = JS/length
        DC = DC/length
        print('Epoch [%d] [Training] Loss: %.4f Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (epoch+1,train_loss,acc,SE,SP,PC,F1,JS,DC))
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.
        with torch.no_grad():
            net.eval()
            for i, (file_name, image_list, mask_list) in tqdm(enumerate(valid_loader)):
                pn_frame = image_list[:,1:,:,:,:]
                frame = image_list[:,:1,:,:,:]
                mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                pn_mask = mask_list[:,1:,:,:]
                output = net(frame, pn_frame).squeeze(dim = 1)
                loss = criterion(output, mask.float())
                train_loss += loss.item()
                SR = torch.where(output > threshold, 1, 0).cpu()
                GT = mask.cpu()
                acc += get_accuracy(SR,GT, threshold)
                SE += get_sensitivity(SR,GT, threshold)
                SP += get_specificity(SR,GT, threshold)
                PC += get_precision(SR,GT, threshold)
                F1 += get_F1(SR,GT, threshold)
                JS += get_JS(SR,GT, threshold)
                DC += get_DC(SR,GT, threshold)
            length = len(valid_loader)
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            score = DC + JS
            if score >= best_score:
                best_score = score
                net_save_path = config.save_model_path + model_name +"_Epoch="+str(epoch)+"_Score="+str(round(score,3))
                print("Model save in "+config.save_model_path + "Epoch="+str(epoch)+"_Score="+str(round(score,3))+".pkl")
                best_net = net.state_dict()
                torch.save(best_net,net_save_path)
            print('Epoch [%d] [Validing] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (epoch+1,acc,SE,SP,PC,F1,JS,DC))
            if config.draw_image == 1:
                for i, (crop_image ,file_name, image_list) in tqdm(enumerate(test_loader)):
                    pn_frame = image_list[:,1:,:,:,:]
                    frame = image_list[:,:1,:,:,:]
                    output = net(frame, pn_frame).squeeze(dim = 1)
                    output = Sigmoid_func(output)
                    crop_image = crop_image.squeeze().data.numpy()
                    origin_crop_image = crop_image.copy()
                    SR = torch.where(output > threshlod, 1, 0).squeeze().cpu().data.numpy().astype("uint8")
                    heatmap = np.uint8(255 * SR)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heat_img = heatmap*0.9+origin_crop_image
                    write_path = file_name[0].replace("original", "heat")
                    write_path = "/".join(write_path.split("/")[:-1])
                    if not os.path.isdir(write_path):
                        os.makedirs(write_path)
                    image_save_path = os.path.join(write_path, file_name[0].split("/")[-1])
                    cv2.imwrite(image_save_path, heat_img)

def train_single(config, net, threshold, best_score, criterion, OPTIMIZER, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR):
    Sigmoid_func = nn.Sigmoid()
    for epoch in range(EPOCH):
        train_loss = 0
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        net.train()
        for i, (image, mask) in tqdm(enumerate(train_loader)):
            image = image.cuda()
            mask = mask.cuda()
            output = net(image).squeeze(dim = 1)
            loss = criterion(output, mask.float())
            OPTIMIZER.zero_grad() 
            loss.backward()
            OPTIMIZER.step()
            train_loss += loss.item()
            SR = torch.where(output > threshlod, 1, 0).cpu()
            #SR = torch.argmax(output, dim = 1).squeeze(dim = 1).cpu()
            GT = mask.cpu()
            acc += get_accuracy(SR,GT, threshlod)
            SE += get_sensitivity(SR,GT, threshlod)
            SP += get_specificity(SR,GT, threshlod)
            PC += get_precision(SR,GT, threshlod)
            F1 += get_F1(SR,GT, threshlod)
            JS += get_JS(SR,GT, threshlod)
            DC += get_DC(SR,GT, threshlod)
            if i % 50 == 0:
                print('[Training] Loss: %.4f Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (train_loss/(i+1),acc /(i+1), SE/(i+1), SP/(i+1), PC/(i+1), F1 /(i+1), JS/(i+1), DC/(i+1)) )
        length = len(train_loader)
        train_loss / length
        acc = acc/length
        SE = SE/length
        SP = SP/length
        PC = PC/length
        F1 = F1/length
        JS = JS/length
        DC = DC/length
        print('Epoch [%d] [Training] Loss: %.4f Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (epoch+1,train_loss,acc,SE,SP,PC,F1,JS,DC))
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.
        with torch.no_grad():
            net.eval()
            for i, (image, mask) in tqdm(enumerate(valid_loader)):
                image = image.cuda()
                mask = mask.cuda()
                output = net(image).squeeze(dim = 1)
                output = Sigmoid_func(output)
                loss = criterion(output, mask.float())
                train_loss += loss.item()
                SR = torch.where(output > threshlod, 1, 0).cpu()
                GT = mask.cpu()
                acc += get_accuracy(SR,GT, threshlod)
                SE += get_sensitivity(SR,GT, threshlod)
                SP += get_specificity(SR,GT, threshlod)
                PC += get_precision(SR,GT, threshlod)
                F1 += get_F1(SR,GT, threshlod)
                JS += get_JS(SR,GT, threshlod)
                DC += get_DC(SR,GT, threshlod)
            length = len(valid_loader)
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            score = DC + JS
            if score >= best_score:
                best_score = score
                net_save_path = config.save_model_path + "Epoch="+str(epoch)+"_Score="+str(round(score,3))
                print("Model save in "+config.save_model_path + "Epoch="+str(epoch)+"_Score="+str(round(score,3))+".pkl")
                best_net = net.state_dict()
                torch.save(best_net,net_save_path)
            print('Epoch [%d] [Validing] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (epoch+1,acc,SE,SP,PC,F1,JS,DC))
            for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
                image = image.cuda()
                output = net(image)
                crop_image = crop_image.squeeze().data.numpy()
                origin_crop_image = crop_image.copy()
                SR = torch.where(output > threshlod, 1, 0).squeeze().cpu().data.numpy().astype("uint8")
                contours, hierarchy = cv2.findContours(SR, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                write_path = file_name[0].replace("original", "forfilm")
                write_path = "/".join(write_path.split("/")[:-1])
                if not os.path.isdir(write_path):
                    os.makedirs(write_path)
                image_save_path = os.path.join(write_path, file_name[0].split("/")[-1])
                if contours ==[]:
                    imageio.imwrite(image_save_path, crop_image)
                else:
                    cv2.drawContours(np.uint8(crop_image), contours, -1, (0,255,0), 3)
                    imageio.imwrite(image_save_path, crop_image)