import numpy as np
import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import imageio
from train_src.Score import Scorer
import logging
import time
from datetime import datetime
## need to remove before submit
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import segmentation_models_pytorch as smp
def train_continuous(config, net, model_name, threshold, best_score, criterion, OPTIMIZER, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR):
    Sigmoid_func = nn.Sigmoid()
    now_time = datetime.now().strftime("%Y_%m_%d_%I:%M:%S_")
    log_name = os.path.join('My_Image_Segmentation', 'log', now_time+"_"+model_name+".log")
    print("log_name ", log_name)
    logging.basicConfig(level=logging.DEBUG,
                        handlers = [logging.FileHandler(log_name, 'w', 'utf-8'),logging.StreamHandler()])
    logging.info(config)
    for epoch in range(EPOCH):
        net.train()
        train_Scorer = Scorer(config)
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
            output = Sigmoid_func(output)
            SR = torch.where(output > threshold, 1, 0).cpu()
            GT = mask.cpu()
            if i % 100 == 1:
                train_Scorer.add(SR, GT)
                logging.info('Epoch[%d] Training[%d/%d] F1: %.4f, F2 : %.4f' %(epoch+1, i,len(train_loader) ,train_Scorer.f1(), train_Scorer.f2()))
        with torch.no_grad():
            net.eval()
            valid_Scorer = Scorer(config)
            for i, (file_name, image_list, mask_list) in tqdm(enumerate(valid_loader)):
                pn_frame = image_list[:,1:,:,:,:]
                frame = image_list[:,:1,:,:,:]
                mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                pn_mask = mask_list[:,1:,:,:]
                output = net(frame, pn_frame).squeeze(dim = 1)
                loss = criterion(output, mask.float())
                output = Sigmoid_func(output)
                SR = torch.where(output > threshold, 1, 0).cpu()
                GT = mask.cpu()
                valid_Scorer.add(SR, GT)
            f1 = valid_Scorer.f1()
            #f2 = valid_Scorer.f2()
            logging.info('Epoch [%d] [Valid] F1: %.4f' %(epoch+1, f1))
            if not os.path.isdir(config.save_model_path + now_time + model_name):
                os.makedirs(config.save_model_path + now_time + model_name)
            if f1 >= best_score:
                best_score = f1
                net_save_path = os.path.join(config.save_model_path, now_time+model_name)
                net_save_path = os.path.join(net_save_path, "Epoch="+str(epoch+1)+"_Score="+str(round(f1,3))+".pt")
                #net_save_path = config.save_model_path + now_time + model_name +"/Epoch="+str(epoch)+"_Score="+str(round(f1,3))+".pt"
                logging.info("Model save in "+ net_save_path)
                best_net = net.state_dict()
                torch.save(best_net,net_save_path)
                if config.draw_image == 1:
                    for i, (crop_image ,file_name, image_list) in tqdm(enumerate(test_loader)):
                        pn_frame = image_list[:,1:,:,:,:]
                        frame = image_list[:,:1,:,:,:]
                        output = net(frame, pn_frame).squeeze(dim = 1)
                        output = Sigmoid_func(output)
                        crop_image = crop_image.squeeze().data.numpy()
                        origin_crop_image = crop_image.copy()
                        SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy().astype("uint8")
                        heatmap = np.uint8(255 * SR)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        heat_img = heatmap*0.9+origin_crop_image
                        write_path = file_name[0].replace("original", "heat")
                        write_path = "/".join(write_path.split("/")[:-1])
                        if not os.path.isdir(write_path):
                            os.makedirs(write_path)
                        image_save_path = os.path.join(write_path, file_name[0].split("/")[-1])
                        cv2.imwrite(image_save_path, heat_img)

def train_single(config, net, model_name, threshold, best_score, criterion, OPTIMIZER, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR):
    Sigmoid_func = nn.Sigmoid()
    now_time = datetime.now().strftime("%Y_%m_%d_%I:%M:%S_")
    log_name = os.path.join('My_Image_Segmentation', 'log', now_time+"_"+model_name+".log")
    print("log_name ", log_name)
    logging.basicConfig(level=logging.DEBUG,
                        handlers = [logging.FileHandler(log_name, 'w', 'utf-8'),logging.StreamHandler()])
    logging.info(config)                   
    for epoch in range(EPOCH):
        net.train()
        train_Scorer = Scorer(config)
        for i, (image, mask) in tqdm(enumerate(train_loader)):
            image = image.cuda()
            mask = mask.cuda()
            output = net(image).squeeze(dim = 1)
            loss = criterion(output, mask.float())
            OPTIMIZER.zero_grad() 
            loss.backward()
            OPTIMIZER.step()
            output = Sigmoid_func(output)
            SR = torch.where(output > threshold, 1, 0).cpu()
            GT = mask.cpu()
            if i % 100 == 1:
                train_Scorer.add(SR, GT)
                logging.info('Epoch[%d] Training[%d/%d] F1: %.4f, F2 : %.4f' %(epoch+1, i,len(train_loader) ,train_Scorer.f1(), train_Scorer.f2()))
        with torch.no_grad():
            net.eval()
            valid_Scorer = Scorer(config)
            for i, (image, mask) in tqdm(enumerate(valid_loader)):
                image = image.cuda()
                mask = mask.cuda()
                output = net(image).squeeze(dim = 1)
                loss = criterion(output, mask.float())
                output = Sigmoid_func(output)
                SR = torch.where(output > threshold, 1, 0).cpu()
                GT = mask.cpu()
                valid_Scorer.add(SR, GT)
            f1 = valid_Scorer.f1()
            #f2 = valid_Scorer.f2()
            logging.info('Epoch [%d] [Valid] F1: %.4f' %(epoch+1, f1))
            if not os.path.isdir(config.save_model_path + now_time + model_name):
                os.makedirs(config.save_model_path + now_time + model_name)
            if f1 >= best_score:
                best_score = f1
                net_save_path = os.path.join(config.save_model_path, now_time+model_name)
                net_save_path = os.path.join(net_save_path, "Epoch="+str(epoch+1)+"_Score="+str(round(f1,3))+".pt")
                logging.info("Model save in "+ net_save_path)
                best_net = net.state_dict()
                torch.save(best_net,net_save_path)
                if config.draw_image == 1:
                    for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
                        image = image.cuda()
                        output = net(image)
                        output = Sigmoid_func(output)
                        crop_image = crop_image.squeeze().data.numpy()
                        origin_crop_image = crop_image.copy()
                        SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy().astype("uint8")
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