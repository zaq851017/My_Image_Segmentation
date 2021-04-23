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
from train_src.Score import Scorer, Losser
import logging
import time
from datetime import datetime
## need to remove before submit
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import segmentation_models_pytorch as smp
import copy
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
        Temporal_Losser = Losser()
        Single_Losser = Losser()
        for i, (file_name, image_list, mask_list) in tqdm(enumerate(train_loader)):
            if config.which_model != 13:
                pn_frame = image_list[:,1:,:,:,:]
                frame = image_list[:,:1,:,:,:]
                mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                pn_mask = mask_list[:,1:,:,:].cuda()
                temporal_mask, output = net(frame, pn_frame)
                output = output.squeeze(dim = 1)
                loss = criterion(output, mask.float())
                pn_loss = criterion(temporal_mask, pn_mask)
                GT = mask.cpu()
            else:
                pn_frame = image_list[:,1:,:,:,:]
                frame = image_list[:,:1,:,:,:]
                mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                pn_mask = mask_list[:,1:,:,:].cuda()
                output = net(pn_frame)
                output = output.squeeze(dim = 1)
                loss = torch.tensor(0)
                pn_loss = criterion(output, pn_mask.float())
                GT = mask.cpu()
                output = torch.mean(output, dim = 1)
            total_loss = loss + pn_loss
            OPTIMIZER.zero_grad() 
            total_loss.backward()
            OPTIMIZER.step()
            output = Sigmoid_func(output)
            SR = torch.where(output > threshold, 1, 0).cpu()
            Temporal_Losser.add(pn_loss.item())
            Single_Losser.add(loss.item())
            if i % 100 == 1:
                train_Scorer.add(SR, GT)
                logging.info('Epoch[%d] Training[%d/%d] F1: %.4f, IOU : %.4f, Temporal_Loss: %.4f, Single_Loss: %.4f' %(epoch+1, i,len(train_loader) ,train_Scorer.f1(), train_Scorer.iou(), Temporal_Losser.mean(), Single_Losser.mean()))
        with torch.no_grad():
            net.eval()
            valid_Scorer = Scorer(config)
            for i, (file_name, image_list, mask_list) in tqdm(enumerate(valid_loader)):
                if config.which_model != 13:
                    pn_frame = image_list[:,1:,:,:,:]
                    frame = image_list[:,:1,:,:,:]
                    mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                    pn_mask = mask_list[:,1:,:,:].cuda()
                    temporal_mask, output = net(frame, pn_frame)
                    output = output.squeeze(dim = 1)
                    GT = mask.cpu()
                else:
                    pn_frame = image_list[:,1:,:,:,:]
                    frame = image_list[:,:1,:,:,:]
                    mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                    pn_mask = mask_list[:,1:,:,:].cuda()
                    output = net(pn_frame)
                    output = output.squeeze(dim = 1)
                    output = torch.mean(output, dim = 1)
                    GT = mask.cpu()                
                output = Sigmoid_func(output)
                SR = torch.where(output > threshold, 1, 0).cpu()
                valid_Scorer.add(SR, GT)
            f1 = valid_Scorer.f1()
            iou = valid_Scorer.iou()
            logging.info('Epoch [%d] [Valid] F1: %.4f, IOU: %.4f' %(epoch+1, f1, iou))
            if not os.path.isdir(config.save_model_path + now_time + model_name):
                os.makedirs(config.save_model_path + now_time + model_name)
            if f1 >= best_score or epoch % 5 == 0:
                best_score = f1
                net_save_path = os.path.join(config.save_model_path, now_time+model_name)
                net_save_path = os.path.join(net_save_path, "Epoch="+str(epoch+1)+"_Score="+str(round(f1,3))+".pt")
                logging.info("Model save in "+ net_save_path)
                best_net = net.state_dict()
                torch.save(best_net,net_save_path)
            if config.draw_image == 1:
                if config.draw_image_path == "":
                    continue
                net.eval()
                for i, (crop_image ,file_name, image_list) in tqdm(enumerate(test_loader)):
                    pn_frame = image_list[:,1:,:,:,:]
                    frame = image_list[:,:1,:,:,:]
                    temporal_mask, output = net(frame, pn_frame)
                    output = output.squeeze(dim = 1)
                    if config.which_model != 13:
                        pn_frame = image_list[:,1:,:,:,:]
                        frame = image_list[:,:1,:,:,:]
                        mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                        pn_mask = mask_list[:,1:,:,:].cuda()
                        temporal_mask, output = net(frame, pn_frame)
                        output = output.squeeze(dim = 1)
                    else:
                        pn_frame = image_list[:,1:,:,:,:]
                        frame = image_list[:,:1,:,:,:]
                        mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                        pn_mask = mask_list[:,1:,:,:].cuda()
                        output = net(pn_frame)
                        output = output.squeeze(dim = 1)
                    output = Sigmoid_func(output)
                    crop_image = crop_image.squeeze().data.numpy()
                    origin_crop_image = crop_image.copy()
                    SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy().astype("uint8")
                    heatmap = np.uint8(255 * SR)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heat_img = heatmap*0.9+origin_crop_image
                    write_path = file_name[0].replace("original", "heat")
                    write_path = "/".join(write_path.split("/")[:-1])
                    save_path = os.path.join(config.draw_image_path, "/".join(write_path.split("/")[2:]))
                    heatmap = np.uint8(255 * SR)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heat_img = heatmap*0.9+origin_crop_image
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    image_save_path = os.path.join(save_path, file_name[0].split("/")[-1])
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
        train_Losser = Losser()
        for i, (image, mask) in tqdm(enumerate(train_loader)):
            image = image.cuda()
            mask = mask.cuda()
            output = net(image).squeeze(dim = 1)
            SR = torch.where(output > threshold, 1, 0).cpu()
            GT = mask.cpu()
            loss = criterion(output, mask.float())
            OPTIMIZER.zero_grad() 
            loss.backward()
            OPTIMIZER.step()
            output = Sigmoid_func(output)
            train_Losser.add(loss.item())
            if i % 100 == 1:
                train_Scorer.add(SR, GT)
                logging.info('Epoch[%d] Training[%d/%d] F1: %.4f, IOU : %.4f Loss: %.4f' %(epoch+1, i,len(train_loader) ,train_Scorer.f1(), train_Scorer.iou(), train_Losser.mean()))
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
            iou = valid_Scorer.iou()
            logging.info('Epoch [%d] [Valid] F1: %.4f, IOU: %.4f' %(epoch+1, f1, iou))
            if not os.path.isdir(config.save_model_path + now_time + model_name):
                os.makedirs(config.save_model_path + now_time + model_name)
            if f1 >= best_score or epoch % 5 == 0:
                best_score = f1
                net_save_path = os.path.join(config.save_model_path, now_time+model_name)
                net_save_path = os.path.join(net_save_path, "Epoch="+str(epoch+1)+"_Score="+str(round(f1,3))+".pt")
                logging.info("Model save in "+ net_save_path)
                best_net = net.state_dict()
                torch.save(best_net,net_save_path)
            if config.draw_image == 1:
                if config.draw_image_path == "":
                    continue
                net.eval()
                for i, (crop_image ,file_name, image) in tqdm(enumerate(test_loader)):
                    image = image.cuda()
                    output = net(image).squeeze(dim = 1)
                    output = Sigmoid_func(output)
                    crop_image = crop_image.squeeze().data.numpy()
                    origin_crop_image = crop_image.copy()
                    SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy().astype("uint8")
                    write_path = file_name[0].replace("original", "forfilm")
                    write_path = "/".join(write_path.split("/")[:-1])
                    save_path = os.path.join(config.draw_image_path, "/".join(write_path.split("/")[2:]))
                    heatmap = np.uint8(255 * SR)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heat_img = heatmap*0.9+origin_crop_image
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    image_save_path = os.path.join(save_path, file_name[0].split("/")[-1])
                    cv2.imwrite(image_save_path, heat_img)
                    