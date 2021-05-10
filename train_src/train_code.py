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
## need to remove before submit
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import segmentation_models_pytorch as smp
import copy
import random
def train_single(config, logging, net, model_name, threshold, best_score, criterion, OPTIMIZER, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR, now_time):
    Sigmoid_func = nn.Sigmoid()
    if config.random_train == 0:
        random_para = -0.7
    elif config.random_train == 1:
        random_para = 0.7
    for epoch in range(EPOCH):
        net.train()
        train_Scorer = Scorer(config)
        train_Losser = Losser()
        for i, (image, mask) in tqdm(enumerate(train_loader)):
            if random.random() >= random_para:
                image = image.cuda()
                mask = mask.cuda()
                if config.which_model != 4:
                    output = net(image).squeeze(dim = 1)
                    loss = criterion(output, mask.float())
                    output = Sigmoid_func(output)
                    SR = torch.where(output > threshold, 1, 0).cpu()
                    GT = mask.cpu()
                else:
                    output5, output4, output3, output2 = net(image)
                    output5 = output5.squeeze(dim = 1)
                    output4 = output5.squeeze(dim = 1)
                    output3 = output5.squeeze(dim = 1)
                    output2 = output5.squeeze(dim = 1)
                    loss2 = criterion(output2, mask.float())
                    loss3 = criterion(output2, mask.float())
                    loss4 = criterion(output2, mask.float())
                    loss5 = criterion(output2, mask.float())
                    loss = loss2 + loss3 + loss4 + loss5
                    output = Sigmoid_func(output2)
                    SR = torch.where(output2 > threshold, 1, 0).cpu()
                    GT = mask.cpu()
                if random.random() >= random_para:
                    OPTIMIZER.zero_grad() 
                    loss.backward()
                    OPTIMIZER.step()
                train_Losser.add(loss.item())
            if i % 100 == 1:
                train_Scorer.add(SR, GT)
                logging.info('Epoch[%d] Training[%d/%d] F1: %.4f, IOU : %.4f Loss: %.4f' %(epoch+1, i,len(train_loader) ,train_Scorer.f1(), train_Scorer.iou(), train_Losser.mean()))
        with torch.no_grad():
            net.eval()
            valid_Scorer = Scorer(config)
            valid_Losser = Losser()
            for i, (image, mask) in tqdm(enumerate(valid_loader)):
                image = image.cuda()
                mask = mask.cuda()
                if config.which_model != 4:
                    output = net(image).squeeze(dim = 1)
                    loss = criterion(output, mask.float())
                    output = Sigmoid_func(output)
                    SR = torch.where(output > threshold, 1, 0).cpu()
                    GT = mask.cpu()
                else:
                    output5, output4, output3, output2 = net(image)
                    output5 = output5.squeeze(dim = 1)
                    output4 = output5.squeeze(dim = 1)
                    output3 = output5.squeeze(dim = 1)
                    output2 = output5.squeeze(dim = 1)
                    loss2 = criterion(output2, mask.float())
                    loss3 = criterion(output2, mask.float())
                    loss4 = criterion(output2, mask.float())
                    loss5 = criterion(output2, mask.float())
                    loss = loss2 + loss3 + loss4 + loss5
                    output = Sigmoid_func(output2)
                    SR = torch.where(output2 > threshold, 1, 0).cpu()
                    GT = mask.cpu()
                valid_Scorer.add(SR, GT)
                valid_Losser.add(loss.item())
            f1 = valid_Scorer.f1()
            iou = valid_Scorer.iou()
            logging.info('Epoch [%d] [Valid] F1: %.4f, IOU: %.4f, Loss: %.4f' %(epoch+1, f1, iou, valid_Losser.mean()))
            if not os.path.isdir(config.save_model_path + now_time + model_name):
                os.makedirs(config.save_model_path + now_time + model_name)
            if f1 >= best_score or epoch % 5 == 0:
                best_score = f1
                net_save_path = os.path.join(config.save_model_path, now_time+model_name)
                net_save_path = os.path.join(net_save_path, "Epoch="+str(epoch+1)+"_Score="+str(round(f1,4))+".pt")
                logging.info("Model save in "+ net_save_path)
                best_net = net.state_dict()
                torch.save(best_net,net_save_path)

def train_continuous(config, logging, net, model_name, threshold, best_score, criterion_single, criterion_temporal, OPTIMIZER, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR, continue_num, now_time):
    Sigmoid_func = nn.Sigmoid()
    if config.random_train == 0:
        random_para = -0.7
    elif config.random_train == 1:
        random_para = 0.8
    for epoch in range(EPOCH):
        net.train()
        train_Scorer = Scorer(config)
        Temporal_Losser = Losser()
        Single_Losser = Losser()
        for i, (file_name, image_list, mask_list) in tqdm(enumerate(train_loader)):
            pn_frame = image_list[:,1:,:,:,:]
            frame = image_list[:,:1,:,:,:]
            mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
            pn_mask = mask_list[:,1:,:,:].cuda()
            temporal_mask, output = net(frame, pn_frame)
            output = output.squeeze(dim = 1)
            loss = criterion_single(output, mask.float())
            pn_loss = criterion_temporal(temporal_mask, pn_mask)
            GT = mask.cpu()
            if random.random() >= random_para:
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
            Valid_Temporal_Losser = Losser()
            Valid_Single_Losser = Losser()
            for i, (file_name, image_list, mask_list) in tqdm(enumerate(valid_loader)):
                pn_frame = image_list[:,1:,:,:,:]
                frame = image_list[:,:1,:,:,:]
                mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                pn_mask = mask_list[:,1:,:,:].cuda()
                temporal_mask, output = net(frame, pn_frame)
                output = output.squeeze(dim = 1)
                loss = criterion_single(output, mask.float())
                pn_loss = criterion_temporal(temporal_mask, pn_mask)
                GT = mask.cpu() 
                output = Sigmoid_func(output)
                SR = torch.where(output > threshold, 1, 0).cpu()
                valid_Scorer.add(SR, GT)
                Valid_Temporal_Losser.add(pn_loss.item())
                Valid_Single_Losser.add(loss.item())
            f1 = valid_Scorer.f1()
            iou = valid_Scorer.iou()
            logging.info('Epoch [%d] [Valid] F1: %.4f, IOU: %.4f, Temporal_Loss: %.4f, Single_Loss: %.4f' %(epoch+1, f1, iou, Valid_Temporal_Losser.mean(), Valid_Single_Losser.mean()))
            if not os.path.isdir(os.path.join(config.save_model_path, now_time + model_name +str(continue_num))):
                os.makedirs(os.path.join(config.save_model_path, now_time + model_name+str(continue_num)))
            if f1 >= best_score or epoch % 5 == 0:
                best_score = f1
                net_save_path = os.path.join(config.save_model_path, now_time+model_name+str(continue_num))
                net_save_path = os.path.join(net_save_path, "Epoch="+str(epoch+1)+"_Score="+str(round(f1,4))+".pt")
                logging.info("Model save in "+ net_save_path)
                best_net = net.state_dict()
                torch.save(best_net,net_save_path)

def train_temporal(config, logging, net, model_name, threshold, best_score, criterion_single, criterion_temporal, OPTIMIZER, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR, continue_num, now_time):
    Sigmoid_func = nn.Sigmoid()
    for epoch in range(EPOCH):
        net.train()
        Temporal_Losser = Losser()
        for i, (file_name, image_list, mask_list) in tqdm(enumerate(train_loader)):
            pn_frame = image_list[:,1:,:,:,:]
            frame = image_list[:,:1,:,:,:]
            mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
            pn_mask = mask_list[:,1:,:,:].cuda()
            temporal_mask = net(frame, pn_frame).squeeze(dim = 1)
            pn_loss = criterion_temporal(temporal_mask, pn_mask)
            total_loss = pn_loss
            OPTIMIZER.zero_grad() 
            total_loss.backward()
            OPTIMIZER.step()
            Temporal_Losser.add(pn_loss.item())
            if i % 100 == 1:
                logging.info('Epoch[%d] Training[%d/%d], Temporal_Loss: %.4f' %(epoch+1, i,len(train_loader), Temporal_Losser.mean()))
        with torch.no_grad():
            net.eval()
            Valid_Temporal_Losser = Losser()
            for i, (file_name, image_list, mask_list) in tqdm(enumerate(valid_loader)):
                pn_frame = image_list[:,1:,:,:,:]
                frame = image_list[:,:1,:,:,:]
                mask = mask_list[:,:1,:,:].squeeze(dim = 1).cuda()
                pn_mask = mask_list[:,1:,:,:].cuda()
                temporal_mask = net(frame, pn_frame).squeeze(dim = 1)
                pn_loss = criterion_temporal(temporal_mask, pn_mask)
                Valid_Temporal_Losser.add(pn_loss.item())
                if config.draw_temporal == 1:
                    temp = [config.draw_image_path] + file_name[0][0].split("/")[2:-2]
                    write_path = "/".join(temp)
                    if not os.path.isdir(write_path+"/temporal_mask"):
                        os.makedirs(write_path+"/temporal_mask")
                    for j in range(temporal_mask.shape[1]):
                        temporal_res = temporal_mask[:,j:j+1,:,:].squeeze(dim = 1)
                        t_SR = torch.where(temporal_res > threshold, 1, 0).squeeze().cpu().data.numpy()
                        t_img_name = file_name[0][0].split("/")[-1].split(".")[0]+"_"+str(j)+".jpg"
                        cv2.imwrite(os.path.join(write_path+"/temporal_mask", t_img_name), t_SR*255)
            logging.info('Epoch [%d] Temporal_Loss: %.4f' %(epoch+1, Valid_Temporal_Losser.mean()))
            if not os.path.isdir(os.path.join(config.save_model_path, now_time + model_name +str(continue_num))):
                os.makedirs(os.path.join(config.save_model_path, now_time + model_name+str(continue_num)))
            net_save_path = os.path.join(config.save_model_path, now_time+model_name+str(continue_num))
            net_save_path = os.path.join(net_save_path, "Epoch="+str(epoch+1)+".pt")
            logging.info("Model save in "+ net_save_path)
            best_net = net.state_dict()
            torch.save(best_net,net_save_path)