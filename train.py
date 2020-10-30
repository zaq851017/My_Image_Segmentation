import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
from dataloader import get_loader
from network import *
from eval import *
from PIL import Image
import imageio
from mean_iou_evaluate import *
from loss_func import *
## need to remove before submit
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def train(config, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR):
    print(config)
    if config.which_model == 1:
        net = DLCVHW2_2_Basic_Net(2)
        print("Base model FCN32S")
    elif config.which_model == 2:
        net = DLCVHW2_2_Improve_Net(2)
        print("Improve model")
    elif config.which_model == 3:
        net = FCN8s(2)
        print("Model FCN8S")
    elif config.which_model == 0:
        print("No assign which model!")
    net = net.cuda()

    best_score = config.best_score
    #class_weight = torch.FloatTensor([0.2,1.0]).cuda()
    #CRITERION = nn.CrossEntropyLoss(weight = class_weight)
    OPTIMIZER = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = LR)
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
            output = net(image)
            mask = torch.squeeze(mask)
            loss = FocalLoss(gamma=0)(output, mask)
            OPTIMIZER.zero_grad() 
            loss.backward()
            OPTIMIZER.step()
            train_loss += loss.item()
            SR = torch.argmax(output, dim = 1).squeeze().cpu()
            GT = mask.cpu()
            acc += get_accuracy(SR,GT)
            SE += get_sensitivity(SR,GT)
            SP += get_specificity(SR,GT)
            PC += get_precision(SR,GT)
            F1 += get_F1(SR,GT)
            JS += get_JS(SR,GT)
            DC += get_DC(SR,GT)
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
                mask = torch.squeeze(mask)
                output = net(image)
                SR = torch.argmax(output, dim = 1).squeeze().cpu()
                GT = mask.cpu()
                SR = torch.unsqueeze(SR, 0)
                GT = torch.unsqueeze(GT, 0)
                acc += get_accuracy(SR,GT)
                SE += get_sensitivity(SR,GT)
                SP += get_specificity(SR,GT)
                PC += get_precision(SR,GT)
                F1 += get_F1(SR,GT)
                JS += get_JS(SR,GT)
                DC += get_DC(SR,GT)
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

            
def main(config):
    # parameter setting
    LR = config.learning_rate
    EPOCH = config.epoch
    BATCH_SIZE = config.batch_size


    train_loader = get_loader(image_path = "label_data/train_dataset/",
                            batch_size = BATCH_SIZE,
                            mode = 'train',
                            augmentation_prob = 0.4,
                            shffule_yn = True)
    valid_loader = get_loader(image_path = "label_data/test_dataset/",
                            batch_size = 1,
                            mode = 'valid',
                            augmentation_prob = 0.,
                            shffule_yn = False)
    test_loader = get_loader(image_path = "label_data/test_dataset/",
                            batch_size = 1,
                            mode = 'test',
                            augmentation_prob = 0.,
                            shffule_yn = False)
    train(config, train_loader, valid_loader, test_loader, BATCH_SIZE, EPOCH, LR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model', type=str, default="")
    parser.add_argument('--which_model', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=int, default=1e-4)
    parser.add_argument('--save_model_path', type=str, default="./models/")
    parser.add_argument('--best_score', type=float, default=1.0)
    config = parser.parse_args()
    main(config)