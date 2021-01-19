import numpy as np
import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
from dataloader import get_loader
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
##net work
from FCN32s import *
from HDC import *
from FCN8s import *
from Pspnet import *
from GCN import *
import segmentation_models_pytorch as smp
def train(config, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR):
    threshlod = 0.2
    print(config)
    if config.which_model == 1:
        net = FCN32s(2)
        print("Base model FCN32S")
    elif config.which_model == 2:
        net = HDC(2)
        print("HDC model")
    elif config.which_model == 3:
        net = FCN8s(1)
        print("Model FCN8S")
    elif config.which_model == 4:
        net = GCN(2)
        print("GCN net")
    elif config.which_model == 5:
        net = smp.Unet('vgg16', encoder_weights='imagenet', classes=2)
        print("Unet Vgg16")
    elif config.which_model == 6:
        net = smp.PSPNet('vgg16', encoder_weights='imagenet', classes=2)
        print("PSPNet Vgg16")
    elif config.which_model == 0:
        print("No assign which model!")

    if config.pretrain_model != "":
        net.load_state_dict(torch.load(config.pretrain_model))
        print("pretrain model loaded!")
    net = net.cuda()

    best_score = config.best_score
    train_weight = torch.FloatTensor([10 / 1]).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight = train_weight)
    #train_weight = torch.FloatTensor([1, 20]).cuda()
    #criterion = nn.CrossEntropyLoss(weight = train_weight)
    #criterion = nn.CrossEntropyLoss()
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
            output = net(image).squeeze(dim = 1)
            loss = criterion(output, mask.float())
            OPTIMIZER.zero_grad() 
            loss.backward()
            OPTIMIZER.step()
            train_loss += loss.item()
            SR = torch.where(output > threshlod, 1, 0).cpu()
            #SR = torch.argmax(output, dim = 1).squeeze(dim = 1).cpu()
            GT = mask.cpu()
            acc += get_accuracy(SR,GT)
            SE += get_sensitivity(SR,GT)
            SP += get_specificity(SR,GT)
            PC += get_precision(SR,GT)
            F1 += get_F1(SR,GT)
            JS += get_JS(SR,GT)
            DC += get_DC(SR,GT)
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
                loss = criterion(output, mask.float())
                train_loss += loss.item()
                SR = torch.where(output > threshlod, 1, 0).cpu()
                #SR = torch.argmax(output, dim = 1).squeeze(dim = 1).cpu()
                GT = mask.cpu()
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

            
def main(config):
    # parameter setting
    LR = config.learning_rate
    EPOCH = config.epoch
    BATCH_SIZE = config.batch_size


    train_loader = get_loader(image_path = "Medical_data/train/",
                            batch_size = BATCH_SIZE,
                            mode = 'train',
                            augmentation_prob = 0.,
                            shffule_yn = True)
    valid_loader = get_loader(image_path = "Medical_data/valid/",
                            batch_size = 1,
                            mode = 'valid',
                            augmentation_prob = 0.,
                            shffule_yn = False)
    test_loader = get_loader(image_path = "Medical_data/valid/",
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
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_model_path', type=str, default="./models/")
    parser.add_argument('--best_score', type=float, default=0.5)
    config = parser.parse_args()
    main(config)