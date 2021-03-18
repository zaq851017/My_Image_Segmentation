import numpy as np
import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms as T
from dataloader import get_continuous_loader
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
from FCN32s import *
from HDC import *
from FCN8s import *
from Pspnet import *
from GCN import *
from T_FCN8s import *
from T_Res_FCN import *
<<<<<<< Updated upstream
=======
from Vgg_Unet import *
>>>>>>> Stashed changes
from loss_func import *
import segmentation_models_pytorch as smp
def train(config, train_loader, valid_loader, test_loader, batch_size, EPOCH, LR):
    Sigmoid_func = nn.Sigmoid()
    threshlod = 0.5
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
    elif config.which_model == 7:
        net = T_FCN8s(1)
        print("Model vgg-temporal Temporal_FCN8S")
    elif config.which_model == 8:
        net = T_Res_FCN(1)
        print("Model res-temporal T_Res_FCN")
<<<<<<< Updated upstream
=======
    elif config.which_model == 9:
        net = Vgg_Unet(1)
        print("Model vgg-unet Vgg_Unet")
>>>>>>> Stashed changes
    elif config.which_model == 0:
        print("No assign which model!")
    if config.pretrain_model != "":
        net.load_state_dict(torch.load(config.pretrain_model))
        print("pretrain model loaded!")
    net = net.cuda()
    #feature_extractor = feature_extractor.cuda()
    best_score = config.best_score
    train_weight = torch.FloatTensor([10 / 1]).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight = train_weight)
    #criterion = DiceBCELoss()
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
            SR = torch.where(output > threshlod, 1, 0).cpu()
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
def main(config):
    # parameter setting
    LR = config.learning_rate
    EPOCH = config.epoch
    BATCH_SIZE = config.batch_size


    train_loader = get_continuous_loader(image_path = "Medical_data/train/", 
                            batch_size = BATCH_SIZE,
                            mode = 'train',
                            augmentation_prob = 0.,
                            shffule_yn = True)
    valid_loader = get_continuous_loader(image_path = "Medical_data/valid/",
                            batch_size = 1,
                            mode = 'valid',
                            augmentation_prob = 0.,
                            shffule_yn = False)
    test_loader = get_continuous_loader(image_path = "Medical_data/test/",
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
    parser.add_argument('--draw_image', type=int, default=0)
    config = parser.parse_args()
    main(config)
