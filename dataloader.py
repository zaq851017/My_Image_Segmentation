import os
import numpy as np
import torch
import random
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import ipdb

def label_mask(m_array):
    new_mrray = np.zeros((m_array.shape[0],m_array.shape[1]))
    new_mrray[m_array >= 128 ] = 1
    return new_mrray
    
class ImageFolder(data.Dataset):
    def __init__(self, root, prob, mode = 'train'):
        self.root = root
        if mode == "train" or mode == "valid":
            self.image_paths = []
            self.mask_paths = []
            for num_file in os.listdir(self.root):
                full_path = os.path.join(self.root, num_file)
                for dir_file in os.listdir(full_path):
                    temp_img_list = []
                    temp_mask_list = []
                    full_path_2 = os.path.join(full_path, dir_file)
                    for original_file in os.listdir(os.path.join(full_path_2, "original")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "original", original_file))
                        temp_img_list.append(full_path_3)
                    for original_file in os.listdir(os.path.join(full_path_2, "mask")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "mask", original_file))
                        temp_mask_list.append(full_path_3)
                    temp_img_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    temp_mask_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    self.image_paths += temp_img_list
                    self.mask_paths += temp_mask_list
            new_img_path = []
            for img in self.image_paths:
                img = img.replace("original", "mask")
                new_img_path.append(img.replace(".jpg", "_out.jpg"))
            if new_img_path == self.mask_paths:
                print("Image and Masks are correct")
            else:
                for i in range(len(self.image_paths)):
                    img = new_img_path[i]
                    mask = self.mask_paths[i]
                    if img != mask:
                        print(img,mask,i)
        if mode == "test":
            self.image_paths = []
            for num_file in os.listdir(self.root):
                full_path = os.path.join(self.root, num_file)
                for dir_file in os.listdir(full_path):
                    temp_img_list = []
                    full_path_2 = os.path.join(full_path, dir_file)
                    for original_file in os.listdir(os.path.join(full_path_2, "original")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "original", original_file))
                        temp_img_list.append(full_path_3)
                    temp_img_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    self.image_paths += temp_img_list
        self.mode = mode
        self.augmentation_prob = prob
        self.RotationDegree = [0,90,180,270]
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "valid":
            image_path = self.image_paths[index]
            mask_path = self.mask_paths[index]
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert("L")
            image = image.resize((720, 540))
            image = image.crop((150,70,574,438))
            mask = mask.resize((720, 540))
            mask = mask.crop((150,70,574,438))
            """ resize 512 """
            image = image.resize((512, 512))
            mask = mask.resize((512, 512))
            """ additional """
            mask = np.array(mask)
            mask = label_mask(mask)
            Transform = []
            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)
            image = Transform(image)
            Norm_ = T.Normalize((0.486, 0.456, 0.406), (0.229, 0.224, 0.225))
            image = Norm_(image)
            
            return image, torch.tensor(mask, dtype=torch.long)    
        if self.mode == "test":
            file_name = self.image_paths[index]
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert('RGB')
            image = image.resize((720, 540))
            image = image.crop((150,70,574,438))
            """ resize 512 """
            image = image.resize((512, 512))
            """ additional """
            crop_origin_image = image
            Transform = []
            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)
            image = Transform(image)
            Norm_ = T.Normalize((0.486, 0.456, 0.406), (0.229, 0.224, 0.225))
            image = Norm_(image)
            
            return  np.array(crop_origin_image), file_name, image
            
    def __len__(self):
        return len(self.image_paths)

def get_loader(image_path, batch_size, mode, augmentation_prob, shffule_yn = False):
    dataset = ImageFolder(root = image_path, prob = augmentation_prob,mode = mode)
    data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=shffule_yn)
    return data_loader

