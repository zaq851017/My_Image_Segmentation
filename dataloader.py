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
            self.image_paths = list(map(lambda x: os.path.join(self.root+"images", x), os.listdir(os.path.join(self.root, "images"))))
            self.mask_paths = list(map(lambda x: os.path.join(self.root+"masks", x), os.listdir(os.path.join(self.root, "masks"))))
            self.mask_paths.sort()
        if mode == "test":
            self.image_paths = list(map(lambda x: os.path.join(self.root+"images", x), os.listdir(os.path.join(self.root, "images"))))
        self.image_paths.sort()
        self.mode = mode
        self.augmentation_prob = prob
        self.RotationDegree = [0,90,180,270]
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "valid":
            file_name = self.image_paths[index].split("/")[-1]
            image_path = self.image_paths[index]
            mask_path = self.mask_paths[index]
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert("L")
            aspect_ratio = image.size[1]/image.size[0]
            p_transform = random.random()
            if (self.mode == 'train') and p_transform <= self.augmentation_prob:
                Transform = []
                RotationDegree = random.randint(0,3)
                RotationDegree = self.RotationDegree[RotationDegree]
                if (RotationDegree == 90) or (RotationDegree == 270):
                    aspect_ratio = 1/aspect_ratio

                Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
                            
                RotationRange = random.randint(-10,10)
                Transform.append(T.RandomRotation((RotationRange,RotationRange)))
                Transform = T.Compose(Transform)
                
                image = Transform(image)
                mask = Transform(mask)

                if random.random() < 0.5:
                    image = F.hflip(image)
                    mask = F.hflip(mask)

                if random.random() < 0.5:
                    image = F.vflip(image)
                    mask = F.vflip(mask)
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
            file_name = self.image_paths[index].split("/")[-1]
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

