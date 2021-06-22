import cv2
import numpy as np
import argparse
import os
from PIL import Image, ImageOps
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def read_dir_path(path, dir_str):
    list_dir = []
    for num_files in LISTDIR(path):
        full_path = os.path.join(path, num_files)
        for dir_files in LISTDIR(full_path):
            dir_path = os.path.join(full_path, dir_files, dir_str)
            for files in LISTDIR(dir_path):
                list_dir.append(os.path.join(dir_path, files))
    return list_dir
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default='')
    parser.add_argument("--mode",type=str, default='')
    args = parser.parse_args()
    if args.mode =='train' or args.mode =='valid':
        o_files = read_dir_path(args.data_path, "original")
        m_files = read_dir_path(args.data_path, "mask")
        h_files = read_dir_path(args.data_path, "heat")
        for files in o_files:
            img = Image.open(files).convert('RGB')
            if img.size != (720, 540):
                print(files)
                img = img.resize((720, 540))
            img = img.crop((148, 72, 571, 424))
            img = img.resize((416, 352))
            img.save(files)
        for files in m_files:
            img = Image.open(files).convert('RGB')
            if img.size != (720, 540):
                print(files)
                img = img.resize((720, 540))
            img = img.crop((148, 72, 571, 424))
            img = img.resize((416, 352))
            img.save(files)
        for files in h_files:
            img = Image.open(files).convert('RGB')
            if img.size != (720, 540):
                print(files)
                img = img.resize((720, 540))
            img = img.crop((148, 72, 571, 424))
            img = img.resize((416, 352))
            img.save(files)
    elif args.mode =='test':
        o_files = read_dir_path(args.data_path, "original")
        for files in o_files:
            img = Image.open(files).convert('RGB')
            if img.size != (720, 540):
                print(files)
                img = img.resize((720, 540))
            img = img.crop((148, 72, 571, 424))
            img = img.resize((416, 352))
            img.save(files)