import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
import argparse
import ipdb
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
    parser.add_argument('--video_path', type=str, default="")
    parser.add_argument('--output_img_path', type=str, default="")
    config = parser.parse_args()
    for num_file in LISTDIR(config.video_path):
        full_path = os.path.join(config.video_path, num_file)
        for video_file in LISTDIR(full_path):
            full_path_2 = os.path.join(full_path, video_file)
            for video_file in LISTDIR(full_path_2):
                video_path = os.path.join(full_path_2, video_file)
                if video_path.split(".")[-1] == 'avi':
                    vidcap = cv2.VideoCapture(video_path)
                    success,image = vidcap.read()
                    count = 0
                    success = True
                    dir_write_path = os.path.join( config.output_img_path,"/".join(video_path.split("/")[2:-2]), "_".join(video_path.split("/")[-2:]).split(".")[0])
                    write_path = os.path.join(dir_write_path, "original")
                    os.makedirs(write_path)
                    while success:
                        image = cv2.resize(image, (720, 540), cv2.INTER_CUBIC)
                        if count<10:
                            cv2.imwrite(write_path+"/frame00%d.jpg" % count, image) 
                        elif count < 100 and count > 9:
                            cv2.imwrite(write_path+"/frame0%d.jpg" % count, image)
                        elif count > 99: 
                            cv2.imwrite(write_path+"/frame%d.jpg" % count, image)
                        count += 1 
                        success,image = vidcap.read()
    print("video to frame finised!")
    o_files = read_dir_path(config.output_img_path, "original")
    for files in o_files:
        img = Image.open(files).convert('RGB')
        if img.size != (720, 540):
            print(files)
            img = img.resize((720, 540))
        img = img.crop((148, 72, 571, 424))
        img = img.resize((416, 352))
        img.save(files)
    print("image croped finished!")
    
    