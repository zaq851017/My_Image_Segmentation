import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
import argparse
import ipdb

def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
if __name__ == "__main__":
    path = "./video_dataset/"
    output_path = './all_medical_dataset/'
    for num_files in LISTDIR(path):
        full_path = os.path.join(path, num_files)
        for date_file in LISTDIR(full_path):
            full_path_2 = os.path.join(full_path, date_file)
            for video_file in LISTDIR(full_path_2):
                full_path_3 = os.path.join(full_path_2, video_file)
                print(full_path_3)
                vidcap = cv2.VideoCapture(full_path_3)
                file_name = str(num_files) + "_" + str(date_file) + "_" + str(video_file[:-4])+"_"
                success,image = vidcap.read()
                count = 0
                success = True
                num_file_name = full_path_3.split("/")[2]
                dir_file_name = "_".join(full_path_3.split("/")[2:5])[:-4]+"_film"
                output_name = os.path.join(output_path, num_file_name, dir_file_name,"original")
                if not os.path.isdir(output_name ):
                    os.makedirs(output_name )
                while success:
                    if count<10:
                        print(( output_name + "/frame00%d.jpg" % count) )
                        cv2.imwrite( output_name + "/frame00%d.jpg" % count, image) 
                    elif count < 100 and count > 9:
                        cv2.imwrite(output_name+ "/frame0%d.jpg" % count, image)
                    elif count > 99: 
                        cv2.imwrite(output_name+ "/frame%d.jpg" % count, image)
                    count += 1 
                    success,image = vidcap.read() 
