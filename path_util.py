import cv2
import numpy as np
import argparse
import os 
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