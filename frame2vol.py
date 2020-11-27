import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
import argparse
import ipdb
import matplotlib.pyplot as plt
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def vol_cal(frame_file_path):
    pixel_area = (1 / 85)**2  # h = 10/85(mm)
    for num_files in LISTDIR(frame_file_path):
        full_path = os.path.join(frame_file_path, num_files)
        for date_file in LISTDIR(full_path):
            full_path_2 = os.path.join(full_path, date_file)
            height_file = os.path.join(full_path_2, "height.txt")
            f = open(height_file, "r")
            height = float(f.read())/10.0
            f.close()
            mask_path = os.path.join(full_path_2, "mask")
            mask_list = []
            for mask_file in LISTDIR(mask_path):
                full_path_3 = os.path.join(mask_path, mask_file)
                mask = cv2.imread(full_path_3, cv2.IMREAD_GRAYSCALE)
                mask_list.append(mask)
            left_area = 0.0
            right_area = 0.0
            area = []
            index = []
            for i in range(0, len(mask_list)-1):
                left_area += np.count_nonzero(mask_list[i] != 0.0)
                temp_area = np.count_nonzero(mask_list[i] != 0.0) * pixel_area / len(mask_list)
                area.append(temp_area)
                index.append(i)
            temp_area = np.count_nonzero(mask_list[len(mask_list)-1] != 0.0) * pixel_area / len(mask_list)
            area.append(temp_area)
            index.append(len(mask_list))
            for i in range(1, len(mask_list)):
                right_area += np.count_nonzero(mask_list[i] != 0.0)    
            ll = (left_area * pixel_area)
            rr = (right_area * pixel_area)
            result = (ll+rr)*height/(2*len(mask_list))
            plt.title("Volume: "+str(round(result,4))+" ml")
            plt.xlabel("frame")
            plt.ylabel("area(cm^2)")
            plt.bar(index, area, width=1.0)
            #plt.show()
            save_path = os.path.join(full_path_2, "Volume.jpg")
            plt.savefig(save_path)
            plt.close()
            print(save_path)
if __name__ == "__main__":
    vol_result = vol_cal("./my_label_result/")