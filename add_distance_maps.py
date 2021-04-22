import cv2
import numpy as np
import argparse
import os 
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default='')
    parser.add_argument("--mode",type=str, default='')
    args = parser.parse_args()
    if args.mode == 'train':
        print("train")
        for num_files in LISTDIR(args.data_path):
            full_path = os.path.join(args.data_path, num_files)
            for dir_files in LISTDIR(full_path):
                full_path_2 = os.path.join(full_path, dir_files)
                mask_path = os.path.join(full_path_2, "mask")
                origin_path = os.path.join(full_path_2, "original")
                total_x = []
                total_y = []
                check = 0
                for mask_files in LISTDIR(mask_path):
                    mask_img_path = os.path.join(mask_path, mask_files)
                    img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
                    if np.sum(img) != 0:
                        x_range = img.nonzero()[0]
                        y_range = img.nonzero()[1]
                        total_x.append(sum(x_range)/len(x_range))
                        total_y.append(sum(y_range)/len(y_range))
                avg_x = sum(total_x)/len(total_x)
                avg_y = sum(total_y)/len(total_y)
                corner = 0
                for i in [0, 540]:
                    for j in [0, 720]:
                        temp = (abs(i-avg_x)**2 + abs(j-avg_y)**2)**0.5
                        if temp >= corner:
                            corner = temp
                output_path = origin_path.replace("original", "distance_map")
                if not os.path.isdir(output_path):
                    print("os.makedirs", output_path)
                    os.makedirs(output_path)
                for mask_files in LISTDIR(origin_path):
                    if check == 0:
                        mask = np.zeros((540,720))
                        for i in range(540):
                            for j in range(720):
                                res = (1-( (abs(i-avg_x)**2 + abs(j-avg_y)**2)**0.5 / corner))
                                mask[i][j] = res
                        heatmap = np.uint8(255 * mask)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        print(mask_path, avg_x, avg_y)
                    check = 1
                    cv2.imwrite(os.path.join(output_path, mask_files), heatmap)
    elif args.mode == 'valid' or args.mode == 'test':
        print("valid")
        for num_files in LISTDIR(args.data_path):
            full_path = os.path.join(args.data_path, num_files)
            for dir_files in LISTDIR(full_path):
                full_path_2 = os.path.join(full_path, dir_files)
                mask_path = os.path.join(full_path_2, "mask")
                origin_path = os.path.join(full_path_2, "original")
                output_path = mask_path.replace("mask", "distance_map")
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                corner = (340 **2 + 360**2)**0.5
                check = 0
                for mask_files in LISTDIR(origin_path):
                    if check == 0:
                        mask = np.zeros((540,720))
                        for i in range(540):
                            for j in range(720):
                                res = (1-( (abs(i-200)**2 + abs(j-360)**2)**0.5 / corner))
                                mask[i][j] = res
                        heatmap = np.uint8(255 * mask)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        print(mask_path, corner)
                    check = 1
                    cv2.imwrite(os.path.join(output_path, mask_files), heatmap)