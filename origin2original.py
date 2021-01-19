import numpy as np
import os
import argparse
from tqdm import tqdm
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--input_path', type=str, default="")
    config = parser.parse_args()
    for dir_files in (LISTDIR(config.input_path)):
        full_path = os.path.join(config.input_path, dir_files)
        for num_files in tqdm(LISTDIR(full_path)):
            full_path_2 = os.path.join(full_path, num_files+"/origin")
            r_path = "/".join(full_path_2.split("/")[2:]).replace("origin", "original")
            r_path = os.path.join(config.output_path, r_path)
            os.system("mv "+full_path_2+" "+r_path)
