import cv2
import ipdb
import argparse
import os
import tqdm
def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out
def frame2video(path):
    video_path = (path[:-9])
    videoWriter = cv2.VideoWriter(os.path.join(video_path,"original_video.mp4"), cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (416, 352))
    for frame_files in  LISTDIR(path):
        if frame_files[-3:] == "jpg":
            full_path = os.path.join(path, frame_files)
            frame = cv2.imread(full_path)
            videoWriter.write(frame)
    videoWriter.release()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="")
    config = parser.parse_args()
    for dir_files in (LISTDIR(config.data_path)):
        full_path = os.path.join(config.data_path, dir_files)
        o_full_path = os.path.join(config.data_path, dir_files)
        if os.path.isdir(full_path):
            print(full_path)
            for num_files in (LISTDIR(full_path)):
                full_path_3 = os.path.join(full_path, num_files+"/original")
                frame2video(full_path_3)