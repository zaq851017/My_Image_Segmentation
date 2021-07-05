import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
import torch.nn as nn
import argparse
import torch
from PIL import Image, ImageOps
from torch.utils import data
import time
import random
import segmentation_models_pytorch as smp
from torchvision import transforms as T
import torch
import torch.nn as nn
from torch.autograd import Variable
import imageio
import copy
# Batch x NumChannels x Height x Width
# UNET --> BatchSize x 1 (3?) x 240 x 240
# BDCLSTM --> BatchSize x 64 x 240 x240

''' Class CLSTMCell.
    This represents a single node in a CLSTM series.
    It produces just one time (spatial) step output.
'''


class CLSTMCell(nn.Module):

    # Constructor
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias=True):
        super(CLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels,
                              self.num_features * self.hidden_channels,
                              self.kernel_size,
                              1,
                              self.padding)

    # Forward propogation formulation
    def forward(self, x, h, c):
        # print('x: ', x.type)
        # print('h: ', h.type)
        if len(x.shape) == 3: # batch, H, W 
            x = x.unsqueeze(dim = 1)
        combined = torch.cat((x, h), dim=1)
        A = self.conv(combined)

        (Ai, Af, Ao, Ag) = torch.split(A,
                                       A.size()[1] // self.num_features,
                                       dim=1)

        i = torch.sigmoid(Ai)     # input gate
        f = torch.sigmoid(Af)     # forget gate
        o = torch.sigmoid(Ao)     # output gate
        g = torch.tanh(Ag)

        c = c * f + i * g           # cell activation state
        h = o * torch.tanh(c)     # cell hidden state

        return h, c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).to(device),
               Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).to(device))
        except:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])),
                    Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])))


''' Class CLSTM.
    This represents a series of CLSTM nodes (one direction)
'''


class CLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True):
        super(CLSTM, self).__init__()

        # store stuff
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)

        self.bias = bias
        self.all_layers = []

        # create a node for each layer in the CLSTM
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = CLSTMCell(self.input_channels[layer],
                             self.hidden_channels[layer],
                             self.kernel_size,
                             self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    # Forward propogation
    # x --> BatchSize x NumSteps x NumChannels x Height x Width
    #       BatchSize x 2 x 64 x 240 x 240
    def forward(self, x):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        outputs = []
        for step in range(steps):
            input = torch.squeeze(x[:, step, :, :, :], dim=1)
            for layer in range(self.num_layers):
                # populate hidden states for all layers
                if step == 0:
                    (h, c) = CLSTMCell.init_hidden(bsize,
                                                   self.hidden_channels[layer],
                                                   (height, width))
                    internal_state.append((h, c))
                # do forward
                name = 'cell{}'.format(layer)
                (h, c) = internal_state[layer]

                input, c = getattr(self, name)(
                    input, h, c)  # forward propogation call
                internal_state[layer] = (input, c)
            outputs.append(input)

        #for i in range(len(outputs)):
        #    print(outputs[i].size())
        return outputs


class New_BDCLSTM(nn.Module):
    # Constructor
    def __init__(self, length, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True, num_classes=1):

        super(New_BDCLSTM, self).__init__()
        self.len = length
        self.forward_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.conv = []
        for i in range(self.len):
            self.conv.append(nn.Conv2d(hidden_channels[-1], num_classes, kernel_size=1).cuda())
        # self.final_conv = nn.Conv2d(self.len, num_classes, kernel_size=1)
        self.final_conv = nn.Conv3d(self.len, num_classes, kernel_size=1)
    def forward(self, continue_list):
        F_concanate_frame = torch.tensor([]).cuda()
        for i in range(len(continue_list)):
            F_concanate_frame = torch.cat((F_concanate_frame, continue_list[i].unsqueeze(dim = 1)), dim = 1)
        yforward = self.forward_net(F_concanate_frame)
        total_y = torch.tensor([]).cuda()
        for i in range(self.len):
            F_y = self.conv[i](yforward[i])
            total_y = torch.cat( (total_y, F_y), dim = 1)
        # current_y = self.final_conv(total_y)
        current_y = self.final_conv(total_y.unsqueeze(dim = 2)).squeeze(dim = 1)
        return current_y, total_y


class New_DeepLabV3Plus_LSTM(nn.Module):
    def __init__(self, num_classes, continue_num = 8, backbone = "resnet34"):
        super().__init__()
        self.unet1 = smp.DeepLabV3Plus(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )
        self.len = continue_num
        self.lstm = New_BDCLSTM(length = continue_num, input_channels = 3, hidden_channels=[8])
    def forward(self, input, other_frame):
        temporal_mask = torch.tensor([]).cuda()
        continue_list = []
        for i in range(self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            continue_list.append(temp)
        final_predict, temporal_mask = self.lstm(continue_list)
        return temporal_mask, final_predict

def postprocess_img(o_img, final_mask_exist, continue_list):
    int8_o_img = np.array(o_img, dtype=np.uint8)
    if np.sum(int8_o_img) < 500 or final_mask_exist == 0 or continue_list == 0:
        return np.zeros((o_img.shape[0],o_img.shape[1]), dtype = np.uint8)
    else:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(int8_o_img, connectivity=8)
        index_sort = np.argsort(-stats[:,4])
        if index_sort.shape[0] > 2:
            for ll in (index_sort[2:]):
                labels[ labels == ll ] = 0
        if np.sum(labels) < 500:
            return np.zeros((o_img.shape[0],o_img.shape[1]), dtype = np.uint8)
        else:
            return np.array(labels, dtype=np.uint8)
def Check_continue(continue_list, postprocess_continue_list, bound_list, distance):
    start = 0
    end = 0
    check_start = False
    check_only_list = False 
    for i in range(len(continue_list)):
        if check_only_list == True:
            postprocess_continue_list[i] = 0
        if continue_list[i] == 1 and check_start == False and i < len(continue_list)-1:
            start = i
            check_start = True
        elif continue_list[i] == 1 and check_start == True and i < len(continue_list)-1 and i not in bound_list:
            end = i
        elif continue_list[i] == 0 and check_start == True:
            temp = (end+1) - start
            if temp < 0:
                postprocess_continue_list[start: start+1] = [0]
            if temp <= distance:
                postprocess_continue_list[start: end+1] = [0] * temp
            if temp > distance:
                check_only_list = True
            check_start = False
        elif continue_list[i] == 1 and i in  bound_list:
            end = i
            temp = (end+1) - start
            if temp < 0:
                postprocess_continue_list[end: end+1] = [0]
            if temp <= distance:
                postprocess_continue_list[start: end+1] = [0] * temp
            check_start = False
            check_only_list = False
        elif i in bound_list:
            check_only_list = False
    return postprocess_continue_list
def Cal_mask_center(mask_img):
    middle_list = {}
    for key in mask_img:
        middle_list[key] = []
        for img_index in range(len(mask_img[key])):
            img = mask_img[key][img_index]
            if np.sum(img) != 0:
                mean_x = np.mean(img.nonzero()[0])
                mean_y = np.mean(img.nonzero()[1])
            else:
                mean_x = 0
                mean_y = 0
            middle_list[key].append([mean_x, mean_y])
    return middle_list
def Cal_Local_Global_mean(middle_list, interval_num = 5):
    mean_list = {}
    global_mean_list = {}
    for key in middle_list:
        temp_total = [0] * interval_num
        temp_x = [0] * interval_num
        temp_y = [0] * interval_num
        temp_global_x = 0
        temp_global_y = 0
        temp_global_total = 0
        len_check_list = []
        for i in range(1,interval_num+1):
            len_check_list.append([(i-1), (i-1)*len(middle_list[key])/interval_num, i*len(middle_list[key])/interval_num ])
        for i, (x,y) in enumerate(middle_list[key]):
            if x!=0 and y != 0:
                temp_global_x += x
                temp_global_y += y
                temp_global_total += 1
                for j in range(len(len_check_list)):
                    if i >= len_check_list[j][1] and i< len_check_list[j][2]:
                        temp_x[len_check_list[j][0]] += x
                        temp_y[len_check_list[j][0]] += y
                        temp_total[len_check_list[j][0]] += 1
        if temp_global_total == 0:
            temp_global_total += 1
        for check_temp in range(len(temp_total)):
            if temp_total[check_temp] == 0:
                temp_total[check_temp] +=1
        temp_list = []
        for temp in range(len(temp_total)):
            temp_list.append([ temp_x[temp]/temp_total[temp], temp_y[temp]/temp_total[temp]])
        global_mean_list[key] = [temp_global_x/ temp_global_total, temp_global_y/ temp_global_total]
        mean_list[key] = temp_list
    return mean_list, global_mean_list

def Final_postprocess(middle_list, mean_list, global_mean_list, interval_num , distance ):
    final_mask_exist = []
    for key in middle_list:
        len_check_list = []
        for i in range(1,interval_num+1):
            len_check_list.append([(i-1), (i-1)*len(middle_list[key])/interval_num, i*len(middle_list[key])/interval_num ])
        for i, (x, y) in enumerate(middle_list[key]):
            if x == 0 and y == 0:
                final_mask_exist.append(0)
            else:
                for j in range(len(len_check_list)):
                    if i >= len_check_list[j][1] and i< len_check_list[j][2]:
                        abs_x = min(abs(x-global_mean_list[key][0]),abs(x - mean_list[key][len_check_list[j][0]][0]))
                        abs_y = min(abs(y-global_mean_list[key][1]), abs(y - mean_list[key][len_check_list[j][0]][1]))
                        if abs_x >= distance or abs_y >= distance:
                            final_mask_exist.append(0)
                        else:
                            final_mask_exist.append(1)
    return final_mask_exist

def LISTDIR(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def read_dir_path(path, dir_str):
    list_dir = []
    for num_files in LISTDIR(path):
        full_path = os.path.join(path, num_files)
        for dir_files in LISTDIR(full_path):
            dir_path = os.path.join(full_path, dir_files, dir_str)
            for files in LISTDIR(dir_path):
                list_dir.append(os.path.join(dir_path, files))
    return list_dir
def test_wo_postprocess(config, test_loader, net):
    if not os.path.isdir(config.output_path):
        print("os.makedirs "+ config.output_path)
        os.makedirs(config.output_path)
    OUTPUT_IMG(config, test_loader, net, False)
    MERGE_VIDEO(config)
def test_w_postprocess(config, test_loader, net):
    net.eval()
    if not os.path.isdir(config.output_path):
        os.makedirs(config.output_path)
    with torch.no_grad():
        Sigmoid_func = nn.Sigmoid()
        threshold = 0.2
        temp_mask_exist, temp_continue_list = [1] * len(test_loader), [1] * len(test_loader)
        mask_img, continue_list, bound_list = {}, [], []
        last_signal, start, end, last_film_name = 0, 0, -1, ""
        for i, (crop_image ,file_name, image) in enumerate(tqdm(test_loader)):
            if config.continuous == 0:
                image = image.cuda()
                output = net(image)
            elif config.continuous == 1:
                pn_frame = image[:,1:,:,:,:]
                frame = image[:,:1,:,:,:]
                temporal_mask, output = net(frame, pn_frame)
                output = output.squeeze(dim = 1)
            output = Sigmoid_func(output)
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
            SR = postprocess_img(SR, temp_mask_exist[i], temp_continue_list[1])
            if np.sum(SR != 0) == 0:
                continue_list.append(0)
            else:
                continue_list.append(1)
            dict_path = file_name[0].split("/")[-3]
            if dict_path not in mask_img:
                if i != 0:
                    bound_list.append(i-1)
                mask_img[dict_path] = []
                mask_img[dict_path].append(SR)
            else:
                mask_img[dict_path].append(SR)
        bound_list.append(i)
        postprocess_continue_list = copy.deepcopy(continue_list)
        postprocess_continue_list = Check_continue(continue_list, postprocess_continue_list, bound_list, distance = 30)
        middle_list = Cal_mask_center(mask_img)
        mean_list, global_mean_list = Cal_Local_Global_mean(middle_list, config.interval_num)
        final_mask_exist = Final_postprocess(middle_list, mean_list, global_mean_list, config.interval_num, config.distance)
        OUTPUT_IMG(config, test_loader, net, True, final_mask_exist, postprocess_continue_list)
        MERGE_VIDEO(config)
def frame2video(path):
    video_path = (path[:-6])
    videoWriter = cv2.VideoWriter(os.path.join(video_path,"merge_video.mp4"), cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (832, 352))
    for frame_files in  LISTDIR(path):
        if frame_files[-3:] == "jpg":
            full_path = os.path.join(path, frame_files)
            frame = cv2.imread(full_path)
            videoWriter.write(frame)
    videoWriter.release()
def film_frame2video(path):
    video_path = (path[:-7])
    videoWriter = cv2.VideoWriter(os.path.join(video_path,"film_video.mp4"), cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (416, 352))
    for frame_files in  LISTDIR(path):
        if frame_files[-3:] == "jpg":
            full_path = os.path.join(path, frame_files)
            frame = cv2.imread(full_path)
            videoWriter.write(frame)
    videoWriter.release()
def OUTPUT_IMG(config, test_loader, net, postprocess = False, final_mask_exist = [], postprocess_continue_list = []):
    if postprocess == False:
        print("No postprocessing!!")
    else:
        print("Has postprocessing!!")
    Sigmoid_func = nn.Sigmoid()
    threshold = 0.5
    with torch.no_grad():
        net.eval()
        tStart = time.time()
        for i, (crop_image ,file_name, image) in enumerate(tqdm(test_loader)):
            pn_frame = image[:,1:,:,:,:]
            frame = image[:,:1,:,:,:]
            temporal_mask, output = net(frame, pn_frame)
            output = output.squeeze(dim = 1)
            temporal_mask = Sigmoid_func(temporal_mask)
            temp = [config.output_path] + file_name[0].split("/")[-4:-2]
            write_path = "/".join(temp)
            img_name = file_name[0].split("/")[-1]
            if not os.path.isdir(write_path+"/original"):
                os.makedirs(write_path+"/original")
            if not os.path.isdir(write_path+"/forfilm"):
                os.makedirs(write_path+"/forfilm")
            if not os.path.isdir(write_path+"/merge"):
                os.makedirs(write_path+"/merge")
            if not os.path.isdir(write_path+"/vol_mask"):
                os.makedirs(write_path+"/vol_mask")
            output = Sigmoid_func(output)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
            if postprocess == True:
                SR = postprocess_img(SR, final_mask_exist[i], postprocess_continue_list[i])
                SR = np.where(SR > 0.5, 1, 0)
            heatmap = np.uint8(110 * SR)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            heat_img = heatmap*0.6+origin_crop_image
            merge_img = np.hstack([origin_crop_image, heat_img])
            cv2.imwrite(os.path.join(write_path+"/merge", img_name), merge_img)
            imageio.imwrite(os.path.join(write_path+"/original", img_name), origin_crop_image)
            cv2.imwrite(os.path.join(write_path+"/forfilm", img_name), heat_img)
            cv2.imwrite(os.path.join(write_path+"/vol_mask", img_name), SR*255)
        tEnd = time.time()
        print("Cost time(seconds)= "+str(tEnd-tStart))
def MERGE_VIDEO(config):
    for dir_files in (LISTDIR(config.output_path)):
        full_path = os.path.join(config.output_path, dir_files)
        o_full_path = os.path.join(config.output_img_path, dir_files)
        if os.path.isdir(full_path):
            for num_files in tqdm(LISTDIR(full_path)):
                full_path_2 = os.path.join(full_path, num_files+"/merge")
                full_path_3 = os.path.join(full_path, num_files+"/forfilm")
                height_path = os.path.join(o_full_path, num_files, "height.txt")
                s_height_path = os.path.join(full_path, num_files)
                os.system("cp "+height_path+" "+s_height_path)
                print("cp "+height_path+" "+s_height_path)
                frame2video(full_path_2)
                film_frame2video(full_path_3)
                if config.keep_image == 0:
                    full_path_3 = os.path.join(full_path, num_files+"/original")
                    os.system("rm -r "+full_path_3)
                    full_path_3 = os.path.join(full_path, num_files+"/vol_mask")
                    os.system("rm -r "+full_path_3)
                    full_path_3 = os.path.join(full_path, num_files+"/forfilm")
                    os.system("rm -r "+full_path_3)

def read_img_continuous(continuous_frame_num, temp_img_list ,img_dir_file, index):
    list_num = []
    frame_num = int(temp_img_list[index].split("/")[-1].split(".")[0][-3:])
    for check_frame in continuous_frame_num:
        if frame_num + check_frame < 0:
            file_path = img_dir_file+"/frame" + "%03d" % 0 + ".jpg"
        elif frame_num + check_frame > len(temp_img_list) - 1:
            file_path = img_dir_file+"/frame"+ "%03d" % (len(temp_img_list) - 1) + ".jpg"
        else:
            file_path = img_dir_file+ "/frame"+ "%03d" % (frame_num + check_frame)+ ".jpg"
        if not os.path.isfile(file_path):
            file_path = img_dir_file+"/frame" + "%03d"% frame_num+".jpg"
        list_num.append(file_path)
    return frame_num, list_num

def test_preprocess_img(image):
    crop_origin_image = image
    Transform = []
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    image = Transform(image)
    Norm_ = T.Normalize((0.486, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = Norm_(image)
    return crop_origin_image, image
class Continuos_Image(data.Dataset):
    def __init__(self, root, prob, mode = 'train', continuous_frame_num = [1, 2, 3, 4, 5, 6, 7, 8]):
        self.root = root
        self.mode = mode
        self.augmentation_prob = prob
        self.continuous_frame_num = continuous_frame_num
        if mode == "test":
            self.image_paths = {}
            for num_file in os.listdir(self.root):
                full_path = os.path.join(self.root, num_file)
                for dir_file in os.listdir(full_path):
                    temp_img_list = []
                    full_path_2 = os.path.join(full_path, dir_file)
                    for original_file in os.listdir(os.path.join(full_path_2, "original")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "original", original_file))
                        temp_img_list.append(full_path_3)
                    temp_img_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    img_dir_file = "/".join(temp_img_list[0].split("/")[:-1])
                    total_img_num = []
                    for i in range(len(temp_img_list)):
                        frame_num, list_num = read_img_continuous(self.continuous_frame_num, temp_img_list, img_dir_file, i) 
                        order_num = [img_dir_file+ "/frame" + "%03d"% frame_num+".jpg"] + list_num
                        total_img_num.append(order_num)
                    self.image_paths[img_dir_file] = total_img_num
            temp_list = [*self.image_paths.values()]
            self.image_paths_list = [val for sublist in temp_list for val in sublist]
        print("image count in {} path :{}".format(self.mode,len(self.image_paths_list)))
    def __getitem__(self, index):
        dist_x = 416
        dist_y = 352
        if self.mode == "test":
            image_list = self.image_paths_list[index]
            image = torch.tensor([]).to(device)
            for i, image_path in enumerate(image_list):
                i_image = Image.open(image_path).convert('RGB')
                image = torch.cat((image, test_preprocess_img(i_image)[1].to(device)), dim = 0)
                if i == 0:
                    o_image = np.array(test_preprocess_img(i_image)[0])
            image = image.view(-1, 3, dist_y, dist_x)
            return o_image, image_list[0], image
    def __len__(self):
        return len(self.image_paths_list)
    def its_continue_num(self):
        return self.continuous_frame_num
def get_continuous_loader(image_path, batch_size, mode, augmentation_prob, shffule_yn = False, continue_num = [1, 2, 3, 4, 5, 6, 7, 8]):
    dataset = Continuos_Image(root = image_path, prob = augmentation_prob,mode = mode, continuous_frame_num = continue_num)
    data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=shffule_yn,
                                  drop_last=True )
    return data_loader, dataset.its_continue_num()

if __name__ == "__main__":
    print(os.getcwd())
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    seed = 1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default="input_video")
    parser.add_argument('--output_img_path', type=str, default="output_frame")
    parser.add_argument('--model_path', type=str, default="model.pt")
    parser.add_argument('--output_path', type=str, default="output_prediction")
    parser.add_argument('--keep_image', type= int, default=0)
    parser.add_argument('--continuous', type=int, default=1)
    parser.add_argument('--distance', type=int, default=50)
    parser.add_argument('--interval_num', type=int, default=5)
    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                    dir_write_path = os.path.join( config.output_img_path,"/".join(video_path.split("/")[-3:-2]), "_".join(video_path.split("/")[-2:]).split(".")[0])
                    write_path = os.path.join(dir_write_path, "original")
                    if not os.path.isdir(write_path):
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
    with torch.no_grad():
        frame_continue_num = [-6, -3, 0, 3, 6]
        test_loader, continue_num = get_continuous_loader(image_path = config.output_img_path,
                                    batch_size = 1,
                                    mode = 'test',
                                    augmentation_prob = 0.,
                                    shffule_yn = False,
                                    continue_num = frame_continue_num)
        net = New_DeepLabV3Plus_LSTM(1, len(frame_continue_num), "resnet34")
        model_name = "New_DeepLabV3Plus_LSTM"+"_"+"resnet34"
        net.load_state_dict(torch.load("model.pt", map_location='cpu'))
        net = net.to(device)
        print("pretrain model loaded!")
        test_w_postprocess(config, test_loader, net)
    