import segmentation_models_pytorch as smp
import torch
from network.UnetLSTM import *
from network.Vgg_FCN8s import Single_vgg_FCN8s
from network.Unet3D  import UNet_3D_Seg
from network.new_Unet3d import New_UNet3d
def WHICH_MODEL(config, frame_continue_num):
    if config.which_model == 1:
        net = Single_vgg_FCN8s(1)
        model_name = "Single_vgg__FCN8s"
        print(model_name)
    elif config.which_model == 2:
        net = smp.Unet(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_Unet"+"_"+config.backbone
        print(model_name)
    elif config.which_model == 3:
        net = smp.UnetPlusPlus(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_UnetPlusPlus"+"_"+config.backbone
        print(model_name)
    elif config.which_model == 4:
        net = smp.PSPNet(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_PSPNet"+"_"+config.backbone
        print(model_name)
    elif config.which_model == 5:
        net = smp.Linknet (
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_Linknet "+"_"+config.backbone
        print(model_name)
    elif config.which_model == 6:
        net = smp.DeepLabV3Plus(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_DeepLabV3Plus"+"_"+config.backbone
        print(model_name)
    elif config.which_model == 18:
        net = Unet_LSTM(1, len(frame_continue_num), config.backbone)
        model_name = "Unet_LSTM"+"_"+config.backbone+"_TLOSS="+str(config.w_T_LOSS)
        print(model_name)
    elif config.which_model == 19:
        net = UnetPlusPlus_LSTM(1, len(frame_continue_num), config.backbone)
        model_name = "UnetPlusPlus_LSTM"+"_"+config.backbone
        print(model_name)
    elif config.which_model == 20:
        net = Linknet_LSTM(1, len(frame_continue_num), config.backbone)
        model_name = "Linknet_LSTM"+"_"+config.backbone
        print(model_name)
    elif config.which_model == 21:
        net = PSPNet_LSTM(1, len(frame_continue_num), config.backbone)
        model_name = "PSPNet_LSTM"+"_"+config.backbone
        print(model_name)
    elif config.which_model == 22:
        net = DeepLabV3Plus_LSTM(1, len(frame_continue_num), config.backbone)
        model_name = "DeepLabV3Plus_LSTM"+"_"+config.backbone
        print(model_name)
    elif config.which_model == -22:
        net = New_DeepLabV3Plus_LSTM(1, len(frame_continue_num), config.backbone)
        model_name = "New_DeepLabV3Plus_LSTM"+"_"+config.backbone
        print(model_name)
    elif config.which_model == -23:
        net = Temp_New_DeepLabV3Plus_LSTM(1, len(frame_continue_num), config.backbone)
        model_name = "Temp_New_DeepLabV3Plus_LSTM"+"_"+config.backbone
        print(model_name)
    elif config.which_model == 23:
        net = DeepLabV3_LSTM(1, len(frame_continue_num), config.backbone)
        model_name = "DeepLabV3_LSTM"+"_"+config.backbone
        print(model_name)
    elif config.which_model == -1:
        net = UNet_3D_Seg(1, Unet_3D_channel = config.Unet_3D_channel, continue_num = len(frame_continue_num))
        model_name = "3DUnet"
        print(model_name)
    elif config.which_model == -2:
        net = New_UNet3d(in_dim = 3, out_dim = 1, num_filters = config.Unet_3D_channel)
        model_name = "New_UNet3d"
        print(model_name)
    elif config.which_model == 0:
        print("No assign which model!")
    if config.model_path != "":
            net.load_state_dict(torch.load(config.model_path))
            print("pretrain model loaded!")
    return net, model_name