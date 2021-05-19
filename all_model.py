import segmentation_models_pytorch as smp
import torch
from network.Vgg_FCN8s import Single_vgg_FCN8s
from network.Vgg_Unet import Single_vgg_Unet
from network.Res_Unet import Single_Res_Unet
from network.Nested_Unet import Single_Nested_Unet
from network.DeepLab import DeepLab
from network.Unet3D import UNet_3D_Seg
from network.Two_Level_Net import Two_Level_Nested_Unet, Two_Level_Res_Unet, Two_Level_Deeplab, Two_Level_Res_Unet_with_backbone, Unet_LSTM
def WHICH_MODEL(config, frame_continue_num):
    if config.which_model == 1:
        net = Single_vgg_FCN8s(1)
        model_name = "Single_vgg__FCN8s"
        print("Model Single_vgg__FCN8s")
    elif config.which_model == 2:
        net = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_Unet"
        print("smp.Unet")
    elif config.which_model == 3:
        net = smp.UnetPlusPlus(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_UnetPlusPlus"
        print("smp.UnetPlusPlus")
    elif config.which_model == 4:
        net = smp.PSPNet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_PSPNet"
        print("smp_PSPNet")
    elif config.which_model == 5:
        net = smp.MAnet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_ MAnet"
        print("smp_ MAnet")
    elif config.which_model == 6:
        net = smp.DeepLabV3Plus(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "smp_DeepLabV3Plus"
        print("smp_DeepLabV3Plus")
    elif config.which_model == 11:
        net = Two_Level_Res_Unet(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Res_Unet"
        print("Model Two_Level_Res_Unet")
    elif config.which_model == 12:
        net = Two_Level_Nested_Unet(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Nested_Unet"
        print("Model Two_Level_Nested_Unet")
    elif config.which_model == 13:
        net = UNet_3D_Seg(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "UNet_3D_Seg"
        print("Model UNet_3D_Seg")
    elif config.which_model == 14:
        net = Two_Level_Deeplab(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Deeplab"
        print("Two_Level_Deeplab")
    elif config.which_model == 15:
        net = Two_Level_Res_Unet_with_backbone(1, config.Unet_3D_channel, len(frame_continue_num))
        model_name = "Two_Level_Res_Unet_with_backbone"
        print("Two_Level_Res_Unet_with_backbone")        
    elif config.which_model == 18:
        net = Unet_LSTM(1, len(frame_continue_num))
        model_name = "Unet_LSTM"
        print("Unet_LSTM")
    elif config.which_model == -1:
        net = _Temporal_Module(1, config.Unet_3D_channel)
        model_name = "_Temporal_Module"
        print("_Temporal_Module")
    elif config.which_model == 0:
        print("No assign which model!")
    if config.model_path != "":
            net.load_state_dict(torch.load(config.model_path))
            print("pretrain model loaded!")
    return net, model_name