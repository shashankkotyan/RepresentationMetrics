#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

import os, sys, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')

import tensorflow as tf, numpy as np

def set_tensorflow_config(g_index=0):

    tf.get_logger().setLevel("ERROR")

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:

        try:
            if len(gpus) == 1:
                gpu_index = 0
            else:
                gpu_index = g_index

            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        except RuntimeError as e: print(e)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Representation Metrics'))
        
    parser.add_argument('-f','--family_dataset',        type=int, default=1, choices=[0,1,2],                    help='Family of the Dataset to be used')
    parser.add_argument('-d','--use_dataset',           type=int, default=0, choices=[0,1],                      help='Dataset to be used')
    
    parser.add_argument('-m','--model',                 type=int, default=3,                                     help='Model to be used')
    parser.add_argument('-xl','--excluded_label',       type=int, default=10, choices=[10,0,1,2,3,4,5,6,7,8,9],  help='Label Index to be exlcuded from training')
    
    parser.add_argument('--epochs',                     type=int, default=200,                                   help='Number of epochs Model Needs To be trained, if weight doesnt exist')
    parser.add_argument('--batch_size',                 type=int, default=64,                                    help='Batch Size')
    parser.add_argument('--augmentation',               action='store_true',                                     help='Use Augmentation in training networks')
    
    parser.add_argument('--custom_name',                type=str, default='custom',                              help='Name of custom model')
    
    parser.add_argument('--defence',                    type=int, default=0,                                     help='Defence to be used')
    parser.add_argument('--defence_range',              type=int, default=1, choices=[1, 255],                   help='Pixel Range for Defences')
    
    parser.add_argument('--gpu_index',                  type=int, default=0,                                     help='GPU to be used')
    parser.add_argument('-v','--verbose',               action="store_true",                                     help='Verbosity')
    parser.add_argument('--test',                       action="store_true",                                     help='Dry Run Attacks')

    
    args = parser.parse_args()
    print(args)

    if args.family_dataset == 0:
            
        from networks import mnist

        if args.model == 0:   model = mnist.mlp.MLP(args)
        elif args.model == 1: model = mnist.conv.Conv(args)

    elif args.family_dataset == 1:
        
        from networks import cifar

        if args.model == 0:   model = cifar.lenet.LeNet(args)
        elif args.model == 1: model = cifar.all_conv.AllConv(args)
        elif args.model == 2: model = cifar.network_in_network.NetworkInNetwork(args)
        elif args.model == 3: model = cifar.resnet.ResNet(args)
        elif args.model == 4: model = cifar.densenet.DenseNet(args)
        elif args.model == 5: model = cifar.wide_resnet.WideResNet(args)
        elif args.model == 6: model = cifar.vgg.VGG16(args)
        elif args.model == 7: model = cifar.capsnet.CapsNet(args)
        elif args.model == 8: model = cifar.vgg.VGG19(args)
        
    elif args.family_dataset == 2:

        from networks import imagenet

        if args.model == 0:    model = imagenet.keras_applications.InceptionV3(args)
        elif args.model == 1:  model = imagenet.keras_applications.InceptionResNetV2(args)
        elif args.model == 2:  model = imagenet.keras_applications.Xception(args)
        elif args.model == 3:  model = imagenet.keras_applications.ResNet50(args)
        elif args.model == 4:  model = imagenet.keras_applications.ResNet101(args)
        elif args.model == 5:  model = imagenet.keras_applications.Resnet152(args)
        elif args.model == 6:  model = imagenet.keras_applications.ResnetV250(args)
        elif args.model == 7:  model = imagenet.keras_applications.ResNetV2101(args)
        elif args.model == 8:  model = imagenet.keras_applications.ResnetV2152(args)
        elif args.model == 9:  model = imagenet.keras_applications.DenseNet121(args)
        elif args.model == 10: model = imagenet.keras_applications.DenseNet169(args)
        elif args.model == 11: model = imagenet.keras_applications.DenseNet201(args)
        elif args.model == 12: model = imagenet.keras_applications.MobileNet(args)
        elif args.model == 13: model = imagenet.keras_applications.MobileNetV2(args)
        elif args.model == 14: model = imagenet.keras_applications.NASNetMobile(args)
        elif args.model == 15: model = imagenet.keras_applications.NASNetLarge(args)
        elif args.model == 16: model = imagenet.keras_applications.VGG16(args)
        elif args.model == 17: model = imagenet.keras_applications.VGG19(args)

    model.load()
