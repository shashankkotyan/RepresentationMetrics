#!/usr/bin/env python

from networks.imagenet.imagenet_model import ImagenetModel

from tensorflow.keras import applications


class ResNet50(ImagenetModel):
    def __init__(self, args):
        self.name = f"ResNet-50"
        self.model_class = applications.ResNet50
        ImagenetModel.__init__(self, args)

class ResNet101(ImagenetModel):
    def __init__(self, args):
        self.name = f"ResNet-101"
        self.model_class = applications.ResNet101
        ImagenetModel.__init__(self, args)

class ResNet152(ImagenetModel):
    def __init__(self, args):
        self.name = f"ResNet-152"
        self.model_class = applications.ResNet152
        ImagenetModel.__init__(self, args)

class ResNetV250(ImagenetModel):
    def __init__(self, args):
        self.name = f"ResNetV2-50"
        self.model_class = applications.ResNet50V2
        ImagenetModel.__init__(self, args)

class ResNetV2101(ImagenetModel):
    def __init__(self, args):
        self.name = f"ResNetV2-101"
        self.model_class = applications.ResNet101V2
        ImagenetModel.__init__(self, args)

class ResNetV2152(ImagenetModel):
    def __init__(self, args):
        self.name = f"ResNetV2-152"
        self.model_class = applications.ResNet152V2
        ImagenetModel.__init__(self, args)


class InceptionV3(ImagenetModel):
    def __init__(self, args):
        self.name = 'InceptionV3'
        self.model_class = applications.InceptionV3
        ImagenetModel.__init__(self, args)


class InceptionResNetV2(ImagenetModel):
    def __init__(self, args):
        self.name = 'InceptionResnetV2'
        self.model_class = applications.InceptionResNetV2
        ImagenetModel.__init__(self, args)


class Xception(ImagenetModel):
    def __init__(self, args):
        self.name = 'Xception'
        self.model_class = applications.Xception
        ImagenetModel.__init__(self, args)


class DenseNet121(ImagenetModel):
    def __init__(self, args):
        self.name = f"DenseNet-121"
        self.model_class = applications.DenseNet121
        ImagenetModel.__init__(self, args) 

class DenseNet169(ImagenetModel):
    def __init__(self, args):
        self.name = f"DenseNet-169"
        self.model_class = applications.DenseNet169
        ImagenetModel.__init__(self, args)    

class DenseNet201(ImagenetModel):
    def __init__(self, args):
        self.name = f"DenseNet-201"
        self.model_class = applications.DenseNet201
        ImagenetModel.__init__(self, args)   


class VGG16(ImagenetModel):
    def __init__(self, args):
        self.name = 'VGG-16'
        self.model_class = applications.VGG16
        ImagenetModel.__init__(self, args)

class VGG_19(ImagenetModel):
    def __init__(self, args):
        self.name = 'VGG-19'
        self.model_class = applications.VGG19
        ImagenetModel.__init__(self, args)


class MobileNet(ImagenetModel):
    def __init__(self, args):
        self.name = 'MobileNet'
        self.model_class = applications.MobileNet
        ImagenetModel.__init__(self, args)

class MobileNetV2(ImagenetModel):
    def __init__(self, args):
        self.name = 'MobileNetV2'
        self.model_class = applications.MobileNetV2
        ImagenetModel.__init__(self, args) 


class NasNet(ImagenetModel):
    def __init__(self, args):
        self.name = f"Nasnet-{self.value}"
        ImagenetModel.__init__(self, args)

class NASNetMobile(ImagenetModel):
    def __init__(self, args):
        self.name = f"NasNet-Mobile"
        self.model_class = applications.NASNetMobile
        ImagenetModel.__init__(self, args)

class NASNetLarge(ImagenetModel):
    def __init__(self, args):
        self.name = f"NasNet-Large"
        self.model_class = applications.NASNetLarge
        ImagenetModel.__init__(self, args)