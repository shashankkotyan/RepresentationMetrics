#!/usr/bin/env python

from networks.cifar.cifar_model import CifarModel


class LeNet(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name = 'LeNet'

        CifarModel.__init__(self, args)
        
    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers
        
        weight_decay = 0.0001

        x = layers.Conv2D(6, (5, 5), padding='valid', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(img_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        x = layers.Conv2D(16, (5, 5), padding='valid', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        x = layers.Flatten()(x)
        
        x = layers.Dense(120, kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay) )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(84, kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay) )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu',  name='Penultimate')(x)

        x = layers.Dense(self.num_classes, name='Output', activation = 'softmax', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay) )(x)
        
        return x

    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        if epoch < 100:
            return 0.01
        if epoch < 150:
            return 0.005
        return 0.001
