#!/usr/bin/env python

from networks.cifar.cifar_model import CifarModel


class NetworkInNetwork(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name = 'NIN'
        
        CifarModel.__init__(self, args)

    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers

        weight_decay = 0.0001
        dropout      = 0.5

        x = layers.Conv2D(192, (5, 5), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal())(img_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(160, (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(96, (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same')(x)

        x = layers.Dropout(dropout)(x)

        x = layers.Conv2D(192, (5, 5), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(192, (1, 1),padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(192, (1, 1),padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same')(x)

        x = layers.Dropout(dropout)(x)

        x = layers.Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(192, (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(self.num_classes , (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)

        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Activation('softmax', name='Output')(x)

        return x

    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        if epoch <= 60:  
            return 0.05
        if epoch <= 120: 
            return 0.01
        if epoch <= 160: 
            return 0.002
        return 0.0004

    