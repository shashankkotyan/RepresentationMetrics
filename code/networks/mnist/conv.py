#!/usr/bin/env python

from networks.mnist.mnist_model import MnistModel
     

class Conv(MnistModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name         = 'Convolutional'

        self.learn_rate   = 0.001
        
        MnistModel.__init__(self, args)

    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers

        weight_decay = 0.0001

        x = layers.Conv2D(32, (5,5), padding='valid', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(img_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (5,5), padding='valid', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2), strides=(2,2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(64, (3,3), padding='valid', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3,3), padding='valid', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2), strides=(2,2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(256, kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(self.num_classes, activation = "softmax", name='Output')(x)

        return x


    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        return self.learn_rate