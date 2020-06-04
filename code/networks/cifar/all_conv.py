#!/usr/bin/env python

from networks.cifar.cifar_model import CifarModel

class AllConv(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """
        from tensorflow.keras import optimizers

        self.name         = 'AllConv'

        CifarModel.__init__(self, args)

        self.optimizer = optimizers.Adam(lr=1.0e-4)

    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers

        weight_decay = 0.0001

        x = layers.Conv2D(96, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding = 'same')(img_input)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(96, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding = 'same')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(96, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding = 'same', strides = 2)(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(192, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding = 'same')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(192, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding = 'same')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(192, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding = 'same', strides = 2)(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(192, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding = 'same')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(192, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding = 'same')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(192, (1, 1), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding='valid')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(self.num_classes, (1, 1), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializers.he_normal(), padding='valid')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Activation('softmax', name='Output')(x)
        
        return x
    
    def scheduler(self, epoch): 
        """
        TODO: Write Comment
        """

        return 1.0e-4
