#!/usr/bin/env python

from networks.cifar.cifar_model import CifarModel


class ResNet(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name = 'ResNet'

        CifarModel.__init__(self, args)

    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers

        stack_n      = 5    
        weight_decay = 0.0001

        def residual_block(img_input, out_channel,increase=False):
            """
            TODO: Write Comment
            """

            if increase: 
                stride = (2,2)
            else: 
                stride = (1,1)

            x = img_input

            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            x = layers.Conv2D(out_channel,kernel_size=(3,3),strides=stride,padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            x = layers.Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
            
            if increase:

                projection = layers.Conv2D(out_channel, kernel_size=(1,1), strides=(2,2), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(img_input)
                
                return layers.add([x, projection])

            else: 
                return layers.add([img_input, x])
            
        x = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(img_input)

        for _ in range(stack_n):    
        	x = residual_block(x, 16, False)
        x = residual_block(x, 32, True)
        
        for _ in range(1, stack_n): 
        	x = residual_block(x, 32, False)
        x = residual_block(x, 64, True)
        
        for _ in range(1, stack_n): 
        	x = residual_block(x, 64, False)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)

        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(self.num_classes, name='Output', activation='softmax', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        
        return x

    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        if epoch < 80: 
            return 0.1
        if epoch < 150: 
            return 0.01
        return 0.001