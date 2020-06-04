#!/usr/bin/env python

from networks.cifar.cifar_model import CifarModel


class VGG(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        CifarModel.__init__(self, args)

    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers

        weight_decay = 0.0005

        if self.vgg_type == 16:
            stack = [2,2,3,3,3]
        else:
            stack = [2,2,4,4,4]

        filters = [64,128,256,512,512]

        def block(input, filters, n_stack):
            """
            TODO: Write Comment
            """
            
            x = input

            for i in range(n_stack):

                x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)

                if i < n_stack - 1:
                    x = layers.Dropout(0.4)(x)
                    
            x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

            return x

        x = img_input

        for i in range(len(filters)):
            x = block(x, filters[i], stack[i])
        
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(512, kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.num_classes, activation = 'softmax', name='Output')(x)

        return x

    def scheduler(self, epoch): 
        """
        TODO: Write Comment
        """
        if epoch < 80:
            return 0.1
        if epoch < 160:
            return 0.01
        return 0.001
        # return 0.1 * (0.5 ** (epoch // 20))

class VGG16(VGG):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name = 'VGG-16'
        self.vgg_type = 16

        VGG.__init__(self, args)

class VGG19(VGG):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name = 'VGG-19'
        self.vgg_type = 19
        
        VGG.__init__(self, args)