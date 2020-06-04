#!/usr/bin/env python

from networks.cifar.cifar_model import CifarModel


class WideResNet(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name         = 'WideResNet'
        
        CifarModel.__init__(self, args)

    def network(self, img_input):
        """
        TODO: Write Comment
        """
        
        from tensorflow.keras import initializers, layers, regularizers

        depth        = 16
        wide         = 8
        weight_decay = 0.0005

        def conv3x3(x,filters):
            """
            TODO: Write Comment
            """

            return layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)

        def residual_block(img_input, out_filters, increase_filter=False):
            """
            TODO: Write Comment
            """

            if increase_filter: 
                first_stride = (2,2)
            else:               
                first_stride = (1,1)

            x = img_input

            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            x = layers.Conv2D(out_filters, kernel_size=(3,3), strides=first_stride, padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            x = layers.Conv2D(out_filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
            
            if increase_filter or out_filters != 16:

                projection = layers.Conv2D(out_filters, kernel_size=(1,1), strides=first_stride, padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(img_input)
                block = layers.add([x, projection])

            else: block = layers.add([x, img_input])

            return block

        def wide_residual_layer(x,out_filters,increase_filter=False):
            """
            TODO: Write Comment
            """

            x = residual_block(x, out_filters, increase_filter)

            in_filters = out_filters

            for _ in range(1, (depth - 4) // 6): x = residual_block(x, out_filters)

            return x

        x = conv3x3(img_input,     16)
        x = wide_residual_layer(x, 16*wide)
        x = wide_residual_layer(x, 32*wide, increase_filter=True)
        x = wide_residual_layer(x, 64*wide, increase_filter=True)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)

        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(self.num_classes, name='Output', activation='softmax',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)
        
        return x

    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        if epoch <= 60:  
            return 0.1
        if epoch <= 120: 
            return 0.02
        if epoch <= 160: 
            return 0.004
        return 0.0008

