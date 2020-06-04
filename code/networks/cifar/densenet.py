#!/usr/bin/env python

from networks.cifar.cifar_model import CifarModel


class DenseNet(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name         = 'DenseNet'

        CifarModel.__init__(self, args)
    
    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers

        growth_rate  = 12 
        depth        = 100
        compression  = 0.5
        weight_decay = 0.0001

        nblocks   = (depth - 4) // 6 
        nchannels = growth_rate * 2

        def bn_relu(x, name=None):
            """
            TODO: Write Comment
            """

            x = layers.BatchNormalization()(x)

            if name is None: x = layers.Activation('relu')(x)
            else:            x = layers.Activation('relu', name=name)(x)

            return x

        def bottleneck(x):
            """
            TODO: Write Comment
            """

            channels = growth_rate * 4

            x = bn_relu(x)
            x = layers.Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
            
            x = bn_relu(x)
            x = layers.Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
            
            return x

        def single(x):
            """
            TODO: Write Comment
            """
            
            x = bn_relu(x)
            x = layers.Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
            
            return x

        def transition(x, inchannels):
            """
            TODO: Write Comment
            """
            
            outchannels = int(inchannels * compression)
            
            x = bn_relu(x)
            x = layers.Conv2D(outchannels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
            x = layers.AveragePooling2D((2,2), strides=(2, 2))(x)
            
            return x, outchannels

        def dense_block(x,blocks,nchannels):
            """
            TODO: Write Comment
            """
            
            concat = x
            
            for i in range(blocks):
                
                x = bottleneck(concat)
                concat = layers.concatenate([x,concat], axis=-1)
                
                nchannels += growth_rate

            return concat, nchannels
        
        x = layers.Conv2D(nchannels,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(img_input)
        
        x, nchannels = dense_block(x,nblocks,nchannels)
        x, nchannels = transition(x,nchannels)

        x, nchannels = dense_block(x,nblocks,nchannels)
        x, nchannels = transition(x,nchannels)

        x, nchannels = dense_block(x,nblocks,nchannels)
        x, nchannels = transition(x,nchannels)

        x = bn_relu(x, 'Penultimate')

        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(self.num_classes , name='Output', activation='softmax', kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)
        
        return x
        
    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        if epoch <= 75:  
            return 0.1
        if epoch <= 150: 
            return 0.01
        return 0.001