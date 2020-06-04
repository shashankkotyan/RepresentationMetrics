#!/usr/bin/env python

from networks.mnist.mnist_model import MnistModel
     

class MLP(MnistModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name         = 'MLP'

        self.learn_rate   = 0.001

        MnistModel.__init__(self, args)

    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers

        x = layers.Flatten()(img_input)

        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)

        x = layers.Dense(self.num_classes, activation = "softmax", name='Output')(x)

        return x


    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        return self.learn_rate