#!/usr/bin/env python

from networks.cifar.cifar_model import CifarModel

import tensorflow as tf, numpy as np
from tensorflow.keras import layers, backend as K

class CapsNet(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import optimizers

        self.name       = 'CapsNet'
        self.num_routes = 3

        CifarModel.__init__(self, args)

        self.optimizer = optimizers.Adam(lr=0.001)
        
    
    def network(self, img_input):
        """
        TODO: Write Comment
        """

        dim_capsule  = 8
        
        x = layers.Conv2D(filters=256, kernel_size=8, strides=1, padding='valid', name='conv2d')(img_input)
        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=dim_capsule*32, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2d')(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Reshape(target_shape=(2592, dim_capsule), name='primarycap_reshape')(x)
        x = layers.Lambda(squash, name='primarycap_squash')(x)

        x = CapsuleLayer(num_capsule=self.num_classes, dim_capsule=dim_capsule*2, routings=self.num_routes, name='digitcaps')(x)
        self.digitcaps = x

        x = Length(name='output')(x)

        return x

    def load(self):
        """
        TODO: Write Comment
        """

        if self.args.verbose: print(f"Loading Model...")
        
        self._model = self.build_model()

        try:
            self._model.load_weights(f"{self.log_filepath}model_weights.h5")
        except:
            self.train()
            
        # utils.plot_model(self._model, show_shapes=True, to_file=f"{self.log_filepath}model.png")
        # self._model.summary()

    def train(self):
        """
        TODO: Write Comment
        """
        
        self.train_model = self.build_train_model()
        self.train_model.compile(optimizer= self.optimizer, loss=[margin_loss, 'mse'], loss_weights=[1., 0.1], metrics={'out_recon':'accuracy', 'output':'accuracy'})

        history = self.fit_normal()
        
        self.save(history)

    def build_train_model(self): 
        """
        TODO: Write Comment
        """

        from tensorflow.keras import models 

        y       = layers.Input(shape=(self.num_classes,))
        masked  = Mask()([self.digitcaps, y])  
        x_recon = layers.Dense(512)(masked)
        x_recon = layers.Activation('relu')(x_recon)
        x_recon = layers.Dense(1024)(x_recon)
        x_recon = layers.Activation('relu')(x_recon)
        x_recon = layers.Dense(np.prod(self.input_shape))(x_recon)
        x_recon = layers.Activation('sigmoid')(x_recon)
        x_recon = layers.Reshape(target_shape=self.input_shape, name='out_recon')(x_recon)

        return models.Model([self._model.input, y], [self._model.output, x_recon])

    def fit_model(self, x_train, y_train, x_test, y_test, batch_size, epochs, iterations, cbks, verbose):
        """
        TODO: Write Comment
        """
        
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        epochs = 50

        if self.augmentation:

            datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant',cval=0.)
            datagen.fit(x_train)

            generator = datagen.flow(x_train, y_train, batch_size=batch_size)

            def generate():
                while True:
                    x,y  = generator.next()
                    yield ([x,y],[y,x])

            history = self.train_model.fit_generator(
                generate(),
                steps_per_epoch=iterations, epochs=epochs, callbacks=cbks, verbose=verbose,
                # workers=12, use_multiprocessing=True,
                validation_data=([x_test, y_test], [y_test, x_test])
            )
        else:
            
            history = self.train_model.fit(
                [x_train, y_train], [y_train, x_train], batch_size=batch_size, 
                epochs=epochs, callbacks=cbks, verbose=verbose,
                # workers=12, use_multiprocessing=True,
                validation_data=([x_test, y_test], [y_test, x_test])
            )
        
        return history.history

    
    def scheduler(self, epoch): 
        """
        TODO: Write Comment
        """

        return 0.001 * np.exp(-epoch / 10.) 
        #return 0.001 * (0.95 ** epoch)

def caps_batch_dot(x, y, axes=None):
    """
    TODO: Write Comment
    """

    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = K.ndim(x)
    y_ndim = K.ndim(y)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x_ndim - 1, y_ndim - 2]
    if any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if K.ndim(x) == 2 and K.ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == K.ndim(x) - 1 else True
            adj_y = True if axes[1] == K.ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if K.ndim(out) == 1:
        out = expand_dims(out, 1)
    return out
    

def margin_loss(y_true, y_pred): 
    """
    TODO: Write Comment
    """
    # L = y_true * K.square(K.relu(0.9 - y_pred))        + 0.5 * (1 - y_true) * K.square(K.relu( y_pred - 0.1))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, axis=-1))


class Length(layers.Layer):
    """
    TODO: Write Comment
    """
    
    def call(self, inputs, **kwargs):
        """
        TODO: Write Comment
        """   
        # return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))         
        return K.sqrt(tf.reduce_sum(K.square(inputs), -1) + K.epsilon())
    
    def compute_output_shape(self, input_shape): 
        return input_shape[:-1]
    
    def get_config(self):                        
        return super(Length, self).get_config()


class Mask(layers.Layer):
    """
    TODO: Write Comment
    """

    def call(self, inputs, **kwargs):
        """
        TODO: Write Comment
        """
        if type(inputs) is list:  

            assert len(inputs) == 2
            inputs, mask = inputs

        else:  

            # mask =  K.one_hot(indices=K.sqrt(tf.reduce_sum(K.square(inputs), -1)), num_classes=self.num_classes)
            mask = tf.clip_by_value((inputs - tf.reduce_max(inputs, 1, True)) / K.epsilon() + 1, 0, 1)  

        # x = inputs * tf.expand_dims(mask, -1)
        # output = tf.reshape(x, tf.stack([-1, K.prod(x.shape[1:])]))
        output = caps_batch_dot(inputs, mask, [1, 1])
        return output

    def compute_output_shape(self, input_shape):

        if type(input_shape[0]) is tuple:  
            return tuple([None, input_shape[0][2]])
        else:    
            return tuple([None, input_shape[2]])
            
    def get_config(self): 
        return super(Mask, self).get_config()

def squash(vectors, axis=-1):
    """
    TODO: Write Comment
    """

    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    TODO: Write Comment
    """

    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=False, activation='squash', kernel_initializer='glorot_uniform', **kwargs):

        from tensorflow.keras import initializers

        super(CapsuleLayer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings    = routings
        
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        self.kernel = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_capsule, self.dim_capsule], initializer=self.kernel_initializer, name='capsule_kernel')
        self.bias   = tf.Variable(initial_value=tf.zeros(([1, self.input_num_capsule, self.num_capsule, 1, 1])), trainable=False)
        
        self.built = True

    def call(self, inputs, training=None):
        
        inputs_hat = tf.map_fn(lambda x: caps_batch_dot(x, self.kernel, [3, 2]),
                             elems=tf.tile(tf.expand_dims(tf.expand_dims(inputs, 2), 2), [1, 1, self.num_capsule, 1, 1]),
                             )

        assert self.routings > 0, 'The num_routing should be > 0.'
        
        for i in range(self.routings):
                        
            outputs = self.activation(tf.reduce_sum(tf.nn.softmax(self.bias, axis=2) * inputs_hat, axis=1, keepdims=True))

            if i != self.routings - 1:
                increment = tf.reduce_sum(inputs_hat * outputs, [0,-1], keepdims=True)
                self.bias.assign_add(increment)
                
        outputs = tf.reshape(outputs, [-1, self.num_capsule, self.dim_capsule])
        return outputs

    
    def compute_output_shape(self, input_shape): 
        return tuple([None, self.num_capsule, self.dim_capsule])
    
    def get_config(self): 
        return dict(list(super(CapsuleLayer, self).get_config().items()) + list( {'num_capsule': self.num_capsule, 'dim_capsule': self.dim_capsule, 'routings': self.routings}.items()))
