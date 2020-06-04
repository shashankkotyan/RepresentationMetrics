#!/usr/bin/env python

from networks.model import Model

class ImagenetModel(Model):

    def __init__(self, args):

        self.size = 224
        self.input_shape = (224, 224, 3)
        
        Model.__init__(self, args)

        self.weight_decay = 0.0001

    def dataset(self):

        import os, tensorflow as tf, numpy as np

        def parse_record(raw_record, is_training, exclusion=None):

            feature_map = {
                    'image/encoded':     tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
                    'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                    'image/class/text':  tf.io.FixedLenFeature([], dtype=tf.string,  default_value=''),
                    'image/filename':    tf.io.FixedLenFeature([], dtype=tf.string, default_value='')
            }

            sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
            feature_map.update({k: sparse_float32 for k in ['image/object/bbox/xmin', 'image/object/bbox/ymin', 'image/object/bbox/xmax', 'image/object/bbox/ymax']})

            features = tf.io.parse_single_example(serialized=raw_record, features=feature_map)
            bbox     = tf.transpose(a=tf.expand_dims(tf.concat([ tf.expand_dims(features['image/object/bbox/ymin'].values, 0), tf.expand_dims(features['image/object/bbox/xmin'].values, 0), tf.expand_dims(features['image/object/bbox/ymax'].values, 0), tf.expand_dims(features['image/object/bbox/xmax'].values, 0)], 0), 0), perm=[0, 2, 1])
            
            if is_training:

                bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
                    tf.image.extract_jpeg_shape(features['image/encoded']),
                    bounding_boxes=bbox,
                    min_object_covered=0.1,
                    aspect_ratio_range=[0.75, 1.33],
                    area_range=[0.05, 1.0],
                    max_attempts=100,
                    use_image_if_no_bounding_boxes=True)

                offset_y, offset_x, _          = tf.unstack(bbox_begin)
                target_height, target_width, _ = tf.unstack(bbox_size)

                image = tf.compat.v1.image.resize(tf.image.random_flip_left_right(tf.image.decode_and_crop_jpeg(features['image/encoded'], tf.stack([offset_y, offset_x, target_height, target_width]), channels=self.img_channels)), [self.img_rows, self.img_cols], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

            else:

                image = tf.image.decode_jpeg(features['image/encoded'], channels=self.img_channels)

                shape = tf.shape(input=image)
                height, width = shape[0], shape[1]

                height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
                scale_ratio   = tf.cast(256, tf.float32) / tf.minimum(height, width)

                image = tf.compat.v1.image.resize(image, [tf.cast(height * scale_ratio, tf.int32), tf.cast(width * scale_ratio, tf.int32)], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

                shape = tf.shape(input=image)
                height, width = shape[0], shape[1]

                image = tf.slice(image, [(height - self.img_rows) // 2, (width - self.img_cols) // 2, 0], [self.img_rows, self.img_cols, -1])
            
            image.set_shape([self.img_rows, self.img_cols, self.img_channels])

            label = tf.cast(features['image/class/label'], dtype=tf.int32) 
            
            if self.USE_DATASET == 0: label = label-1 

            if exclusion is not None:
                if label == exclusion:
                    label = -1
                else:                  
                    label = label if label < exclusion else label-1

            return tf.cast(features['image/filename'], dtype=tf.string), tf.cast(image, dtype=tf.float32), tf.cast(label,dtype=tf.int64)

        def defence_wrapper(image, label):
            image, label = self.defence(np.array([image]), np.array([label]))
            return image[0], label[0]

        def parser(filename, image, label, istrain, exclusion): 
            if self.raw_defence: 
                if not self.istrain_defence:
                    image, label = tf.numpy_function(defence_wrapper, [image, label], [tf.float32, tf.int64])
                else:
                    if istrain: image, label = tf.numpy_function(defence_wrapper, [image, label], [tf.float32, tf.int64])

                image.set_shape(self.input_shape)
                label.set_shape(())

            image = image - tf.broadcast_to(self.mean, tf.shape(image))
            label = tf.one_hot(label, self.num_classes)
            
            if self.processed_defence:
                image, label = tf.numpy_function(defence_wrapper, [image, label], [tf.float32, tf.float32])
                image.set_shape(self.input_shape)
                label.set_shape((self.num_classes,))
                
            if exclusion:
                return image
            else:
                return image, label
        
        def create_dataset(is_training, exclusion=None):                

            options = tf.data.Options()
            options.experimental_threading.max_intra_op_parallelism = 1
            
            if is_training:
                filenames = [os.path.join(f"{self.data_dir}/{self.dataset_name}_img_train/", 'train-%05d-of-01024' % i) for i in range(1024)]
            else:
                filenames = [os.path.join(f"{self.data_dir}/{self.dataset_name}_img_val/", 'val-%05d-of-00128' % i)   for i in range(128)]
        
            raw_dataset = tf.data.Dataset.from_tensor_slices(filenames)
            # raw_dataset = raw_dataset.shuffle(buffer_size= 1024 if is_training else 128)
            raw_dataset = raw_dataset.interleave(tf.data.TFRecordDataset, cycle_length= 12, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if is_training:
                raw_dataset = raw_dataset.shuffle(buffer_size=10000)
                raw_dataset = raw_dataset.repeat(count=self.epochs)
            
            raw_dataset = raw_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            raw_dataset = raw_dataset.map(lambda value: parse_record(value, is_training, exclusion), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            raw_dataset = raw_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            
            raw_excluded_dataset  = raw_dataset.filter(lambda filename, image, label: tf.equal(tf.convert_to_tensor(-1, dtype=tf.int64), label))
            
            raw_dataset       = raw_dataset.filter(lambda filename, image, label: tf.not_equal(tf.convert_to_tensor(-1, dtype=tf.int64), label))    
            raw_dataset       = raw_dataset.with_options(options)
            processed_dataset = raw_dataset.map(lambda filename, image, label: parser(filename, image, label, is_training, False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            processed_dataset = processed_dataset.batch(self.batch_size)
            # if not is_training: processed_dataset  = processed_dataset.cache()
            processed_dataset = processed_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            if exclusion is not None:
                raw_excluded_dataset       = raw_excluded_dataset.with_options(options)
                processed_excluded_dataset = raw_excluded_dataset.map(lambda filename, image, label: parser(filename, image, label, is_training, True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                processed_excluded_dataset = processed_excluded_dataset.batch(self.batch_size)
                # if not is_training: processed_excluded_dataset  = processed_excluded_dataset.cache()
                processed_excluded_dataset = processed_excluded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

                return raw_dataset, processed_dataset, raw_excluded_dataset, processed_excluded_dataset

            else:
                return raw_dataset, processed_dataset
            
        self.img_rows, self.img_cols, self.img_channels = self.size, self.size, 3
        self.data_dir = '/home/danilo/imagenet_files'

        self.mean = [123.68, 116.78, 103.94]
        self.std  = [1., 1., 1.]

        if self.USE_DATASET == 0:     
            
            self.dataset_name = 'Imagenet'
            self.num_images   = {'train': 1281167, 'test': 50000}
            self.num_classes  = 1000
            with open(f"{self.data_dir}/labels-imagenet.txt") as f: self.class_names = f.read().splitlines()

        elif self.USE_DATASET == 1:     

            self.dataset_name = 'RestrictedImagenet'
            self.num_images   = {'train': 129359, 'test': 5000}
            self.num_classes  = 10
            self.class_names  = ['Automobile', 'Ball', 'Bird', 'Dog', 'Feline', 'Fruit', 'Insect', 'Snake', 'Primate', 'Vegetable']
        
        if self.EXCLUDED_LABEL != 10:    
            self.label_name                = f"{self.class_names[self.EXCLUDED_LABEL]}"

            self.iterations_excluded_test  = ((self.num_images['test']//self.num_classes) // self.batch_size) + 1   
            self.num_images['test']       -= ( self.num_images['test']//self.num_classes)
            
            # self.num_images['train'] -= (self.num_images['test']//self.num_classes) # 12935

            if self.USE_DATASET == 1:
                self.num_images_class          = [12981, 12971, 12990, 12904, 13000, 12986, 12985, 12758, 12979, 12815]
                self.iterations_excluded_train = (self.num_images_class[self.EXCLUDED_LABEL] // self.batch_size) + 1   
                self.num_images['train']      -= self.num_images_class[self.EXCLUDED_LABEL]

            self.num_classes -= 1

            self.raw_train_dataset, self.processed_train_dataset, self.raw_excluded_train_dataset, self.processed_excluded_train_dataset = create_dataset(True  ,self.EXCLUDED_LABEL)
            self.raw_test_dataset,  self.processed_test_dataset , self.raw_excluded_test_dataset,  self.processed_excluded_test_dataset  = create_dataset(False ,self.EXCLUDED_LABEL)
            
        else:

            self.raw_train_dataset, self.processed_train_dataset, = create_dataset(True)
            self.raw_test_dataset,  self.processed_test_dataset , = create_dataset(False)
    
        self.iterations_train = (self.num_images['train'] // self.batch_size) + 1   
        self.iterations_test  = (self.num_images['test']  // self.batch_size) + 1   


    def build_model(self):

        from tensorflow.keras import initializers, layers, models, optimizers, regularizers

        base_model = self.model_class(weights=None, include_top=False, input_shape=(self.img_rows, self.img_cols, self.img_channels))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(self.num_classes, name='Output', activation='softmax', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        
        final_model = models.Model(inputs=base_model.input, outputs=x)
        for layer in final_model.layers[:]: layer.trainable = True

        final_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')

        return final_model

    def fit_normal(self):
        """
        TODO: Write Comment
        """

        history = self._model.fit(
                self.processed_train_dataset, steps_per_epoch=self.iterations_train, 
                epochs=self.epochs, callbacks=self.cbks, verbose=1,
                # workers=12, use_multiprocessing=True, 
                validation_data=self.processed_test_dataset, validation_steps=self.iterations_test
            )

        return history.history

    def get(self, samples):
        """
        TODO: Write Comment
        """
    
        indices, xs, ys = [], [], []
        count = 0

        for index, x, y in self.raw_test_dataset.take(self.num_images['test']):
            index = index.numpy()
            x = x.numpy()
            y = y.numpy()
            # if np.argmax(self.predict(x)) == y: 
            #    load = False
            if True:
                indices.append(index)
                xs.append(x)
                ys.append(y)
                count +=1
            if count == samples: break
        return indices, xs, ys

    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        if epoch < 30: return 0.1
        if epoch < 60: return 0.01
        if epoch < 80: return 0.001
        return 0.0001
