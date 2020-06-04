#!/usr/bin/env python


from networks.model import Model


class MnistModel(Model):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """
        self.input_shape = (28, 28, 1)

        Model.__init__(self, args)

    def dataset(self):
        """
        TODO: Write Comment
        """
        
        from tensorflow.keras import datasets, utils
        import numpy as np
        
        self.num_images = {'train': 60000, 'test': 10000}

        self.mean = [0.]
        self.std  = [255.]

        if self.USE_DATASET == 0:
 
            self.num_classes  = 10
            self.dataset_name = 'Mnist'
            self.class_names = [
                                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
                               ]
            self.num_images_class_train = [ 5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
            self.num_images_class_test =  [ 980,  1135, 1032, 1010, 982,  892,  958,  1028, 974,  1009]

            __datasets = datasets.mnist

        elif self.USE_DATASET == 1:
            
            self.num_classes  = 10
            self.dataset_name = 'FashionMnist'
            self.class_names  = [
                                 'T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot'
                                ]
            self.num_images_class_train = [6000]*self.num_classes
            self.num_images_class_test =  [1000]*self.num_classes

            __datasets = datasets.fashion_mnist
            
        (self.raw_x_train, self.raw_y_train), (self.raw_x_test, self.raw_y_test) = __datasets.load_data()
        self.raw_x_train, self.raw_x_test = self.raw_x_train.reshape(-1,28,28,1), self.raw_x_test.reshape(-1,28,28,1)
        
        if self.EXCLUDED_LABEL != 10:
            self.label_name  = f"{self.class_names[self.EXCLUDED_LABEL]}"

            self.num_images['train'] -= self.num_images_class_train[self.EXCLUDED_LABEL]
            self.num_images['test']  -= self.num_images_class_test[self.EXCLUDED_LABEL]

            self.num_classes -= 1
            
            def form_subset(xs, ys):

                included_x, included_y, excluded_x = [], [], []
            
                for x, y in zip(xs, ys):
                    if y == self.EXCLUDED_LABEL: 
                        excluded_x += [x] 
                    else:
                       included_x += [x] 
                       included_y += [y if y<self.EXCLUDED_LABEL else y-1]  

                return np.array(included_x), np.array(included_y), np.array(excluded_x)

            self.raw_x_train, self.raw_y_train, self.raw_excluded_x_train = form_subset(self.raw_x_train, self.raw_y_train)
            self.raw_x_test,  self.raw_y_test,  self.raw_excluded_x_test  = form_subset(self.raw_x_test,  self.raw_y_test)

            self.iterations_excluded_train = (self.num_images_class_train[self.EXCLUDED_LABEL] // self.batch_size) + 1   
            self.iterations_excluded_test  = (self.num_images_class_test[self.EXCLUDED_LABEL] // self.batch_size) + 1       

            self.processed_excluded_x_train, self.processed_excluded_x_test = self.color_preprocess(self.raw_excluded_x_train, True), self.color_preprocess(self.raw_excluded_x_test, False)
        
        self.processed_x_train, self.processed_x_test = self.color_preprocess(self.raw_x_train, True),          self.color_preprocess(self.raw_x_test, False)
        self.processed_y_train, self.processed_y_test = utils.to_categorical(self.raw_y_train, self.num_classes), utils.to_categorical(self.raw_y_test, self.num_classes)        

        if self.label_defence:
            self.processed_x_train, self.processed_y_train = self.defence(self.processed_x_train, self.processed_y_train)
            self.processed_x_test, self.processed_y_test   = self.defence(self.processed_x_test, self.processed_y_test)

        self.iterations_train = (self.num_images['train'] // self.batch_size) + 1   
        self.iterations_test  = (self.num_images['test']  // self.batch_size) + 1  
        