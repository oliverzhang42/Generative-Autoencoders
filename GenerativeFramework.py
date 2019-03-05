from autoencoder import Autoencoder
from convolutional_autoencoder import ConvolutionalAutoencoder
from keras.datasets import mnist, fashion_mnist

import os

class GenerativeFramework():
    def __init__(self, path, dataset='mnist', autoencoder='dense', dim=10, generator='ot transport', pruning=False):
        assert os.path.isdir(path), 'The given path isnt a directory!'
        assert dataset is in ['mnist', 'fashion-mnist', 'lfw', 'cifar10'], 'The dataset is not recognized!'
        assert autoencoder is in ['dense', 'conv', 'cond'], 'The autoencoder is not recognized!'
        assert generator is in ['ot transport', 'ot generator', 'st transport', 'kde', 'k-means'], 'The generator is not recognized!'

        if dataset == 'mnist':
            shape = (28, 28, 1)
			(x_train, y_train), (x_test, y_test) = mnist.load_data()
			x_train = x_train.astype('float32') / 255.
			x_test = x_test.astype('float32') / 255.

			self.x_train = x_train.reshape(60000, 28, 28, 1)
			self.y_train = y_train.reshape(60000, 1)
			self.y_train_onehot = keras.utils.to_categorical(y_train, num_classes=10)

			self.x_test = x_test.reshape(10000, 28, 28, 1)
			self.y_test = y_test.reshape(10000, 1)
			self.y_test_onehot = keras.utils.to_categorical(y_test, num_classes=10)

        elif dataset == 'fasthion-mnist':
        	shape = (28, 28, 1)
			(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
			x_train = x_train.astype('float32') / 255.
			x_test = x_test.astype('float32') / 255.

			self.x_train = x_train.reshape(60000, 28, 28, 1)
			self.y_train = y_train.reshape(60000, 1)
			self.y_train_onehot = keras.utils.to_categorical(y_train, num_classes=10)

			self.x_test = x_test.reshape(10000, 28, 28, 1)
			self.y_test = y_test.reshape(10000, 1)
			self.y_test_onehot = keras.utils.to_categorical(y_test, num_classes=10)

        elif dataset == 'cifar10':
            shape = (32, 32, 3)
            raise NotImplementedError
        elif dataset == 'lfw':
        	raise NotImplementedError

        if autoencoder == 'dense':
            self.autoencoder = Autoencoder(path, dim=dim, input_shape=shape)
        elif autoencoder == 'conv':
            self.autoencoder = ConvolutionalAutoencoder(path, dim=dim, input_shape=shape)
        elif autoencoder == 'cond':
            raise NotImplementedError
        


    def train_autoencoder():

    def 
