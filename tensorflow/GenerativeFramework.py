from autoencoder import Autoencoder
from autoencoder_identity import Identity
from ClusteringGenerator import ClusteringGenerator
from convolutional_autoencoder import ConvolutionalAutoencoder
import keras
from KDEGenerator import KDEGenerator
from keras.datasets import mnist, fashion_mnist

import numpy as np

from math import *

import os
from OTTransporter import OTTransporter
from OTGenerator import OTGenerator
from OTMultiHead import OTMultiHead
from PTTransporter import PTTransporter
from PTGenerator import PTGenerator
from RandomGenerator import RandomGenerator

class GenerativeFramework():
    def __init__(self, weight_path, img_path, dataset='mnist', autoencoder='dense', 
        dim=10, generator='ot transport', pruning=False, layers=4, batch_size=512, 
        distr='uniform', ratio=1, clusters=20, noise_intensity=0.15, heads=4):
        assert os.path.isdir(weight_path), 'The given path isnt a directory!'
        assert os.path.isdir(img_path), 'The given path isnt a directory!'
        assert dataset in ['mnist', 'fashion-mnist', 'lfw', 'cifar10', 'toy_eight', 'toy_two'], 'The dataset is not recognized!'
        assert autoencoder in ['dense', 'conv', 'cond', 'identity'], 'The autoencoder is not recognized!'
        assert generator in ['ot transport', 'ot generator', 'pt transport', 'pt generator', 'kde', 'k-means', 'random', 'ot multi'], 'The generator is not recognized!'
        assert distr in ['uniform', 'normal', 'multi'], 'The random distribution is not recognized!'

        self.weight_path = weight_path
        self.img_path = img_path
        self.batch_size = batch_size

        if dataset == 'mnist':
        	# TODO: Make the preprocessing into a separate file.
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

        elif dataset == 'fashion-mnist':
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
        elif dataset == 'toy_eight':
            x_train = []
            x_test = []

            d = {0: (10, 0), 1: (10/sqrt(2), 10/sqrt(2)), 2: (0, 10), 3: (-10/sqrt(2), 10/sqrt(2)), \
                 4: (-10, 0), 5: (-10/sqrt(2), -10/sqrt(2)), 6: (0, -10), 7: (10/sqrt(2), -10/sqrt(2))}

            for i in range(60000):
                noise = 0.1 * np.random.normal(size=2)
                noisex = noise[0]
                noisey = noise[1]
                
                center = d[i%8]
                l = [center[0] + noisex, center[1] + noisey]
                x_train.append(l)

            for i in range(10000):
                noise = 0.1 * np.random.normal(size=2)
                noisex = noise[0]
                noisey = noise[1]
              
                center = d[i%8]
                l = [center[0] + noisex, center[1] + noisey]
                x_test.append(l)

            self.x_train = np.array(x_train)
            self.x_test = np.array(x_test)
        elif dataset == 'toy_two':
            x_train = []
            x_test = []

            d = {0: (1, 0), 1: (-1, 0)}

            for i in range(60000):
                noise = 0.1 * np.random.normal(size=2)
                noisex = noise[0]
                noisey = noise[1]
                
                center = d[i%2]
                l = [center[0] + noisex, center[1] + noisey]
                x_train.append(l)

            for i in range(10000):
                noise = 0.05 * np.random.normal(size=2)
                noisex = noise[0]
                noisey = noise[1]
              
                center = d[i%2]
                l = [center[0] + noisex, center[1] + noisey]
                x_test.append(l)

            self.x_train = np.array(x_train)
            self.x_test = np.array(x_test)

        if autoencoder == 'dense':
            self.autoencoder = Autoencoder(weight_path, batch_size=batch_size, dim=dim, input_shape=shape)
        elif autoencoder == 'conv':
            self.autoencoder = ConvolutionalAutoencoder(weight_path, batch_size=batch_size, dim=dim, input_shape=shape)
        elif autoencoder == 'cond':
            raise NotImplementedError
        elif autoencoder == 'identity':
            self.autoencoder = Identity()
        else:
            raise NotImplementedError
        
        if generator == 'ot transport':
            self.generator = OTTransporter(weight_path, batch_size=batch_size, pruning=pruning, layers=layers, distr=distr, ratio=ratio)
        elif generator == 'ot generator':
            self.generator = OTGenerator(weight_path, batch_size=batch_size, layers=layers, distr=distr, ratio=ratio)
        elif generator == 'ot multi':
            self.generator = OTMultiHead(weight_path, batch_size=batch_size, layers=layers, distr=distr, ratio=ratio, heads=heads)
        elif generator == 'pt transport':
            self.generator = PTTransporter(weight_path, batch_size=batch_size, pruning=pruning, layers=layers, distr=distr, ratio=ratio)
        elif generator == 'pt generator':
            self.generator = PTGenerator(weight_path, batch_size=batch_size, pruning=pruning, layers=layers, distr=distr, ratio=ratio)
        elif generator == 'kde':
            self.generator = KDEGenerator(distr=distr, noise_intensity=noise_intensity)
        elif generator == 'k-means':
            self.generator = ClusteringGenerator(clusters=clusters)
        elif generator == 'random':
            self.generator = RandomGenerator(distr=distr)
        else:
            raise NotImplementedError

    def train_autoencoder(self, lr=0.001, epochs=10):
        self.autoencoder.train(self.x_train, x_test=self.x_test, lr=lr, epochs=epochs)
        
    def evaluate_autoencoder(self, data=None):
        if data is None:
        	data = self.x_test

        loss = self.autoencoder.evaluate(data)
        print(loss)

    def load_autoencoder(self, file_name):
        self.autoencoder.load_weights(file_name)

    def save_autoencoder(self, file_name):
        self.autoencoder.save_weights(file_name)

    def initialize_generator(self, recompute, file_inputs, file_answers, activation='sigmoid'): # TODO: Either switch everything to regular args or switch this to kwargs
    	encodings = self.autoencoder.encode(self.x_train) #Fix?
    	self.generator.initialize(encodings, recompute, file_inputs, file_answers, activation=activation)

    def train_generator(self, **kwargs):
        self.generator.train(**kwargs)

    def save_generator(self, file_name):
    	self.generator.save(file_name)

    def load_generator(self, file_name):
    	self.generator.load(file_name)

    def generate_images(self, file_name, number=10000):
        encodings = self.generator.generate(number)
        images = self.autoencoder.decode(encodings)

        full_path = os.path.join(self.img_path, file_name)
        np.save(full_path, images)