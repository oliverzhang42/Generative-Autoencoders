import keras
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

import os

class Autoencoder():
    def __init__(self, path, dim=10, input_shape=(28,28,1)):
    	assert os.path.isdir(path), "Given path is not a proper directory!"

    	self.path = path
        self.dim = dim
        self.create_model()

    def create_model(self, input_shape):
        input_img = Input(shape=(28, 28, 1))

        net1 = Flatten()(input_img)
        net1 = Dense(512)(net1)
        net1 = LeakyReLU()(net1)
        net1 = BatchNormalization()(net1)
        net1 = Dense(512)(net1)
        net1 = LeakyReLU()(net1)
        net1 = BatchNormalization()(net1)
        encoded = Dense(dim, activation='sigmoid')(net1)

        input_sequence = Input(shape=(dim,))
        net2 = Dense(512)(input_sequence)
        net2 = LeakyReLU()(net2)
        net2 = BatchNormalization()(net2)
        net2 = Dense(512)(net2)
        net2 = LeakyReLU()(net2)
        net2 = BatchNormalization()(net2)
        net2 = Dense(784, activation='sigmoid')(net2)
        decoded = Reshape((28, 28, 1))(net2)

        self.encoder = Model(input_img, encoded)
        self.decoder = Model(input_sequence, decoded)
        self.autoencoder = Model(input_img, decoder(encoded))

    def compile(self, lr-0.001):
    	opt = Adam(lr, beta_1=0.5, beta_2=0.999)
        self.autoencoder.compile(optimizer=opt, loss='mse')

    def train(self, x_train, lr=0.001, epochs=10):
    	autoencoder = self.autoencoder
    	self.compile(lr=lr)

        autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test), 
        	epochs=epochs)

    def encode(self, data)
        self.compile()
        return self.encoder.predict(data)

    def save_weights(self, name):
        full_path = os.path.join(self.path, name)
    	self.autoencoder.save_weights(full_path)

    def load_weights(self, name):
        full_path = os.path.join(self.path, name)
    	self.autoencoder.load_weights(full_path)

    def evaluate(self, x_test, verbose=0):
        self.compile()
        loss = self.autoencoder.evaluate(x_test, x_test)
        
        if verbose == 1:
        	print("Evaluating the Autoencoder. Loss:")
        	print(loss)

        return loss