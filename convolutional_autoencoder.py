from autoencoder import Autoencoder

import keras
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization, Conv2D, Conv2DTranspose, Reshape
from keras.optimizers import Adam

import numpy as np

import os

class ConvolutionalAutoencoder(Autoencoder):
    def create_model(self, input_shape):
        input_img = Input(shape=(28, 28, 1))

        net1 = Conv2D(256, kernel_size=(3, 3))(input_img)
        net1 = LeakyReLU()(net1)
        net1 = BatchNormalization()(net1)
        net1 = MaxPooling2D(pool_size=(2, 2))(net1)
        net1 = Conv2D(256, (3, 3))(net1)
        net1 = LeakyReLU()(net1)
        net1 = BatchNormalization()(net1)
        net1 = MaxPooling2D(pool_size=(2, 2))(net1)
        net1 = Flatten()(net1)
        net1 = Dense(128)(net1)
        net1 = LeakyReLU()(net1)
        net1 = BatchNormalization()(net1)
        encoded = Dense(self.dim, activation='sigmoid')(net1)

        input_sequence = Input(shape=(self.dim,))
        net2 = Dense(128)(input_sequence)
        net2 = LeakyReLU()(net2)
        net2 = Dense(49)(input_sequence)
        net2 = LeakyReLU()(net2)
        net2 = BatchNormalization()(net2)
        net2 = Reshape((7, 7, 1))(net2)
        net2 = Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')(net2)
        net2 = LeakyReLU()(net2)
        net2 = BatchNormalization()(net2)
        net2 = Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')(net2)
        net2 = LeakyReLU()(net2)
        decoded = BatchNormalization()(net2)
        decoded = Conv2DTranspose(1, (3,3), activation='sigmoid', padding='same')(net2)

        self.encoder = Model(input_img, encoded)
        self.decoder = Model(input_sequence, decoded)
        self.autoencoder = Model(input_img, self.decoder(encoded))