from OTTransporter import OTTransporter

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization
from keras.optimizers import Adam

import numpy as np

import os
import ot

from utils import *

class OTGenerator(OTTransporter):
    def initialize(self, encodings, *args, **kwargs):
        assert len(encodings) % self.batch_size == 0, "The length of the encodings has to be divisible by batch_size!" #TODO: make this general!
        assert len(encodings.shape) == 2

        self.encodings = encodings
        self.encoding_length = encodings.shape[1]

        self.create_model(layers=self.layers)

    def one_step(self):
        if self.distr == 'uniform':
            inputs = np.random.random(size=(self.batch_size, self.encoding_length))
        elif self.distr == 'normal':
            inputs = np.random.normal(size=(self.batch_size, self.encoding_length))

        predicted = self.model.predict(inputs)
        
        indicies = np.random.choice(range(len(self.encodings)), self.batch_size)
        real = self.encodings[indicies]

        test, answers = ot_compute_answers(predicted, real, self.batch_size, verbose=False)

        return self.model.train_on_batch(inputs, answers)

    def train(self, lr=0.001, epochs=10):
        model = self.model
        self.compile(lr=lr)

        for i in range(epochs):
            print("Epoch Number {}".format(i))

            total_loss = 0
            for j in range(len(self.encodings)//self.batch_size):
                loss = self.one_step()
                total_loss += loss
                print("Loss: {}".format(total_loss/(j+1)))

        