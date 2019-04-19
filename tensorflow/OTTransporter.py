from GeneratorInterface import GeneratorInterface

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization
from keras.optimizers import Adam
import keras.backend as K

import numpy as np

import tensorflow as tf

import os
import ot

from utils import *

# Needs debugging...
def sqrt_loss(y_true, y_pred):
    distances = K.sum(K.abs(y_true - y_pred), axis=1)
    distances = K.sqrt(distances+tf.convert_to_tensor([1], dtype='float32'))
    loss = K.mean(distances)

    return loss

def log_loss(y_true, y_pred):
    distances = K.sum(K.abs(y_true - y_pred), axis=1)
    distances = K.log(distances+tf.convert_to_tensor([1], dtype='float32'))
    loss = K.mean(distances)

    return loss

class OTTransporter(GeneratorInterface):
    def __init__(self, path, batch_size=512, pruning=False, layers=4, distr='uniform', ratio=1):
        assert os.path.isdir(path), "Given path is not a proper directory!"
        assert ratio <= 1, "You cannot have more than 100\% of the data!"

        self.path = path
        self.batch_size = batch_size
        self.layers=layers
        self.distr=distr

        # The proportion of the encodings you use to generate your training answers
        self.ratio=ratio

        if pruning:
            raise NotImplementedError

        self.pruning = pruning

    def initialize(self, encodings, recompute=True, file_inputs=None, file_answers=None, activation='sigmoid'):
        assert len(encodings) % self.batch_size == 0, "The length of the encodings has to be divisible by batch_size!" #TODO: make this general!
        assert len(encodings.shape) == 2

        self.encodings = encodings
        self.encoding_length = encodings.shape[1]
        self.input_length = self.encoding_length

        if recompute:
            self.get_answers()

            if (not (file_inputs is None)) and (not (file_answers is None)):
                self.save_data(file_inputs, file_answers)
        else:
            full_inputs = os.path.join(self.path, file_inputs)
            full_answers = os.path.join(self.path, file_answers)

            self.inputs = np.load(full_inputs)
            self.answers = np.load(full_answers)

        self.create_model(layers=self.layers, activation=activation)
        
    def create_model(self, layers=4, activation='sigmoid'):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.input_length,)))
        model.add(LeakyReLU(alpha=0.2))

        for i in range(layers):
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(self.encoding_length, name='dense', activation=activation))

        self.model = model

    def compile(self, lr=0.001):
        print("yay")
        opt = Adam(lr, beta_1=0.5, beta_2=0.999)
        self.model.compile(optimizer=opt, loss=log_loss) #Fix?

    def get_answers(self):
        if self.distr == 'uniform':
            random_input = np.random.random(size=(len(self.encodings), self.input_length)) # Don't let it be hard coded
        elif self.distr == 'normal':
            random_input = np.random.normal(size=(len(self.encodings), self.input_length))
        else:
            raise Exception("I don't understand the distribution {}".fomat(self.distr))

        self.answers, self.inputs = ot_compute_answers(random_input, 
            self.encodings, self.batch_size, ratio=self.ratio)

    def save_data(self, file_inputs, file_answers):
        full_inputs = os.path.join(self.path, file_inputs)
        full_answers = os.path.join(self.path, file_answers)

        np.save(full_inputs, self.inputs)
        np.save(full_answers, self.answers)

    def train(self, lr=0.001, epochs=10):
        model = self.model
        self.compile(lr=lr)

        train_set = self.inputs[:7*len(self.inputs)//10]
        train_ans = self.answers[:7*len(self.answers)//10]

        val_set = self.inputs[7*len(self.inputs)//10:]
        val_ans = self.answers[7*len(self.answers)//10:]

        model.fit(train_set, train_ans, validation_data=(val_set, val_ans), epochs=epochs)

    def load(self, name):
        full_path = os.path.join(self.path, name)
        self.model.load_weights(full_path)

    def save(self, name):
        full_path = os.path.join(self.path, name)
        self.model.save_weights(full_path)

    def generate(self, number):
        model = self.model
        self.compile()

        if self.distr == 'uniform':
            inputs = np.random.random(size=(number, self.input_length))
        elif self.distr == 'normal':
            inputs = np.random.normal(size=(number, self.input_length))
        elif self.distr == 'multi':
            assert number % 10 == 0, "Batch Size needs to be divisible by 10."

            inputs = None

            for i in range(10):
                shifted = np.random.normal(size=(number//10, self.input_length))
                shifted[:,i] += 2*np.ones(shape=(number//10))

                if i == 0:
                    inputs = shifted
                else:
                    inputs = np.concatenate((inputs, shifted))
        else:
            raise Exception("I don't know what this distribution is: {}".format(distr))

        np.random.shuffle(inputs)

        return self.model.predict(inputs)