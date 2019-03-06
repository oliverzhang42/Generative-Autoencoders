from GeneratorInterface import GeneratorInterface

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization
from keras.optimizers import Adam

import numpy as np

import os
import ot

class OTGenerator(GeneratorInterface):
    def __init__(self, path, batch_size=512, pruning=False, layers=4):
        assert os.path.isdir(path), "Given path is not a proper directory!"

        self.path = path
        self.batch_size = batch_size
        self.layers=layers

        if pruning:
            raise NotImplementedError

        self.pruning = pruning

    def initialize(self, encodings, recompute=True, file_inputs=None, file_answers=None):
        assert len(encodings) % self.batch_size == 0, "The length of the encodings has to be divisible by batch_size!" #TODO: make this general!
        assert len(encodings.shape) == 2

        self.encodings = encodings
        self.encoding_length = encodings.shape[1]

        if recompute:
            self.ot_compute_answers(encodings)

            if (not (file_inputs is None)) and (not (file_answers is None)):
                self.save_data(file_inputs, file_answers)
        else:
            full_inputs = os.path.join(self.path, file_inputs)
            full_answers = os.path.join(self.path, file_answers)

            self.inputs = np.load(full_inputs)
            self.answers = np.load(full_answers)

        self.create_model(layers=self.layers)
        
    def create_model(self, layers=4):
        model = Sequential()
        for i in range(layers):
          model.add(Dense(512, input_shape=(self.encoding_length,)))
          model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.encoding_length, name='dense'))

        self.model = model

    def compile(self, lr=0.001):
        opt = Adam(lr, beta_1=0.5, beta_2=0.999)
        self.model.compile(optimizer=opt, loss='mae')

    def ot_compute_answers(self, encodings=None):
        if encodings is None:
            encodings = self.encodings

        batch_size = self.batch_size
        inputs = np.random.random(size=encodings.shape)
        answers = []

        for i in range(len(encodings) // batch_size):
            if i % 10 == 0:
                  print("Batch Number: {}".format(i))

            real_vec = encodings[i*batch_size: (i+1)*batch_size]
            fake_vec = inputs[i*batch_size: (i+1)*batch_size]
          
            a = np.ones((batch_size, ))
            b = np.ones((batch_size, ))
          
            M = []
          
            for j in range(batch_size):
                costs = []
            
                for k in range(batch_size):
                    costs.append(np.sum(np.abs(fake_vec[j] - real_vec[k]))) # TODO: Debug the abs & sum & encodings vs real_vec?
                M.append(costs)
            
            M = np.array(M)
            mapping = ot.emd(a, b, M)
          
            for j in range(batch_size):
                index = np.argmax(mapping[j])
                answers.append(real_vec[index])

        self.answers = np.array(answers)
        self.inputs = inputs

    def save_data(self, file_inputs, file_answers):
        full_inputs = os.path.join(self.path, file_inputs)
        full_answers = os.path.join(self.path, file_answers)

        np.save(full_inputs, self.inputs)
        np.save(full_answers, self.answers)

    def train(self, inputs=None, answers=None, lr=0.001, epochs=10):
        if inputs is None:
            inputs = self.inputs
        if answers is None:
            answers = self.answers

        model = self.model
        self.compile(lr=lr)

        train_set = inputs[:7*len(inputs)//10]
        train_ans = answers[:7*len(inputs)//10]

        val_set = inputs[7*len(inputs)//10:]
        val_ans = answers[7*len(inputs)//10:]

        model.fit(train_set, train_ans, validation_data=(val_set, val_ans), epochs=epochs)

    def load_weights(self, name):
        full_path = os.path.join(self.path, name)
        self.model.load_weights(full_path)

    def save_weights(self, name):
        full_path = os.path.join(self.path, name)
        self.model.save_weights(full_path)

    def generate(self, number):
        model = self.model
        self.compile()

        inputs = np.random.random(size=(number, self.encoding_length))
        return self.model.predict(inputs)