from OTTransporter import OTTransporter

import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization
from keras.optimizers import Adam

import numpy as np
import tensorflow as tf

import os
import ot

from utils import *

def customLoss(y_pred, y_true):
    import pudb; pudb.set_trace()

    y_pred_expanded = tf.expand_dims(y_pred, 1)

    for i in range(499):
    	y_pred_expanded = tf.concat((y_pred_expanded, tf.expand_dims(y_pred, 1)), axis=1)

    y_true_expanded = tf.convert_to_tensor([y_true for i in range(500)])

    dist = tf.norm(y_pred_expanded-y_true_expanded, axis=2)
    weighted = -K.log(dist + tf.convert_to_tensor(1, dtype='float32'))
    losses = tf.reduce_sum(weighted, axis=1)

    '''
    For the penalty term, compare y_pred_expanded to itself... D:
    '''

    return losses + penalty

class PTGenerator(OTTransporter):
    def initialize(self, encodings, *args, **kwargs):
        assert len(encodings) % self.batch_size == 0, "The length of the encodings has to be divisible by batch_size!" #TODO: make this general!
        assert len(encodings.shape) == 2

        self.encodings = encodings
        self.encoding_length = encodings.shape[1]

        if self.distr == 'uniform':
            self.inputs = np.random.uniform(size=encodings.shape)
        elif self.distr == 'normal':
            self.inputs = np.random.normal(size=encodings.shape)

        self.create_model(layers=self.layers)

    def compile(self, lr=0.001):
        opt = Adam(lr, beta_1=0.5, beta_2=0.999)
        self.model.compile(optimizer=opt, loss=customLoss)

    def train(self, lr=0.001, epochs=10):
        model = self.model
        self.compile(lr=lr)

        train_set = self.inputs[:7*len(self.inputs)//10]
        train_ans = self.encodings[:7*len(self.encodings)//10]

        val_set = self.inputs[7*len(self.inputs)//10:]
        val_ans = self.encodings[7*len(self.encodings)//10:]

        model.fit(train_set, train_ans, validation_data=(val_set, val_ans), epochs=epochs, batch_size=500)