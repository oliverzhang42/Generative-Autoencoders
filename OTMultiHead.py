from OTTransporter import OTTransporter

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization, Add
from keras.optimizers import Adam

import numpy as np

import os
import ot

from utils import *

class OTMultiHead(OTTransporter):
    def __init__(self, path, batch_size=512, pruning=False, layers=4, distr='uniform', ratio=1, heads=4):
        OTTransporter.__init__(self, path, batch_size, pruning, layers, distr, ratio)
        self.num_heads = heads

    def create_model(self, layers=4):
        input1 = Input(shape=(self.encoding_length,))

        net = Dense(512)(input1)
        net = LeakyReLU(alpha=0.2)(net)

        net = Dense(512)(net)
        net = LeakyReLU(alpha=0.2)(net)

        base = Dense(self.encoding_length, activation='sigmoid')(net)
        self.base = Model(inputs=input1, outputs=base)


        input2 = Input(shape=(self.encoding_length,))
        predictor = Dense(512)(input2)
        predictor = LeakyReLU()(predictor)
        predictor = Dense(self.num_heads, activation='softmax')(predictor)
        self.predictor = Model(inputs=input2, outputs=predictor)

        input3 = Input(shape=(self.encoding_length,))

        heads = self.num_heads

        head = []

        for i in range(heads):
            net2 = Dense(512)(input3)
            net2 = LeakyReLU(alpha=0.2)(net2)
            net2 = Dense(self.encoding_length, activation='sigmoid')(net2)
            net2 = Add()([net2, input3])
            head.append(net2)

        self.heads = []

        for i in range(heads):
            self.heads.append(Model(inputs=input3, outputs=head[i]))

    def compile(self, lr=0.001):
        opt = Adam(lr, beta_1=0.5, beta_2=0.999)
        
        self.base.compile(optimizer=opt, loss='mse')
        self.predictor.compile(optimizer=opt, loss='categorical_crossentropy')

        for i in range(len(self.heads)):
            self.heads[i].compile(optimizer=opt, loss='mse') #Fix?
    
    def one_step(self):
        indices = np.random.choice(range(len(self.inputs)), size=self.batch_size)
        inputs = self.base.predict(self.inputs[indices])
        answers = self.answers[indices]

        losses = []

        for i in range(len(self.heads)):
            predicted = self.heads[i].predict(inputs)
            losses.append(np.sum((predicted - answers)**2, axis=1))
        
        train = [[] for i in range(len(self.heads))]

        predictor_train = [[0 for i in range(len(self.heads))] for j in range(len(inputs))]

        for i in range(len(inputs)):
            index = np.argmin([losses[j][i] for j in range(len(self.heads))])

            train[index].append(i)
            predictor_train[i][index] = 1

        losses = []

        for i in range(len(self.heads)):
            if len(train[i]) != 0:
                loss = self.heads[i].train_on_batch(inputs[train[i]], answers[train[i]])
                losses.append(loss * len(train[i]))
        
        predict_loss = self.predictor.train_on_batch(inputs, np.array(predictor_train))

        s = ""

        for i in range(len(self.heads)):
            s += 'train{}: {}, '.format(i, len(train[i]))

        print(s)
        print("Predict loss: {}".format(predict_loss))

        return np.sum(losses)/self.batch_size

    def load(self, name):
        pass

    def save(self, name):
        full_path = os.path.join(self.path, name)

        for i in range(len(self.heads)):
            self.heads[i].save_weights(full_path + str(i))

    def train(self, lr=0.001, epochs=10):
        self.compile(lr=lr)
        train_set = self.inputs[:7*len(self.inputs)//10]
        train_ans = self.answers[:7*len(self.answers)//10]

        val_set = self.inputs[7*len(self.inputs)//10:]
        val_ans = self.answers[7*len(self.answers)//10:]

        self.base.fit(train_set, train_ans, validation_data=(val_set, val_ans), epochs=epochs)

        #'''

        self.base.trainable=False

        # print(self.base.get_weights()[0][0])

        #import pudb; pudb.set_trace()

        for i in range(epochs):
            print("Epoch Number {}".format(i))

            total_loss = 0
            for j in range(len(self.encodings)//self.batch_size):
                loss = self.one_step()
                total_loss += loss
                print("Loss: {}".format(total_loss/(j+1)))

        # print(self.base.get_weights()[0][0])
        '''#'''

    def generate(self, number):
        self.compile()

        if self.distr == 'uniform':
            inputs = np.random.random(size=(number, self.encoding_length))
        elif self.distr == 'normal':
            inputs = np.random.normal(size=(number, self.encoding_length))
        else:
            raise Exception("I don't know what this distribution is: {}".format(distr))


        #'''
        import pudb; pudb.set_trace()
        
        generated = self.base.predict(inputs)

        predicted = self.predictor.predict(generated)

        predict = [[] for i in range(self.num_heads)]

        for i in range(len(predicted)):
            index = np.argmax(predicted[i])
            predict[index].append(generated[i])

        final_generated = np.array([])

        for i in range(self.num_heads):
            g = self.heads[i].predict(np.array(predict[i]))
            if len(final_generated) == 0:
                final_generated = g
            else:
                final_generated = np.concatenate((final_generated, g), axis=0)

        #for i in range(1, len(self.heads)):
        #    generated = np.concatenate((generated, self.heads[i].predict(inputs)))

        return final_generated
        '''

        return self.base.predict(inputs)
        #'''