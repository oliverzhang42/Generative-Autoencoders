from keras.optimizers import Adam

from OTTransporter import OTTransporter

from utils import *

class PTTransporter(OTTransporter):
    def compile(self, lr=0.001):
        opt = Adam(lr, beta_1=0.5, beta_2=0.999)
        self.model.compile(optimizer=opt, loss='mse')

    def get_answers(self):
        self.answers, self.inputs = pt_compute_answers(self.encodings, self.batch_size, distr=self.distr, ratio=self.ratio)
