from GeneratorInterface import GeneratorInterface

import numpy as np

class RandomGenerator(GeneratorInterface):
    def __init__(self, distr):
        self.distr = distr

    def initialize(self, encodings, *args, **kwargs):
    	self.shape = encodings[0].shape

    def generate(self, number):
        shape = (number,) + self.shape

        if self.distr == 'uniform':
            return np.random.random(size=shape)
        elif self.distr == 'normal':
        	return np.random.normal(size=shape)