from GeneratorInterface import GeneratorInterface

import numpy as np

class KDEGenerator(GeneratorInterface):
    def __init__(self, noise_intensity=0.15, distr='uniform'):
        assert distr in ['uniform', 'normal'], "I don't recognize that random distribution!"

        self.noise_intensity = noise_intensity
        self.distr = distr

    def initialize(self, encodings, *args, **kwargs): # I think this is what we do when there's more kwargs than we accept
        self.encodings = encodings

    def generate(self, number):
        generated = []

        for i in range(number):
            index = np.random.choice(range(len(self.encodings)))
            sample = self.encodings[index]

            if self.distr == 'uniform':
                noise = np.random.random(size=sample.shape) - 0.5
            elif self.distr == 'normal':
            	noise = np.random.normal(size=sample.shape)
            
            sample += noise * self.noise_intensity
            generated.append(sample)

        return np.array(generated)

