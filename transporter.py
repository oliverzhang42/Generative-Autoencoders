from copy import deepcopy
import numpy as np
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.distributions import uniform, normal

from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''
Transporter Class. Uses optimal transport to learn the mapping from
a noise distribution to some latent space distribution.
'''
class Transporter():
    def __init__(self, latent, noise='uniform', path='.', batch_size=512):
        '''
        Initializes the Transporter, including creating the model.

        latent: (np array) Latent space distribution to map to. Must be an
        array of one dimensional vectors.
        noise: (str) Noise distribution to map from. Must be either 'uniform',
        'normal', or 'gaussian'
        path: (str) Path to store any images/weights of the model
        batch_size: (int) Batch Size
        '''
        self.latent = torch.Tensor(latent)
        self.dim = len(latent[0])

        if noise.lower() == 'uniform':
            self.noise = uniform.Uniform(-1, 1)
        elif noise.lower() == 'normal' or noise.lower() == 'gaussian':
            self.noise = normal.Normal(0, 1)
        else:
            raise Exception("{} has not been implemented yet".format(noise))

        self.path = path
        self.batch_size = batch_size

        self.create_model()

    def create_model(self):
        '''
        Creates a model of 7 fully connected layers with LeakyReLU activation.
        All layers but last have 512 neurons.
        '''
        H = 512

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.dim, H),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(H, H),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(H, H),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(H, H),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(H, H),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(H, H),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(H, self.dim),
            torch.nn.Sigmoid()
        )
        print("Sigmoid is added at the end of the generator")

    def save_weights(self, name):
        '''
        Weights saved under self.path defined above.
        '''
        full_path = os.path.join(self.path, name)
        torch.save(self.model.state_dict(), full_path)

    def load_weights(self, name):
        '''
        Weights saved under self.path defined above.
        '''
        full_path = os.path.join(self.path, name)
        self.model.load_state_dict(torch.load(full_path))

    def train(self, steps, lr=0.001, images=False):
        '''
        Train transporter using optimal transport for any number of iterations.

        steps: (int) Number of iterations to train
        lr: (int) Learning Rate
        images: (bool) Whether to store images or not. NOTE: Images will be
        unable to be generated if dimension is > 2. This function is for the
        toy datasets only!!!
        '''
        print("Beginning Transporter Training!")

        loss_fn = torch.nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), betas=[0.5,0.999], lr=lr)

        for i in range(steps):
            # Samples inputs from noise distribution
            inputs = self.noise.sample((self.batch_size, self.dim))

            # Samples latent distribution
            indices = np.random.choice(len(self.latent), size=self.batch_size)
            real_vecs = self.latent[indices]

            # Computes optimal transport from inputs to latent. answers[i] will 
            # be the latent vector that inputs[i] should be mapped to.
            answer_indices = optimal_transport(inputs.cpu().numpy(), real_vecs.cpu().numpy())
            answers = real_vecs[answer_indices]

            generated = self.model(inputs)

            loss = loss_fn(generated, answers)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print("Loss at step {}: {}".format(i, loss.item()))
            
            if images and i % 1000 == 0:
                save_points(generated.detach().cpu(), answers.cpu(), self.path, name="points_{}".format(i))

            if i % 1000 == 0:
                print("Saving checkpoint at 'transporter_{}'".format(i))
                self.save_weights("transporter_{}".format(i))

        print("Saving Transporter Weights...")
        self.save_weights("transporter")

    def generate(self, batches=1):
        '''
        Generate realistic latent vectors by sampling noise and using the
        model to map them.

        batches: (int) The number of batches of realistic latent vectors to 
        generate.
        '''
        outputs = None

        for i in range(batches):
            inputs = self.noise.sample((self.batch_size, self.dim))
            latent_vec = self.model(inputs).detach().cpu().numpy()

            if outputs is None:
                outputs = latent_vec
            else:
                outputs = np.concatenate((outputs, latent_vec), 0)

        return outputs

    def interpolate(self, vec1, vec2, intermediate):
        '''
        Interpolate between two noise vectors.

        vec1: (np array) First noise vector
        vec2: (np array) Second noise vector
        intermediate: (int) Number of intermediate steps when interpolating.
        '''
        increment = (vec1 - vec2) / intermediate

        current = deepcopy(vec1)
        inputs = [deepcopy(current)]

        for i in range(size):
            current = current + increment
            inputs.append(deepcopy(current))

        return self.model(torch.Tensor(inputs)).detach().cpu().numpy()

    def evaluate(self):
        '''
        Evaluates how good the model is doing by calculating the optimal
        transport mapping from the model's predictions to some true data
        and then taking the average distance.
        '''
        generated = self.generate(batches=1)
        
        # Samples latent distribution
        indices = np.random.choice(len(self.latent), size=self.batch_size)
        real_vecs = self.latent[indices].cpu().numpy()

        answer_indices = optimal_transport(generated, real_vecs)

        distances = np.linalg.norm(generated - real_vecs[answer_indices], axis=1)
        mean_distance = np.mean(distances)

        print("Mean distance: {}".format(mean_distance))
