import numpy as np
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.distributions import uniform
from transporter import Transporter

from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Discriminator(Transporter):
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
            torch.nn.Linear(H, 1),
        )

    def train(self, steps, lr=0.001, images=False):
        '''
        Train discriminator to learn kantorovich potential.

        steps: (int) Number of iterations to train
        lr: (int) Learning Rate
        images: (bool) Whether to store images or not. NOTE: Images will be
        unable to be generated if dimension is > 2. This function is for the
        toy datasets only!!! TODO: get rid of images
        '''
        lambda_ = 0
        print("Training AEOT with lambda: {}".format(lambda_))

        loss_fn = torch.nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), betas=[0.5,0.999], lr=lr)

        for i in range(steps):
            # Samples inputs from noise distribution
            inputs = self.noise.sample((self.batch_size, self.dim))
            inputs.requires_grad = True
            # Generators corresponding kantorovich potential predictons
            potentials = self.model(inputs)
            # Calculates gradient wrt inputs.
            potentials.backward() # Is this the right direction?
            grad = inputs.grad
            optimizer.zero_grad()

            # Samples latent distribution
            indices = np.random.choice(range(len(self.latent)), size=self.batch_size)
            real_vecs = self.latent[indices]

            # Compute OT map and kantorovich potentials
            answer_indices, log = optimal_transport(generated.detach().cpu().numpy(), 
                real_vecs.cpu().numpy(), log=True)
            answers = real_vecs[answer_indices]

            real_potentials = log["v"] #u or v I forget lol

            loss1 = loss_fn(real_potentials, potentials)
            loss2 = loss_fn(torch.norm(grad, p=1, dim=1), torch.norm(answers - inputs, p=1, dim=1)) # Do I want 1 norm or 2 norm???
            
            total_loss = loss1 + lambda_ * loss2

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print("Losses at step {}: {} {} {}".format(i, total_loss.item(), loss1.item(), lambda_ * loss2.item()))
            
            if i % 1000 == 999:
                print("Saving checkpoint as 'discriminator_{}'".format(i))
                self.save_weights("discriminator_{}".format(i))

        print("Saving Generator Weights...")
        self.save_weights("discriminator")

    def generate(self, batches=1):
        g = []
       
        for i in range(batches):
            inputs = self.noise.sample((self.batch_size, self.dim))
            potentials = self.model(inputs)
            potentials.backward()

            generated = inputs + inputs.grad

            if g == []:
                g = generated.detach().cpu().numpy()
            else:
                g = np.concat((g, generated.detach().cpu().numpy()), 0)

        return g

