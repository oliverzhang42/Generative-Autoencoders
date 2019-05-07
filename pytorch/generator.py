import argparse
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

'''
Generator class. Serves the same purpose as Transporter Class, namely to
generate points mimicking a latent distribution. However, instead of using
Optimal Transport to calculate how you should map, it first maps the points
and then uses Optimal Transport to calculate feedback.

It turns out, a small difference has a large effect in terms of model quality.
'''
class Generator(Transporter):
    def train(self, steps, lr=0.001, images=False):
        '''
        Train generator using optimal transport for any number of iterations.

        steps: (int) Number of iterations to train
        lr: (int) Learning Rate
        images: (bool) True means images will be stored under self.path, False
        means they will not
        '''
        print("Beginning Generator Training!")

        loss_fn = torch.nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), betas=[0.5,0.999], lr=lr)

        for i in range(steps):
            # Samples inputs from noise distribution
            inputs = self.noise.sample((self.batch_size, self.dim))
            # Generaters corresponding latent distribution predictons
            generated = self.model(inputs)

            # Samples latent distribution
            indices = np.random.choice(range(len(self.inputs)), size=self.batch_size)
            real_vecs = self.inputs[indices]

            # Uses Optimal Transport to compute "Feedback". generated[i] should
            # have actually been answers[i]
            answer_indices = optimal_transport(generated.detach().cpu().numpy(), 
                real_vecs.cpu().numpy())
            answers = real_vecs[answer_indices]

            loss = loss_fn(generated, answers)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100:
                print("Loss at step {}: {}".format(i, loss.item()))
            
            if images and i % 1000 == 0:
                save_points(generated.detach().cpu(), answers.cpu(), self.path, index=i)

        print("Saving Generator Weights...")
        self.save_weights("generator")

