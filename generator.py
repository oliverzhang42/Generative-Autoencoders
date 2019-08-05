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
        images: (bool) Whether to store images or not. NOTE: Images will be
        unable to be generated if dimension is > 2. This function is for the
        toy datasets only!!!
        '''
        lambda_stdev = 0
        lambda_cyclic = 0
        lambda_unbounded = 0.3
        print("Beginning Generator Training!")
        print("USING REGULARIZATION WITH lambda_stdev={} and lambda_cyclic={} and lambda_unbounded={}".format(lambda_stdev, lambda_cyclic, lambda_unbounded))

        loss_fn = torch.nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), betas=[0.5,0.999], lr=lr)

        for i in range(steps):
            # Samples inputs from noise distribution
            inputs = self.noise.sample((self.batch_size, self.dim))
            # Generaters corresponding latent distribution predictons
            generated = self.model(inputs)

            # Samples latent distribution
            indices = np.random.choice(range(len(self.latent)), size=self.batch_size)
            real_vecs = self.latent[indices]

            # Uses Optimal Transport to compute "Feedback". generated[i] should
            # have actually been answers[i]


            answer_indices = optimal_transport(generated.detach().cpu().numpy(), 
                real_vecs.cpu().numpy())
            #answer_indices = sinkhorn_transport(generated.detach().cpu().numpy(), 
            #    real_vecs.cpu().numpy(), reg=0.01)
            answers = real_vecs[answer_indices]

            loss = loss_fn(generated, answers)
            reg1 = reg_stdev(generated, answers, lambda_stdev)
            reg2 = reg_cyclic(generated, answers, lambda_cyclic)
            reg3 = unbounded_cyclic(generated, lambda_unbounded)

            total_loss = loss + reg1 + reg2 + reg3
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print("Losses at step {}: {} {} {} {}".format(i, loss.item(), reg1.item(), reg2.item(), reg3.item()))
            
            if images and i % 1000 == 0:
                save_points(generated.detach().cpu(), answers.cpu(), self.path, index=i)

            if i % 1000 == 999:
                print("Saving checkpoint as 'generator_{}'".format(i))
                self.save_weights("generator_{}".format(i))

        print("Saving Generator Weights...")
        self.save_weights("generator")

