import argparse

import numpy as np

import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.distributions import uniform

from utils import *

from transporter import Transporter

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Generator(Transporter):
    def train(self, steps, lr=0.001, images=False):
        loss_fn = torch.nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), betas=[0.5,0.999], lr=lr)

        for i in range(steps):
            inputs = self.distr.sample((self.batch_size, self.dim))
            generated = self.model(inputs)

            indices = np.random.choice(range(len(self.inputs)), size=self.batch_size)
            real_vecs = self.inputs[indices]

            answer_indices = ot_compute_answers(generated.detach().cpu().numpy(), real_vecs.cpu().numpy())
            answers = real_vecs[answer_indices]

            loss = loss_fn(generated, answers)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(loss.item())
            
            if i % 100 == 0 and images:
                display_points(generated.detach().cpu(), answers.cpu(), self.path, show=False, index=i)

        self.save_weights("generator")

