import argparse

import numpy as np

import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.distributions import uniform

from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Transporter():
    def __init__(self, inputs, dim, path='.', batch_size=512):
        self.inputs = torch.Tensor(inputs)
        self.path = path
        self.dim = dim
        self.batch_size = batch_size

        self.distr = uniform.Uniform(-1, 1)

        self.create_model()

    def create_model(self):
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
            torch.nn.Linear(H, self.dim),
        )

    def save_weights(self, name):
        full_path = os.path.join(self.path, name)
        torch.save(self.model.state_dict(), full_path)

    def load_weights(self, name):
        full_path = os.path.join(self.path, name)
        self.model.load_state_dict(torch.load(full_path))

    def train(self, steps, lr=0.001, images=False):
        loss_fn = torch.nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for i in range(steps):
            inputs = self.distr.sample((self.batch_size, self.dim))

            indices = np.random.choice(range(len(self.inputs)), size=self.batch_size)
            real_vecs = self.inputs[indices]

            answer_indices = ot_compute_answers(inputs.cpu().numpy(), real_vecs.cpu().numpy())
            answers = real_vecs[answer_indices]

            generated =self.model(inputs)

            loss = loss_fn(generated, answers)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(loss.item())
            
            if i % 100 == 0 and images:
                display_points(generated.detach().cpu(), answers.cpu(), self.path, show=False, index=i)

        self.save_weights("transporter")

    def generate(self):
        inputs = self.distr.sample((self.batch_size, self.dim))
        return self.model(inputs).detach().cpu().numpy()