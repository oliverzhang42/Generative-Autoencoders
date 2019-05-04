import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''Data Processing'''

x_train = []
x_test = []

d = {0: (1, 0), 1: (-1, 0)}

for i in range(60000):
    noise = 0.1 * np.random.normal(size=2)
    noisex = noise[0]
    noisey = noise[1]
    
    center = d[i%2]
    l = [center[0] + noisex, center[1] + noisey]
    x_train.append(l)

for i in range(10000):
    noise = 0.05 * np.random.normal(size=2)
    noisex = noise[0]
    noisey = noise[1]
  
    center = d[i%2]
    l = [center[0] + noisex, center[1] + noisey]
    x_test.append(l)

x_train = torch.Tensor(x_train)
x_test = np.array(x_test)

'''Building a Generator''' 

H = 512
dim = 2

block1 = torch.nn.Sequential(
    torch.nn.Linear(dim, H),
    torch.nn.LeakyReLU()
)

block2 = torch.nn.Sequential(
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU()
)

block3 = torch.nn.Sequential(
    torch.nn.Linear(H, dim)
)

model = torch.nn.Sequential(
    block1,
    block2,
    block3
)

'''Building a Discriminator'''

block4 = torch.nn.Sequential(
    torch.nn.Linear(dim, H),
    torch.nn.LeakyReLU()
)

block5 = torch.nn.Sequential(
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU()
)

block6 = torch.nn.Sequential(
    torch.nn.Linear(H, dim),
    torch.nn.Sigmoid()
)

disc = torch.nn.Sequential(
    block4,
    block5,
    block6
)

entire = torch.nn.Sequential(
    model,
    disc
)

#entire.load_state_dict(torch.load("gan"))

import pudb; pudb.set_trace()

#'''
lr = 0.001
gen_optim = optim.Adam(model.parameters(), lr=lr)
disc_optim  = optim.Adam(disc.parameters(), lr=lr)

loss_fn = torch.nn.MSELoss()

def display_points(inputs, answers, show=True, index=0):
    for i in range(512):
        plt.scatter(inputs[i][0], inputs[i][1], color='blue')
        plt.scatter(answers[i][0], answers[i][1], color='green')
        
    if show:
        plt.show()
    else:
        plt.savefig("image{}.png".format(index))
        plt.clf()

for i in range(1000):
    '''Generate Everything Necessary'''
    inputs = torch.rand((512, 2))
    indices = np.random.choice(range(60000), size=512)
    real_vecs = x_train[indices]

    generated = model(inputs)

    fake = disc(generated)
    real = disc(real_vecs)

    '''Discriminator'''
    for params in model.parameters():
        params.requires_grad = False

    disc_loss = loss_fn(fake, torch.zeros(fake.shape)) + loss_fn(real, torch.ones(real.shape))

    disc_loss.backward(retain_graph=True)
    disc_optim.step()
    
    disc_optim.zero_grad()

    print(disc_loss.item())
    
    '''Generator'''
    for params in model.parameters():
        params.requires_grad = True

    for params in disc.parameters():
        params.requires_grad = False

    gen_loss = loss_fn(fake, torch.ones(fake.shape))

    gen_loss.backward()
    gen_optim.step()

    gen_optim.zero_grad()

    print(gen_loss.item())

    for params in disc.parameters():
        params.requires_grad = True

    if i % 10 == 0:
        display_points(generated.detach().cpu(), real_vecs.cpu(), show=False, index=i//10)

torch.save(entire.state_dict(), "gan1")
'''

#import pudb; pudb.set_trace()

'''
inputs = torch.rand((512, 2))
generated = model(inputs)

gen = generated.detach().cpu().numpy()

plt.scatter(x_test[:,0], x_test[:,1], color='blue')
plt.scatter(gen[:,0], gen[:,1], color='green')

plt.show()
#'''
