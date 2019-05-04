import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal, uniform


import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

from math import *
import random

from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''Data Processing'''

(x_train, x_test) = toy_eight()

#import pudb; pudb.set_trace()

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
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU()
)

block4 = torch.nn.Sequential(
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU()
)

block5 = torch.nn.Sequential(
    torch.nn.Linear(H, dim)
)

model = torch.nn.Sequential(
    block1,
    block2,
    block3,
    block4,
    block5
)


#model.load_state_dict(torch.load("ot_gen_deep"))

import pudb; pudb.set_trace()

#'''
lr = 0.0003
optimizer = optim.Adam(model.parameters(), lr=lr)

def loss_fn(gen_vec, real_vec, lambda1=0.1):
    gen = gen_vec.detach().cpu().numpy()

    indices = ot_compute_answers(gen, real_vec.cpu().numpy(), 512, verbose=False)

    answers = real_vec[indices]
    
    dist = torch.mean(torch.sum(torch.abs(gen_vec - answers), 1))

    if iteration % 100 == 0:
        display_points(gen_vec.detach().cpu().numpy(), answers.detach().cpu().numpy(), show=False, index=iteration//100)

    return dist

iteration = 0

def display_points(inputs, answers, show=True, index=0):
    for i in range(512):
        plt.scatter(inputs[i][0], inputs[i][1], color='blue')
        plt.scatter(answers[i][0], answers[i][1], color='green')

        plt.plot([inputs[i][0], answers[i][0]], [inputs[i][1], answers[i][1]], color='red')
    
    if show:
        plt.show()
    else:
        plt.savefig("image{}.png".format(index))
        index += 1
        plt.clf()

#import pudb; pudb.set_trace()

#model.load_state_dict(torch.load("ot_gen_test5"))

m1 = uniform.Uniform(-1, 1)

# Note!!! Right now, there's no random noise going on
# Also, right now there's no random sampling from the real_vecs.

#'''
for i in range(100):
    inputs = m1.sample((512, dim))
    generated = model(inputs)

    indices = np.random.choice(range(60000), 512)
    real_vecs = x_train[indices]

    loss = loss_fn(generated, real_vecs, lambda1=0)

    loss.backward()
    optimizer.step()
    
    optimizer.zero_grad()

    print(loss.item())
    iteration += 1

#torch.save(model.state_dict(), "ot_gen_deep")
#'''

#import pudb; pudb.set_trace()

#### Tracking the Mapping via colors ####
#'''
for i in range(1):
    m1 = uniform.Uniform(-1, 1)

    inputs = m1.sample((128, dim))
    generated = model(inputs).detach().cpu().numpy()

    inputs = inputs.cpu().numpy()

    #circle = []
    
    #for j in range(128):
    #    ran = random.random() * 2 * pi
    #    x_noise = random.random() * 0.5
    #    y_noise = random.random() * 0.5
    #    circle.append([2 * cos(ran) + x_noise, 2 * sin(ran) + y_noise])

    #circle=np.array(circle)

    plt.scatter(inputs[0:128,0], inputs[0:128,1], color='blue')
    #plt.scatter(circle[:,0], circle[:,1], color='purple')
    plt.scatter(x_test[0:128,0], x_test[0:128,1], color='green')
    plt.scatter(generated[:,0], generated[:,1], color='purple')
    
    #'''
    for i in range(128):
        plt.plot([inputs[i,0], generated[i, 0]], [inputs[i,1], generated[i,1]], color='red')

    plt.show()

    plt.scatter(inputs[0:128,0], inputs[0:128,1], color='blue')
    #plt.scatter(circle[:,0], circle[:,1], color='purple')
    plt.scatter(x_test[0:128,0], x_test[0:128,1], color='green')
    plt.scatter(generated[:,0], generated[:,1], color='purple')

    answers = ot_compute_answers(generated, x_test, 128)

    for i in range(128):
        plt.plot([generated[i,0], x_test[answers[i], 0]], [generated[i,1], x_test[answers[i],1]], color='red')

    plt.show()
    '''
#'''

#### Tracking the Mapping via colors ####
'''
inputs = []

for i in range(32):
    for j in range(32):
        inputs.append([i/16 - 1, j/16 - 1])

inputs = torch.Tensor(inputs)
generated = model(inputs).detach().cpu().numpy()

inputs = inputs.cpu().numpy()

colors = []

for i in range(len(inputs)):
    colors.append([(inputs[i][0] + 1) / 2, 0.5, (inputs[i][1] + 1) / 2])

inputs -= 2

#plt.scatter(inputs[:,0], inputs[:,1], c=colors)
plt.scatter(generated[:,0], generated[:,1], c=colors)

plt.show()
#'''

#### Tracking the inverse mapping via colors ####

'''
m1 = uniform.Uniform(-1, 1)

inputs = m1.sample((512, dim))
generated = model(inputs).detach().cpu().numpy()

inputs = inputs.cpu().numpy()

colors = []

x_max = np.max(generated[:,0])
x_min = np.min(generated[:,0])
y_max = np.max(generated[:,1])
y_min = np.min(generated[:,1])

x_mean = (x_max + x_min) / 2
x_range = x_max - x_min
y_mean = (y_max + y_min) / 2
y_range = y_max - y_min


for i in range(512):
    colors.append([(generated[i][0] - x_min) / x_range, 0.5, (generated[i][1] - y_min) / y_range])


inputs -= 2

plt.scatter(inputs[:,0], inputs[:,1], c=colors)
plt.scatter(generated[:,0], generated[:,1], c=colors)

plt.show()
'''