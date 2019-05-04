import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal, uniform

import numpy as np

import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''Data Processing'''

x_train, x_test = toy_eight()

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


model.load_state_dict(torch.load("./Transport Toy Eight/ot_trans_deep2"))

#import pudb; pudb.set_trace()

'''
lr = 0.0003
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_fn = torch.nn.L1Loss()

m1 = uniform.Uniform(-1, 1)

for i in range(10000):
    inputs = m1.sample((512, dim))
    indices = np.random.choice(range(60000), size=512)
    
    real_vecs = x_train[indices]

    answer_indices = ot_compute_answers(inputs.cpu().numpy(), real_vecs.cpu().numpy(), 512, verbose=False)
    answers = real_vecs[answer_indices]

    generated = model(inputs)

    loss = loss_fn(generated, answers)

    loss.backward()
    optimizer.step()
    
    optimizer.zero_grad()

    print(loss.item())
    
    if i % 100 == 0:
        display_points(generated.detach().cpu(), answers.cpu(), ".", show=False, index=i//100)

torch.save(model.state_dict(), "ot_trans_deep2")
#'''

#import pudb; pudb.set_trace()

#'''
for i in range(1):
    num = 175
    m1 = uniform.Uniform(-1, 1)

    inputs = m1.sample((num, dim))
    generated = model(inputs).detach().cpu().numpy()

    inputs = inputs.cpu().numpy()

    #answers = ot_compute_answers(generated, x_test, num)

    #for j in range(num):
    #    plt.plot([inputs[j,0], x_test[answers[j], 0]], [inputs[j,1], x_test[answers[j],1]], color='red', zorder=1)

    #plt.scatter(inputs[0:num,0], inputs[0:num,1], color='blue', zorder=0)
    plt.scatter(x_test[0:num,0], x_test[0:num,1], color='blue')
    plt.scatter(generated[:,0], generated[:,1], color='red')



    plt.show()
#'''

#### Tracking the Mapping via colors ####
'''
inputs = torch.rand((512, 2))
generated = model(inputs).detach().cpu().numpy()

inputs = inputs.cpu().numpy()

colors = []

for i in range(512):
    colors.append([inputs[i][0], 0.5, inputs[i][1]])

plt.scatter(inputs[:,0], inputs[:,1], c=colors)
plt.scatter(generated[:,0], generated[:,1], c=colors)

plt.show()
#'''