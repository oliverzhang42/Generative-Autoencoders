import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt
from keras.datasets import mnist

from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''Data Processing'''

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    labels = np.zeros((len(y), num_classes))

    for i in range(len(y)):
        labels[i][y[i]] = 1 #Hacky?

    return labels

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshapes x_train from (60000, 28, 28) to (60000, 28, 28, 1)
# Reshapes y_train from (60000,) to (60000, 1)
x_train = torch.Tensor(x_train.reshape(60000, 784))
y_train_onehot = torch.Tensor(to_categorical(y_train, num_classes=10))

x_test = torch.Tensor(x_test.reshape(10000, 784))
y_test_onehot = torch.Tensor(to_categorical(y_test, num_classes=10))
y_test = torch.Tensor(y_test)

In_dim = 784
H = 512
dim = 10

block1 = torch.nn.Sequential(
    torch.nn.Linear(In_dim, H),
    torch.nn.LeakyReLU(),
    torch.nn.BatchNorm1d(H)
)

block2 = torch.nn.Sequential(
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU(),
    torch.nn.BatchNorm1d(H)
)

block3 = torch.nn.Sequential(
    torch.nn.Linear(H, dim),
    torch.nn.Sigmoid()
)

block4 = torch.nn.Sequential(
    torch.nn.Linear(dim, H),
    torch.nn.LeakyReLU(),
    torch.nn.BatchNorm1d(H)
)

block5 = torch.nn.Sequential(
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU(),
    torch.nn.BatchNorm1d(H)
)

block6 = torch.nn.Sequential(
    torch.nn.Linear(H, In_dim),
    torch.nn.Sigmoid()
)

encoder = torch.nn.Sequential(
    block1,
    block2,
    block3
)

decoder = torch.nn.Sequential(
    block4,
    block5,
    block6
)

model = torch.nn.Sequential(
    encoder,
    decoder
)

model.load_state_dict(torch.load("generator2"))


