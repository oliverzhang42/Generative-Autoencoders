import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from keras.datasets import mnist


'''Data Processing'''

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    labels = np.zeros((len(y), num_classes))

    for index in y:
        labels[index][y[index]] = 1 #Hacky?

    return labels

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return x

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshapes x_train from (60000, 28, 28) to (60000, 28, 28, 1)
# Reshapes y_train from (60000,) to (60000, 1)
x_train = torch.Tensor(x_train.reshape(60000, 28, 28, 1))
y_train_onehot = torch.Tensor(to_categorical(y_train, num_classes=10))

x_test = torch.Tensor(x_test.reshape(10000, 28, 28, 1))
y_test_onehot = torch.Tensor(to_categorical(y_test, num_classes=10))

'''Building a NN''' 

In_dim = 784
H = 512
D_out = 10

model = torch.nn.Sequential(
    Flatten(),
    torch.nn.Linear(In_dim, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)
loss_fn = torch.nn.MSELoss(reduction='sum')

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

'''Training Process'''

for i in range(500):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train_onehot)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    print("Training Loss: {}".format(loss/len(x_train)))

    if i % 50 == 0:
        y_pred_val = model(x_test)

        '''Calculate Losses'''
        loss_val = loss_fn(y_pred_val, y_test_onehot)
        print("Validation Loss: {}".format(loss_val/len(x_test)))

        '''Calculate Accuracy'''
        _, y_guess_val = y_pred_val.max(1)
        right = torch.eq(y_guess_val, y_test_onehot)

        print("Validation Accuracy: {}".format(right/len(x_test)))