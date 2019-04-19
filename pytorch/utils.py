import numpy as np
import ot
import os
import matplotlib.pyplot as plt
from sklearn import datasets
import torch
from math import *

plt.rcParams["figure.figsize"] = [9.0, 6.0]
plt.rcParams["axes.grid"] = False

def one_hot(labels, num_classes):
    """
    https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/25
    """
    y = torch.eye(num_classes) 
    return y[labels]

def display(file_path):
    images = np.load(file_path)
    indices = np.random.choice(range(len(images)), size=16)
    display_img(images[indices])

# Needs fixing

def display_img(x, path, show=True, shape=(28, 28), single=False, index=0, labels=None, columns=4, channels_first=True):
    if single:
        if not channels_first:
            img = np.moveaxis(x, 0, -1)
        img = x.reshape(shape)
        plt.figure()
        if labels:
            plt.title(labels)
        plt.imshow(img, cmap='gray')
    else:
        fig=plt.figure(figsize=(8, 8))
        rows=len(x)//columns
            
        for i in range(len(x)):
            if not channels_first:
                img = x[i].reshape((shape[1], shape[2], shape[3]))
                img = np.moveaxis(img, 0, -1)
            else:
                img = x[i].reshape(shape)
            if labels:
                plt.title(labels[i])
            
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(img, cmap='gray')

#            if show:
#                subs[i//columns][i%columns].imshow(img, cmap='gray')
    
    if show:
        plt.show()
    else:
        file_path = os.path.join(path, "image{}.png".format(index))
        fig.savefig(file_path)
        fig.clf()
        plt.close(fig)

def display_points(inputs, answers, path, show=True, index=0, lines=False):
    plt.scatter(inputs[:,0], inputs[:,1], color='blue')
    plt.scatter(answers[:,0], answers[:,1], color='green')

    for i in range(512):
        if lines:
            plt.plot([inputs[i][0], answers[i][0]], [inputs[i][1], answers[i][1]], color='red')
    
    if show:
        plt.show()
    else:
        file_path = os.path.join(path, "image{}.png".format(index))
        plt.savefig(file_path)
        plt.clf()

def make_moons(train=60000, test=10000, noise=0.05, shuffle=True):
    moons_train, train_labels = datasets.make_moons(n_samples=train, noise=noise)
    moons_test, test_labels = datasets.make_moons(n_samples=test, noise=noise)

    if not shuffle:
        moons_train1 = []
        moons_train2 = []

        for i in range(len(moons_train)):
            if train_labels[i] == 0:
                moons_train1.append(moons_train[i])
            else:
                moons_train2.append(moons_train[i])

        moons_test1 = []
        moons_test2 = []

        for i in range(len(moons_test)):
            if train_labels[i] == 0:
                moons_test1.append(moons_test[i])
            else:
                moons_test2.append(moons_test[i])

        return np.concatenate((moons_train1, moons_train2)), np.concatenate((moons_test1, moons_test2))

    return moons_train, moons_test

def make_circles(train=60000, test=10000, noise=0.05):
    moon_train, _ = datasets.make_circles(n_samples=train, noise=noise, factor=0.5)
    moon_test, _ = datasets.make_circles(n_samples=test, noise=noise, factor=0.5)
    return moon_train, moon_test

def toy_two(train=60000, test=10000, noise_scale=0.1):
    d = {0: (1, 0), 1: (-1, 0)}

    x_train = []
    x_test = []

    for i in range(60000):
        noise = noise_scale * np.random.normal(size=2)
        noisex = noise[0]
        noisey = noise[1]
        
        center = d[i%2]
        l = [center[0] + noisex, center[1] + noisey]
        x_train.append(l)

    for i in range(10000):
        noise = noise_scale * np.random.normal(size=2)
        noisex = noise[0]
        noisey = noise[1]
      
        center = d[i%2]
        l = [center[0] + noisex, center[1] + noisey]
        x_test.append(l)

    return np.array(x_train), np.array(x_test)

def toy_eight(train=60000, test=10000, noise_scale=0.1):
    x_train = []
    x_test = []

    d = {0: (2, 0), 1: (sqrt(2), sqrt(2)), 2: (0, 2), 3: (-sqrt(2), sqrt(2)), \
         4: (-2, 0), 5: (-sqrt(2), -sqrt(2)), 6: (0, -2), 7: (sqrt(2), -sqrt(2))}

    for i in range(60000):
        noise = 0.1 * np.random.normal(size=2)
        noisex = noise[0]
        noisey = noise[1]
        
        center = d[i%8]
        l = [center[0] + noisex, center[1] + noisey]
        x_train.append(l)

    for i in range(10000):
        noise = 0.1 * np.random.normal(size=2)
        noisex = noise[0]
        noisey = noise[1]
      
        center = d[i%8]
        l = [center[0] + noisex, center[1] + noisey]
        x_test.append(l)

    return np.array(x_train), np.array(x_test)

def ot_compute_answers(inputs, encodings):
    '''
    Computes optimal transport for one batch.
    '''

    batch_size = len(inputs)

    a = np.ones((batch_size, ))
    b = np.ones((batch_size, ))
      
    M = ot.dist(inputs, encodings)
    M = np.array(M)

    mapping = ot.emd(a, b, M)

    answers = []
      
    for j in range(batch_size):
        index = np.argmax(mapping[j])
        answers.append(index)

    answers = np.array(answers)

    return answers

class View(torch.nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def show_obj():
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass