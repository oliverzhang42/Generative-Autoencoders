import numpy as np
import ot
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

def display_img(x, single=False, labels=None, columns=4):
    if single:
        img = x.reshape(28, 28)
        plt.figure()
        if labels:
            plt.title(labels)
        plt.imshow(img, cmap='gray')
    else:
        f, subs = plt.subplots(columns, len(x)//columns, sharex='col', sharey='row')
            
        for i in range(len(x)):
            img = x[i].reshape(28, 28)
            if labels:
                plt.title(labels[i])
            subs[i//columns][i%columns].imshow(img, cmap='gray')
    
    plt.show()

def make_moons(train=60000, test=10000, noise=0.05, shuffle = True):
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

def ot_compute_answers(inputs, encodings, batch_size, verbose=True, ratio=1):
    # TODO: Find some way of getting rid of ratio. Currently it doesn't do anything, 
    # but it's placed here so that OTGen and PTGen fit well together

    answers = []

    for i in range(len(inputs) // batch_size): # VERRY HACKY!!! Right now I have this function handling both the case if you want ot for a whole dataset or a single batch
        # Perhaps split that up later.
        if i % 10 == 0 and verbose:
              print("Batch Number: {}".format(i))

        real_vec = encodings[i*batch_size: (i+1)*batch_size]
        fake_vec = inputs[i*batch_size: (i+1)*batch_size]
          
        a = np.ones((batch_size, ))
        b = np.ones((batch_size, ))
          
        M = ot.dist(fake_vec, real_vec)
        
        '''
        for j in range(batch_size):
            costs = []
            
            for k in range(batch_size):
                costs.append(np.linalg.norm(fake_vec[j] - real_vec[k]))
                #costs.append(np.sum(np.abs(fake_vec[j] - real_vec[k]))) # TODO: Debug the abs & sum & encodings vs real_vec?
            M.append(costs)
        ''' 
        
        M = np.array(M)
        mapping = ot.emd(a, b, M)
          
        for j in range(batch_size):
            index = np.argmax(mapping[j])
            answers.append(index)

    answers = np.array(answers)

    return answers

def pt_compute_answers(encodings, batch_size, inputs=None, distr='uniform', ratio=1):
    if inputs is None:
        if distr == 'uniform':
            inputs = np.random.random(size=encodings.shape)
        elif distr == 'normal':
            inputs = np.random.normal(size=encodings.shape)
        else:
            raise Exception("I don't recognize this distribution: {}".format(distr))
    
    answers = []

    for i in range(len(inputs)):
        min_dist = 100000
        index = 0
        for j in range(int(len(inputs)*ratio)):
            dist = np.linalg.norm(inputs[i] - encodings[j])
            #dist = np.sum(np.abs(inputs[i] - encodings[j]))
          
            if dist < min_dist:
                min_dist = dist
                index = j
          
        answers.append(index)

    answers = np.array(answers)

    return answers

def pt_compute_penalties(inputs, encodings, batch_size):    
    answers = []

    M = ot.dist(inputs, encodings)

    for i in range(len(inputs)):
        answers.append(np.argmin(M[i]))

    '''

    for i in range(len(inputs)):
        min_dist = 100000
        index = 0
        for j in range(len(encodings)):
            dist = np.linalg.norm(inputs[i] - encodings[j])
            #dist = np.sum(np.abs(inputs[i] - encodings[j]))
          
            if dist < min_dist and i != j:
                min_dist = dist
                index = j
          
        answers.append(index)

    answers = np.array(answers)

    '''

    return answers