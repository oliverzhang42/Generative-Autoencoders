from math import *
import numpy as np
import ot
from ot import gpu
import os
import matplotlib.pyplot as plt
import torch

plt.rcParams["figure.figsize"] = [9.0, 6.0]
plt.rcParams["axes.grid"] = False

def one_hot(labels, num_classes):
    '''
    Converts labels to one_hot encoding.
    '''
    y = torch.eye(num_classes) 
    return y[labels]

def display_img(x, columns=4):
    '''
    Displays images in a grid using matplotlib

    x: numpy array containing images to display.
    Must be formatted channels back.
    '''

    # If x has 3 color channels, then we're fine.
    # However, plt.imshow doesn't work well with 1 channel
    # So we reshape to remove the channel if that ever happens.
    if len(x[0][0][0]) == 1:
        x = np.reshape(x, x.shape[0:3])

    rows=len(x)//columns
    fig=plt.figure(figsize=(rows, columns))

    for i in range(len(x)):
        img = x[i]

        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img, cmap='gray')

    plt.show()

def save_points(inputs, answers, path, name='image', lines=False):
    '''
    Saves an img of points in a scatterplot using matplotlib

    inputs: (list or np array) Points to plot
    answers: (list or np array) Points to plot
    path: (str) Where to save the points
    name: (str) Name of the image file
    lines: (bool) Whether to draw lines from inputs[i] to answers[i]
    '''
    plt.scatter(inputs[:,0], inputs[:,1], color='blue')
    plt.scatter(answers[:,0], answers[:,1], color='green')

    if lines:
        for i in range(512):
            plt.plot([inputs[i][0], answers[i][0]], [inputs[i][1], answers[i][1]], color='red')

    file_path = os.path.join(path, "{}.png".format(name))
    plt.savefig(file_path)
    plt.clf()

def make_moons(train=60000, test=10000, noise=0.05):
    '''
    Makes the moons datasets. See
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
    '''
    moons_train, train_labels = sklearn.datasets.make_moons(n_samples=train, noise=noise)
    moons_test, test_labels = sklearn.datasets.make_moons(n_samples=test, noise=noise)
    return moons_train, moons_test

def make_circles(train=60000, test=10000, noise=0.05):
    '''
    Makes the circles datasets. See
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html
    '''
    moon_train, _ = sklearn.datasets.make_circles(n_samples=train, noise=noise, factor=0.5)
    moon_test, _ = sklearn.datasets.make_circles(n_samples=test, noise=noise, factor=0.5)
    return moon_train, moon_test

def two_cluster(train=60000, test=10000, noise_scale=0.1):
    '''
    Makes a toy dataset with two clusters. Clusters are centered
    at (1,0), (-1,0), are normal, and have a stdev = noise_scale.
    '''
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

def eight_cluster(train=60000, test=10000, noise_scale=0.1):
    '''
    Makes a toy dataset with eight clusters. Clusters are centered
    evenly around the origin two away from it. Clusters are normal
    and stdev of clusters = noise_scale
    '''
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

def optimal_transport(inputs, encodings, log=False):
    '''
    Computes optimal transport for one batch.

    inputs (np array): Inputs of the optimal transport mapping
    encodings (np array): Outputs of the optimal transport mapping

    return answers: Index array. encodings[answers[0]] corresponds with
    inputs[0]
    '''

    batch_size = len(inputs)

    a = np.ones((batch_size, ))
    b = np.ones((batch_size, ))
      
    M = ot.dist(inputs, encodings)
    M = np.array(M)

    if log:
        mapping, log = ot.emd(a, b, M, numItermax=1000000)
    else:
        mapping = ot.emd(a, b, M, numItermax=1000000)

    answers = []
      
    for j in range(batch_size):
        index = np.argmax(mapping[j])
        answers.append(index)

    answers = np.array(answers)

    if log:
        return answers, log
    else:
        return answers

def sinkhorn_transport(inputs, encodings, reg):
    '''
    '''

    batch_size = len(inputs)

    a = np.ones((batch_size, ))
    b = np.ones((batch_size, ))

    import pudb; pudb.set_trace()

    M = gpu.dist(inputs, encodings)
    #M = 10*M

    mapping = gpu.sinkhorn(a, b, M, reg)

    answers = []
    
    for j in range(batch_size):
        index = np.argmax(mapping[j])
        answers.append(index)

    answers = np.array(answers)

    return answers

def unload(dataloader):
    '''
    Unloads the dataloader fully, into a numpy array.
    '''
    data_iter = iter(dataloader)
    x = next(data_iter)[0].numpy()

    while True:
        try:
            x_batch = next(data_iter)[0].numpy()
            x = np.concatenate((x, x_batch), 0)
        except StopIteration:
            break

    return x

def reg_stdev(generated, answers, lambda_=1):
    gen_stdev = generated.std(0)
    ans_stdev = answers.std(0)

    return lambda_ * torch.abs(gen_stdev - ans_stdev).mean()

def reg_cyclic(generated, answers, lambda_=1):
    rand1 = np.random.choice(range(generated.shape[0]), 1000)
    rand2 = np.random.choice(range(generated.shape[0]), 1000)
    rand3 = np.random.choice(range(generated.shape[0]), 1000)
    rand4 = np.random.choice(range(generated.shape[0]), 1000)

    # total implies .mean(), separated_along_dimension implies .mean(0)
    generated_cyclic = torch.abs(generated[rand1] - generated[rand2]).mean(0)
    real_cyclic = torch.abs(answers[rand3] - answers[rand4]).mean(0)

    return lambda_ * torch.abs(generated_cyclic - real_cyclic).mean()

def unbounded_cyclic(generated, lambda_=1):
    rand1 = np.random.choice(range(generated.shape[0]), 1000)
    rand2 = np.random.choice(range(generated.shape[0]), 1000)

    generated_cyclic = torch.abs(generated[rand1] - generated[rand2]).mean(0)

    return - lambda_ * torch.abs(generated_cyclic).mean()

class View(torch.nn.Module):
    '''
    With this, we can reshape tensors from within
    the middle of a Sequential object
    '''
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
