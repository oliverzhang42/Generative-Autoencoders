# Script
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from GenerativeFramework import GenerativeFramework
from evaluate import evaluate

import matplotlib.pyplot as plt

from utils import *

import os


PATH = '/data/home/oliver/git/generative_autoencoders/tensorflow'
WEIGHT_PATH = os.path.join(PATH, 'weights')
IMG_PATH = os.path.join(PATH, 'images')
DATASET = 'toy_two'
AUTOENCODER = 'identity'
DIM = 2
GENERATOR = 'ot generator'
PRUNING = False
LAYERS = 4
BATCH_SIZE = 500
DISTR = 'normal'
RATIO = 0.1
CLUSTERS = 5
NOIST_INTENSITY = 0
HEADS = 4

import pudb; pudb.set_trace()

#'''
frame = GenerativeFramework(WEIGHT_PATH, IMG_PATH, dataset=DATASET, 
	autoencoder=AUTOENCODER, dim=DIM, generator=GENERATOR, pruning=PRUNING, 
	layers=LAYERS, batch_size=BATCH_SIZE, distr=DISTR, ratio=RATIO, 
	clusters=CLUSTERS, noise_intensity=NOIST_INTENSITY, heads=HEADS)

frame.initialize_generator(recompute=True, file_inputs='test2.npy', file_answers='test2_.npy', activation='linear')

#frame.train_generator(epochs=3, lr=0.003)
#frame.save_generator("OTgen_2Dtest6")
#frame.load_generator("OTgen_2Dtest6")

name="test9.npy"

#frame.generate_images(name)
#evaluate("images/{}".format(name))
#'''

x_test = frame.x_test
x_pred = np.load("images/test9.npy")

plt.scatter(x_test[:,0], x_test[:,1], color='blue')
plt.scatter(x_pred[:1000,0], x_pred[:1000,1], color='green')

plt.show()

#'''
