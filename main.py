# Script
from GenerativeFramework import GenerativeFramework
from evaluate import evaluate

from utils import *

import os

PATH = '/data/home/oliver/git/generative_autoencoders'
WEIGHT_PATH = os.path.join(PATH, 'weights')
IMG_PATH = os.path.join(PATH, 'images')
DATASET = 'mnist'
AUTOENCODER = 'dense'
DIM = 10
GENERATOR = 'ot transport'
PRUNING = False
LAYERS = 4
BATCH_SIZE = 500
DISTR = 'uniform'
RATIO = 0.1
CLUSTERS = 5
NOIST_INTENSITY = 0

import pudb; pudb.set_trace()
#'''
frame = GenerativeFramework(WEIGHT_PATH, IMG_PATH, dataset=DATASET, 
	autoencoder=AUTOENCODER, dim=DIM, generator=GENERATOR, pruning=PRUNING, 
	layers=LAYERS, batch_size=BATCH_SIZE, distr=DISTR, ratio=RATIO, 
	clusters=CLUSTERS, noise_intensity=NOIST_INTENSITY)

frame.load_autoencoder("AE1")
#frame.train_autoencoder(epochs=10)
#frame.save_autoencoder("AE2")
#frame.evaluate_autoencoder()

frame.initialize_generator(recompute=True, file_inputs='ot_map_inputs_norm.npy', file_answers='ot_map_answers_norm.npy')
frame.train_generator(epochs=10)
frame.save_generator("OTMap1Norm")

#frame.load_generator("OTMap1")

name="AEOT_norm.npy"

frame.generate_images(name)
evaluate("images/{}".format(name))
#'''

display(os.path.join(IMG_PATH, name))