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

frame = GenerativeFramework(WEIGHT_PATH, IMG_PATH, dataset=DATASET, autoencoder=AUTOENCODER, 
    dim=DIM, generator=GENERATOR, pruning=PRUNING, layers=LAYERS, batch_size=BATCH_SIZE)

frame.load_autoencoder("AE1")
#frame.train_autoencoder(epochs=10)
#frame.save_autoencoder("AE2")
#frame.evaluate_autoencoder()

frame.initialize_generator(recompute=False, file_inputs='ot_map_inputs.npy', file_answers='ot_map_answers.npy')
#frame.train_generator(epochs=7)
#frame.save_generator("OTMap1")

frame.load_generator("OTMap1")

#frame.generate_images("AEOT")
#evaluate("images/AEOT.npy")

display(os.path.join(IMG_PATH, 'AEOT.npy'))