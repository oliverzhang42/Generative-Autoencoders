import argparse
from copy import deepcopy
from generator import Generator
import os

from torch.distributions import normal, uniform

from transporter import Transporter
from utils import *


torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='moons',
                    help='dataset name (default: moons)')
parser.add_argument('--folder', default='.',
                    help='folder name to put everything in')

parser.add_argument('--batchsize', type=int, default=128,
                    help='input batch size for the latent space model (default: 128)')
parser.add_argument('--latent_distr', default='uniform',
                    help='input batch size for the latent space model (default: uniform)')
parser.add_argument('--load', type=bool, default=False,
                    help='Whether to load the latent space model (default: False)')
parser.add_argument('--steps', type=int, default=15000,
                    help='number of steps your latent space model takes (default: 15000)')
parser.add_argument('--model', default='generator',
                    help='Is your latent space model a transporter or a generator (default: generator)')

args = parser.parse_args()

DATASET = args.dataset
FOLDER = args.folder

if not os.path.isdir(FOLDER):
    print("Making Folder: {}".format(FOLDER))
    os.mkdir(FOLDER)

BATCH_SIZE_GEN = args.batchsize
DISTR = args.latent_distr
GEN_LOAD = args.load
STEPS = args.steps
MODEL = args.model

# Load the right dataset
if DATASET == 'moons':
    latent, test = make_moons()
elif DATASET == 'two_cluster':
    latent, test = two_cluster()
elif DATASET == 'eight_cluster':
    latent, test = eight_cluster()
elif DATASET == 'circles':
    latent, test = make_circles()
else:
    raise NotImplementedError

# Prepare the Latent Space Model
if MODEL == 'transporter':
    model = Transporter(latent, DISTR, FOLDER, BATCH_SIZE_GEN)
elif MODEL == 'generator':
    model = Generator(latent, DISTR, FOLDER, BATCH_SIZE_GEN)
else:
    raise NotImplementedError

# Train the Latent Space Model
if GEN_LOAD:
    model.load_weights(MODEL)
else:
    model.train(STEPS, lr=0.0001, images=True)

# Evaluate
model.evaluate()

# Display Results
fake_distr = model.generate(batches=1)
save_points(fake_distr, latent[0:BATCH_SIZE_GEN], FOLDER, name='final')
