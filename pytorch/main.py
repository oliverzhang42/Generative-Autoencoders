# Script
'''
Things to Fix:
1. display_image (whether it's flat or not, whether to roll axis or not)
(also all usages of display_image) (DONE)
2. Encoder and decode, how to deal with loaders (DONE)
3. GPU if you have one, otherwise, no GPU 
4. turn main into an argument taking thing
5. Check how to store checkpoints and how to store images well
6. Interpolation better
7. Have presets for MNIST and CelebA for instance
'''

import argparse
from autoencoder import Autoencoder
from copy import deepcopy
from generator import Generator
from keras.datasets import mnist, fashion_mnist, cifar10
import os

from torchvision import transforms
import torchvision
from torchvision.utils import save_image
from torch.distributions import normal, uniform

from transporter import Transporter
from utils import *


torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=bool, action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--dataset', default='mnist',
                    help='dataset name (default: mnist)')
parser.add_argument('--folder', default='.',
                    help='folder name to put everything in')

parser.add_argument('--steps_a', type=int, default=15000,
                    help='number of steps to take (default: 15000)')
parser.add_argument('--batchsize_a', type=int, default=128,
                    help='input batch size for autoencoder (default: 128)')
parser.add_argument('--conv', type=bool, default=False,
                    help='Is your autoencoder convolutional (default: false)')
parser.add_argument('--dim', type=int, default=30,
                    help='dimension of autoencoder (default: 30)')
parser.add_argument('--extra-layers', type=int, default=1,
                    help='Extra layers in the autoencoder (default: 1)')
parser.add_argument('--load_a', type=bool, default=False,
                    help='Whether to load the autoencoder (default: False)')

parser.add_argument('--batchsize_l', type=int, default=128,
                    help='input batch size for the latent space model (default: 128)')
parser.add_argument('--load_l', type=bool, default=False,
                    help='Whether to load the latent space model (default: False)')
parser.add_argument('--steps_l', type=int, default=15000,
                    help='number of steps your latent space model takes (default: 15000)')
parser.add_argument('--model', default='generator',
                    help='Is your latent space model a transporter or a generator (default: generator)')

args = parser.parse_args()

CUDA = args.cuda
DATASET = args.dataset
FOLDER = args.folder

AE_STEPS = args.steps_a
BATCH_SIZE = args.batchsize_a
CONV = args.conv
DIM = args.dim
EXTRA_LAYERS = args.extra-layers
AE_LOAD = args.load_a

BATCH_SIZE_GEN = args.batchsize_l
GEN_LOAD = args.load_l
STEPS = args.steps_l
MODEL = args.model

import pudb; pudb.set_trace()

if DATASET == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    shape = (28, 28)
elif DATASET == 'fashion_mnist':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    shape = (28, 28)
elif DATASET == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    shape = (3, 32, 32)
elif DATASET == 'faces':
    transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    data_dir = os.path.join(PATH, 'celebA')
    dset = torchvision.datasets.ImageFolder(data_dir, transform)
    
    dset_size = len(dset)
    train_size = 99*dset_size//100

    train_set, test_set = torch.utils.data.random_split(dset, (train_size, dset_size-train_size))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

    shape = (3, 64, 64)
else:
    raise NotImplementedError


ae = Autoencoder(shape, DIM, FOLDER, BATCH_SIZE, EXTRA_LAYERS, CONV)
if AE_LOAD:
    ae.load_weights("autoencoder")
else:
    if "x_train" in locals():
        ae.train(AE_STEPS, inputs=x_train, test=x_test, lr=0.003, save_images=True)
    else:
        ae.train(AE_STEPS, input_load=train_loader, test_load=test_loader, lr=0.003, save_images=True)

if not 'x_test' in locals():
    x_test = unload(test_loader)

encodings = ae.encode(x_test)

if MODEL == 'transporter':
    model = Transporter(encodings, DIM, FOLDER, BATCH_SIZE_GEN)
elif MODEL == 'generator':
    model = Generator(encodings, DIM, FOLDER, BATCH_SIZE_GEN)
else:
    raise NotImplementedError

if GEN_LOAD:
    model.load_weights(MODEL)
else:
    model.train(STEPS, lr=0.0001)

fake_distr = model.generate(num_batches=1)
fake_img = ae.decode(fake_distr)

fake_img = np.reshape(fake_img, ((BATCH_SIZE_GEN,) + shape))