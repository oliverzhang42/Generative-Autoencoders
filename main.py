import argparse
from autoencoder import Autoencoder
from copy import deepcopy
from generator import Generator
import os

from torchvision import transforms
import torchvision
from torchvision.utils import save_image
from torch.distributions import normal, uniform
from torch.utils.data import DataLoader

from transporter import Transporter
from utils import *


torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=bool, default=True,
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
parser.add_argument('--load_a', type=bool, default=False,
                    help='Whether to load the autoencoder (default: False)')

parser.add_argument('--batchsize_l', type=int, default=128,
                    help='input batch size for the latent space model (default: 128)')
parser.add_argument('--latent_distr', default='uniform',
                    help='input batch size for the latent space model (default: uniform)')
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
AE_LOAD = args.load_a

BATCH_SIZE_GEN = args.batchsize_l
DISTR = args.latent_distr
GEN_LOAD = args.load_l
STEPS = args.steps_l
MODEL = args.model

# Load the right dataset
if DATASET == 'mnist':
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=BATCH_SIZE, shuffle=True)

    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    mnist_testloader = DataLoader(mnist_testset, batch_size=BATCH_SIZE, shuffle=True)

    x_train = unload(mnist_trainloader)
    x_test = unload(mnist_testloader)

    x_train = x_train / 255
    x_test = x_test / 255
    shape = (1, 28, 28)
elif DATASET == 'fashion_mnist':
    fashion_trainset = datasets.Fashion_MNIST(root='./data', train=True, download=True, transform=None)
    fashion_trainloader = DataLoader(fashion_trainset, batch_size=BATCH_SIZE, shuffle=True)

    fashion_testset = datasets.Fashion_MNIST(root='./data', train=False, download=True, transform=None)
    fashion_testloader = DataLoader(fashion_testset, batch_size=BATCH_SIZE, shuffle=True)

    x_train = unload(fashion_trainloader)
    x_test = unload(fashion_testloader)

    x_train = x_train / 255
    x_test = x_test / 255
    shape = (1, 28, 28)
elif DATASET == 'cifar10':
    cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    cifar10_trainloader = DataLoader(cifar10_trainset, batch_size=BATCH_SIZE, shuffle=True)

    cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    cifar10_testloader = DataLoader(cifar10_testset, batch_size=BATCH_SIZE, shuffle=True)
    
    x_train = unload(cifar10_trainloader)
    x_test = unload(cifar10_testloader)
    
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
    train_size = 7*dset_size//10

    train_set, test_set = torch.utils.data.random_split(dset, (train_size, dset_size-train_size))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

    shape = (3, 64, 64)
else:
    raise NotImplementedError

# Train the autoencoder
ae = Autoencoder(shape, DIM, FOLDER, BATCH_SIZE, CONV)
if AE_LOAD:
    ae.load_weights("autoencoder")
else:
    if "x_train" in locals():
        ae.train(AE_STEPS, x_train, x_test, lr=0.003)
    else:
        ae.train_iter(AE_STEPS, train_loader, test_loader, lr=0.003)

# Prepare the Latent Space Model
if not 'x_test' in locals():
    x_test = unload(test_loader)

encodings = ae.encode(x_test)

if MODEL == 'transporter':
    model = Transporter(encodings, DISTR, FOLDER, BATCH_SIZE_GEN)
elif MODEL == 'generator':
    model = Generator(encodings, DISTR, FOLDER, BATCH_SIZE_GEN)
else:
    raise NotImplementedError

# Train the Latent Space Model
if GEN_LOAD:
    model.load_weights(MODEL)
else:
    model.train(STEPS, lr=0.0001)

import pudb; pudb.set_trace()

# Display Results
fake_distr = model.generate(batches=1)
fake_img = ae.decode(fake_distr)
fake_img = np.reshape(fake_img, ((BATCH_SIZE_GEN,) + shape))

save_image(torch.Tensor(fake_img), os.path.join(FOLDER, "final.png"))

channels_last = np.rollaxis(fake_img[0:16], 1, 4)
display_img(channels_last, columns=4)
