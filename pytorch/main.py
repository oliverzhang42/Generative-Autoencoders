# Script

from autoencoder import Autoencoder
from generator import Generator
from transporter import Transporter

from keras.datasets import mnist, fashion_mnist, cifar10
from torchvision import transforms
import torchvision

from utils import *
import os

torch.set_default_tensor_type('torch.cuda.FloatTensor')

PATH = '/data/home/oliver/git/generative_autoencoders/pytorch'
FOLDER = os.path.join(PATH, "faces_100_g_convtranspose")
DATASET = 'faces'
DIM = 100
EXTRA_LAYERS = 3
BATCH_SIZE = 64
AE_STEPS = 100000
STEPS = 10000
MODEL = 'generator'
CONV = True

import pudb; pudb.set_trace()

if DATASET == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_train / 255
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
    shape = (32, 32, 3)
elif DATASET == 'faces':
    if os.path.exists("faces.npy"):
        data = np.load("faces.npy")
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.ToTensor(),
            transforms.Scale(64),
    #        transforms.Normalize(mean=(0.5, 0.5, 0.5))
        ])
        data_dir = os.path.join(PATH, 'celebA')
        dset = torchvision.datasets.ImageFolder(data_dir, transform)
        train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False)

        data = []

        for x_, _ in train_loader:
            if len(data) == 0:
                data = x_
            else:
                data = torch.cat((data, x_), 0)

            if len(data) > 20000:
                break

        data = data.numpy()
        np.save("faces.npy", data)

    np.random.shuffle(data)

    x_train, x_test = data[0:14000], data[14000:]

    shape = (64, 64, 3)
else:
    raise NotImplementedError


ae = Autoencoder(x_train, x_test, DIM, FOLDER, BATCH_SIZE, EXTRA_LAYERS, CONV)
#ae.load_weights("autoencoder")
ae.train(AE_STEPS, lr=0.003, images=True)
encodings = ae.encode(x_train)

ae_img = ae.decode(encodings)
#display_img(x_train[0:16], PATH, shape=shape, channels_first=False)
#display_img(ae_img[0:16], PATH, shape=shape, channels_first=False)

if MODEL == 'transporter':
    model = Transporter(encodings, DIM, FOLDER, 4*BATCH_SIZE)
elif MODEL == 'generator':
    model = Generator(encodings, DIM, FOLDER, 4*BATCH_SIZE)
else:
    raise NotImplementedError

#model.load_weights("generator")
model.train(STEPS)

fake_distr = model.generate()
fake_img = ae.decode(fake_distr)

display_img(fake_img[0:16], PATH, shape=shape, channels_first=False)