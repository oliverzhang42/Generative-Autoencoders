# Script

from autoencoder import Autoencoder
from generator import Generator
from transporter import Transporter

from keras.datasets import mnist, fashion_mnist, cifar10
from torchvision import transforms
import torchvision
from torchvision.utils import save_image

from utils import *
import os

torch.set_default_tensor_type('torch.cuda.FloatTensor')

PATH = '/data/home/oliver/git/generative_autoencoders/pytorch'
FOLDER = os.path.join(PATH, "faces_100_g_diversity")
DATASET = 'faces'
DIM = 100
EXTRA_LAYERS = 3
BATCH_SIZE = 64
AE_STEPS = 30000
STEPS = 30000
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
ae.load_weights("autoencoder")
#if "x_train" in locals():
#    ae.train(AE_STEPS, inputs=x_train, test=x_test, lr=0.003, save_images=True)
#else:
#    ae.train(AE_STEPS, input_load=train_loader, test_load=test_loader, lr=0.003, save_images=True)

#'''
test_iter = iter(test_loader)
x_test = next(test_iter)[0].numpy()

while True:
    try:
        test_batch = next(test_iter)[0].numpy()
        x_test = np.concatenate((x_test, test_batch), 0)
    except StopIteration:
        break


#'''

encodings = ae.encode(x_test)

#ae_img = ae.decode(encodings)
#display_img(x_test[0:16], PATH, shape=shape, channels_first=True)
#display_img(ae_img[0:16], PATH, shape=shape, channels_first=True)

if MODEL == 'transporter':
    model = Transporter(encodings, DIM, FOLDER, 4096)
elif MODEL == 'generator':
    model = Generator(encodings, DIM, FOLDER, 4096)
else:
    raise NotImplementedError

model.load_weights("generator")
#model.train(STEPS, lr=0.0001)

fake_distr = model.generate()
fake_img = ae.decode(fake_distr)

save_image(fake_img[0:64], MODEL + '.png')

for i in range(len(fake_img)//16):
   display_img(fake_img[i*16:(i+1)*16], PATH, shape=shape, channels_first=True)

#for i in range(100):
#    x, y = np.random.random(size=(2, 100))*2-1
#    gen = model.interpolate(x, y, 16)
#    fake_img = ae.decode(gen)

#    display_img(fake_img[0:16], PATH, shape=shape, channels_first=True)