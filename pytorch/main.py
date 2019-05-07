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

PATH = '/data/home/oliver/git/generative_autoencoders/pytorch'
FOLDER = os.path.join(PATH, "mnist_30_g_2")
DATASET = 'mnist'
DIM = 30
EXTRA_LAYERS = 1
BATCH_SIZE = 128
AE_STEPS = 30000
STEPS = 30000
MODEL = 'generator'
CONV = False

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
ae.load_weights("autoencoder")
'''
if "x_train" in locals():
    ae.train(AE_STEPS, inputs=x_train, test=x_test, lr=0.003, save_images=True)
else:
    ae.train(AE_STEPS, input_load=train_loader, test_load=test_loader, lr=0.003, save_images=True)

#'''
if not 'x_test' in locals():
    test_iter = iter(test_loader)
    x_test = next(test_iter)[0].numpy()

    while True:
        try:
            test_batch = next(test_iter)[0].numpy()
            x_test = np.concatenate((x_test, test_batch), 0)
        except StopIteration:
            break
#'''
#'''
encodings = ae.encode(x_test)
#np.save("real_encodings", encodings)
#encodings = ae.encode(np.load("wae_img.npy"))
#np.save("wae_encodings", encodings)
#encodings = ae.encode(np.load("otgen_img.npy"))
#np.save("otgen_encodings", encodings)



#ae_img = ae.decode(encodings)
#display_img(x_test[0:16], PATH, shape=shape, channels_first=False)
#display_img(ae_img[0:16], PATH, shape=shape, channels_first=False)

if MODEL == 'transporter':
    model = Transporter(encodings, DIM, FOLDER, 128)
elif MODEL == 'generator':
    model = Generator(encodings, DIM, FOLDER, 128)
else:
    raise NotImplementedError

model.load_weights(MODEL)
#model.train(STEPS, lr=0.0001)

#fake_distr = model.generate(num_batches=1)
#generated_images = ae.decode(fake_distr)

#np.save("ottrans_img", generated_images)

'''
m1 = uniform.Uniform(-1, 1)

for k in range(100):
    noise = m1.sample((9,100)).cpu().numpy()

    distr = []
    current = noise[0]

    for i in range(8):
        movement = (noise[i+1] - noise[i])/8
        for j in range(8):
            distr.append(deepcopy(current))
            if j != 7:
                current += movement
    
    fake_distr = model.model(torch.Tensor(distr)).detach().cpu().numpy()
    fake_img = ae.decode(fake_distr)

    save_image(torch.Tensor(fake_img)[0:64], MODEL + '_interpolation.png')

    inter = []
    current = fake_distr[0]

    for i in range(8):
        movement = (fake_distr[8*(i+1)-1] - fake_distr[8*i])/8
        for j in range(8):
            inter.append(deepcopy(current))
            if j != 7:
                current += movement
    
    fake_img2 = ae.decode(inter)

    save_image(torch.Tensor(fake_img2)[0:64], MODEL + '_interpolation_bad.png')
'''

#'''
fake_distr = model.generate(num_batches=1)
fake_img = ae.decode(fake_distr)

save_image(torch.Tensor(fake_img)[0:64].view(64, 1, 28, 28), 'otgen_mnist.png')

#for i in range(len(fake_img)//16):
#   display_img(fake_img[i*16:(i+1)*16], PATH, shape=shape, channels_first=True)

'''
for i in range(100):
    x, y = np.random.random(size=(2, 100))*2-1
    gen = model.interpolate(x, y, 16)
    fake_img = ae.decode(gen)

    display_img(fake_img[0:16], PATH, shape=shape, channels_first=True)

#'''