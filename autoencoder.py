import math
import numpy as np
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''
Autoencoder class. Contains an autoencoder made via pytorch. This autoencoder
can either be convolution or not.
'''
class Autoencoder():
    def __init__(self, img_shape, dim=10, path='.', batch_size=512, conv=False):
        '''
        Constructor of Autoencoder.

        img_shape: (tuple) The shape of the image. Channels first.
        dim: (int) The dimension of the bottleneck layer of the autoencoder.
        path: (str) The path to store all weights/images
        batch_size: (int) batch size
        conv: (bool) Whether we're using the convolutional model or not.
        '''
        if not os.path.isdir(path):
            print("Making folder at path: {}".format(path))
            os.mkdir(path)

        self.batch_size = batch_size
        self.path = path
        self.dim = dim
        self.img_shape = img_shape

        if conv:
            self.input_shape = img_shape
            self.create_conv_model()
        else:
            flatten = np.prod(img_shape)
            self.input_shape = (flatten,)
            self.create_model()

    def create_model(self):
        '''
        Creates a pytorch autoencoder model stored under self.model.
        
        We use 3 layers in each of the encoder and decoder, and a hidden layer
        neuron size of 512. BatchNorm and LeakyReLU are also used as well as a
        Sigmoid activation at the end of each of the encoder and decoder.
        '''

        H = 512
        input_shape = self.input_shape

        # Create the encoder
        encoder = []

        encoder.append(nn.Linear(input_shape[0], H))
        encoder.append(nn.BatchNorm1d(H))
        encoder.append(nn.LeakyReLU())
        encoder.append(nn.Linear(H, H))
        encoder.append(nn.BatchNorm1d(H))
        encoder.append(nn.LeakyReLU())
        encoder.append(nn.Linear(H, self.dim))
        encoder.append(nn.Sigmoid())
        
        self.encoder = nn.Sequential(*encoder)

        # Create the decoder
        decoder = []

        decoder.append(nn.Linear(self.dim, H))
        decoder.append(nn.BatchNorm1d(H))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.Linear(H, H))
        decoder.append(nn.BatchNorm1d(H))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.Linear(H, input_shape[0]))
        decoder.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder)
 
        # Finalize the autoencoder
        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )

    def create_conv_model(self):
        '''
        Creates a pytorch convolutional autoencoder model and stores it under
        self.model. This conv model has the same architecture as in the paper
        "Wasserstein Autoencoders" by Tolstikhin et al.

        Code is taken from:
        https://github.com/1Konny/WAE-pytorch/blob/master/model.py 
        '''

        H = 512
        input_shape = self.input_shape

        assert self.input_shape[1] == self.input_shape[2], \
            "We assume the images are square. (Also, channels are first)"
        sidelength = self.input_shape[1]

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View(-1, 1024*4*4),                                 # B, 1024*4*4
            nn.Linear(1024*4*4, self.dim),                         # B, z_dim
            #nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, 1024*8*8),                           # B, 1024*8*8
            View(-1, 1024, 8, 8),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 1),                       # B,   nc, 64, 64
            nn.Sigmoid()
        )

        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )

    def train(self, steps, inputs, test, lr=0.001):
        '''
        Train the autoencoder using the data formatted directly in a numpy 
        array. (No iteraters or anything)

        steps: (int) The number of iterations to do.
        inputs: (np array) The data to train your autoencoder on.
        test: (np array) The data to test your autoencoder on.
        lr: (int) Learning rate of the autoencoder
        '''
        loss_fn = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(self.model.parameters(), betas=[0.5, 0.999], lr=lr)

        for i in range(steps):
            # Half way through, half the learning rate
            if i == 15000:
                optimizer = optim.Adam(self.model.parameters(), betas=[0.5, 0.999], lr=lr/2)
            
            # Pick a batch to train on
            indices = np.random.choice(len(inputs), size=self.batch_size)
            x_batch = torch.Tensor(inputs[indices])
            x_batch = x_batch.view((-1,) + self.input_shape)

            # Calculate reconstruction loss
            pred = self.model(x_batch)
            loss_re = loss_fn(pred, x_batch)
            loss_re.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            del x_batch
            del pred
            
            # Every 100 iterations, test on a batch
            if i % 100 == 0:
                print("Training Step: {}, loss: {}".format(i, \
                    loss_re.detach().cpu().numpy() / self.batch_size))

                # Pick a batch to test on
                test_indices = np.random.choice(range(len(test)), size=self.batch_size)
                x_test = torch.Tensor(test[test_indices])
                x_test = x_test.view((-1,) + self.input_shape)
                
                # Calculate reconstruction loss
                test_pred = self.model(x_test)
                test_loss = loss_fn(test_pred, x_test)

                print("Testing Step: {}, loss: {}".format(i, \
                    test_loss.detach().cpu().numpy() / self.batch_size))

                del test_loss
                del x_test
                
                # Every 1000 iterations, save 64 images to see how the autoencoder 
                # is doing.
                if i % 1000 == 0:
                    test_imgs = test_pred.detach().cpu()[0:64]
                    save_image(test_imgs.view((64,) + self.img_shape), \
                        self.path + "/autoenocder_" + str(i) + '.png')

                    del test_imgs
                
                del test_pred
            
            del loss_re
            
        self.save_weights("autoencoder")

    def train_iter(self, steps, input_load, test_load, lr=0.001):
        '''
        Train the autoencoder using the data formatted in a torch DataLoader.

        steps: (int) The number of iterations
        input_load: (torch DataLoader) The data you want to train your autoencoder on
        test_load: (torch DataLoader) The data you want to test your autoencoder on
        lr: (int) learning rate
        '''
        loss_fn = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(self.model.parameters(), betas=[0.5, 0.999], lr=lr)

        input_gen = iter(input_load)
        test_gen = iter(test_load)

        for i in range(steps):
            # Half way through, half the learning rate
            if i == 15000:
                optimizer = optim.Adam(self.model.parameters(), betas=[0.5, 0.999], lr=lr/2)
            
            # Get next batch. If you've run out of batches, restart the DataLoader
            try:
                x_batch = next(input_gen)[0].cuda()
            except StopIteration:
                input_gen = iter(input_load)
                x_batch = next(input_gen)[0].cuda()
            
            # Calculate Reconstruction Loss
            pred = self.model(x_batch)
            loss_re = loss_fn(pred, x_batch)
            loss_re.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            del x_batch
            del pred
            
            if i % 100 == 0:
                print("Training Step: {}, loss: {}".format(i, loss_re.detach().cpu().numpy() / self.batch_size))

                # Get next batch. If you've run out of batches, restart the DataLoader
                try:
                    x_test = next(test_gen)[0].cuda()
                except StopIteration:
                    test_gen = iter(test_load)
                    x_test = next(test_gen)[0].cuda()

                # Test Reconstruction Loss
                test_pred = self.model(x_test)
                test_loss = loss_fn(test_pred, x_test)

                print("Testing Step: {}, loss: {}".format(i, \
                    test_loss.detach().cpu().numpy() / self.batch_size))

                del test_loss
                del x_test
                
                # Every 1000 iterations, save 64 images to see how the autoencoder 
                # is doing.
                if i % 1000 == 0:
                    test_imgs = test_pred.detach().cpu()[0:64]
                    save_image(test_imgs.view((64,) + self.img_shape), \
                        self.path + "/autoenocder_" + str(i) + '.png')

                    del test_imgs
                
                del test_pred
            
            del loss_re
            
        self.save_weights("autoencoder")

    def encode(self, data):
        '''
        Use the autoencoder to encode some data.

        data: (np array) The data to encode.
        '''
        data = torch.Tensor(data).cuda()
        encodings = []

        data = data.view((len(data),) + self.input_shape)

        for i in range(math.ceil(len(data)/self.batch_size)):
            x_batch = data[i*self.batch_size:(i+1)*self.batch_size]
            pred_batch = self.encoder(x_batch).detach().cpu().numpy()

            if len(encodings) == 0:
                encodings = pred_batch
            else:
                encodings = np.concatenate((encodings, pred_batch), 0)

        return encodings

    def decode(self, data):
        '''
        Use the autoencoder to decode some data.

        data: (np array) The data to decode.
        '''
        data = torch.Tensor(data).cuda()
        decodings = []

        for i in range(math.ceil(len(data)/self.batch_size)):
            x_batch = data[i*self.batch_size:(i+1)*self.batch_size]
            pred_batch = self.decoder(x_batch).detach().cpu().numpy()

            if len(decodings) == 0:
                decodings = pred_batch
            else:
                decodings = np.concatenate((decodings, pred_batch), 0)

        return decodings

    def save_weights(self, name):
        full_path = os.path.join(self.path, name)
        torch.save(self.model.state_dict(), full_path)

    def load_weights(self, name):
        full_path = os.path.join(self.path, name)
        self.model.load_state_dict(torch.load(full_path))

    def evaluate(self, x_test):
        loss = self.autoencoder.evaluate(x_test, x_test)
        print("Evaluating the Autoencoder. Loss: {}".format(loss))
        return loss

# WIP I guess...

if __name__ == '__main__':
    # Defaults:
    path = '.'
    dim = 10
    batch_size = 512
    lr = 0.001

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("file")
    parser.add_argument("dataset")
    parser.add_argument("dim")
    parser.add_argument("bs")
    parser.add_argument("train")
    parser.add_argument("lr")

    args = parser.parse_args()

    if not args.path is None:
        path = args.path

    if not args.dim is None:
        dim = args.dim

    if not args.bs is None:
        batch_size = args.bs

    if not args.lr is None:
        lr = args.lr

    if args.dataset is None:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif args.dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise NotImplementedError

    autoencoder = Autoencoder(inputs, path=PATH, dim=dim, batch_size=batch_size)

    if not args.file is None:
        autoencoder.load(args.file)

    if not args.train is None:
        autoencoder.train(args.train, lr=lr)

        if not args.file is None:
            autoencoder.save(args.file)




