import argparse
from  keras.datasets import mnist, fashion_mnist

import numpy as np

import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Autoencoder():
    def __init__(self, inputs, test, dim=10, path='.', batch_size=512, extra_layers=1, conv=False):
        ''' 
        inputs is meant to keep its original shape.
        '''
        if not os.path.isdir(path):
            print("Making folder at path: {}".format(path))
            os.mkdir(path)

        self.batch_size = batch_size
        self.path = path
        self.dim = dim
        self.inputs = torch.Tensor(inputs)
        self.test = torch.Tensor(test)
        self.img_shape = self.inputs.shape

        if conv:
            self.input_shape = self.inputs.shape
            self.create_conv_model(extra_layers)
        else:
            self.inputs = self.inputs.view((self.inputs.shape[0],-1))
            self.test = self.test.view((self.test.shape[0],-1))
            self.input_shape = self.inputs.shape
            self.create_model(extra_layers)

    def create_model(self, extra_layers):
        H = 512
        input_shape = self.input_shape

        encoder = []

        encoder.append(nn.Linear(input_shape[1], H))
        encoder.append(nn.LeakyReLU())
        encoder.append(nn.BatchNorm1d(H))
        for i in range(extra_layers):
            encoder.append(nn.Linear(H, H))
            encoder.append(nn.LeakyReLU())
            encoder.append(nn.BatchNorm1d(H))
        encoder.append(nn.Linear(H, self.dim))
        encoder.append(nn.Sigmoid())
        
        self.encoder = nn.Sequential(*encoder)

        decoder = []

        decoder.append(nn.Linear(self.dim, H))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.BatchNorm1d(H))
        for i in range(extra_layers):
            decoder.append(nn.Linear(H, H))
            decoder.append(nn.LeakyReLU())
            decoder.append(nn.BatchNorm1d(H))
        decoder.append(nn.Linear(H, input_shape[1]))
        decoder.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder)

        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )

    def create_conv_model(self, extra_layers):
        # Assume we're going for Cifar10 (32x32)

        H = 512
        input_shape = self.input_shape

        assert self.input_shape[2] == self.input_shape[3], "We assume the images are square. (Also, channels are first)"
        sidelength = self.input_shape[2]

        '''
        encoder = []

        encoder.append(nn.Conv2d(input_shape[1], 128, 3, padding=1)) #128x128
        encoder.append(nn.BatchNorm2d(128))
        encoder.append(nn.LeakyReLU())
        encoder.append(nn.MaxPool2d(2)) #64x64
        encoder.append(nn.Conv2d(64, 32, 3, padding=1)) #64x64
        encoder.append(nn.BatchNorm2d(32))
        encoder.append(nn.LeakyReLU())
        encoder.append(nn.MaxPool2d(2)) #32x32
        encoder.append(nn.Conv2d(32, 16, 3, padding=1)) #32x32
        encoder.append(nn.BatchNorm2d(16))
        encoder.append(nn.LeakyReLU())
        encoder.append(nn.MaxPool2d(2)) #32x32

        
        encoder.append(View(self.batch_size,-1))
        encoder.append(nn.Linear((sidelength//8)*(sidelength//8)*16, 128))
        encoder.append(nn.LeakyReLU())
        encoder.append(nn.Linear(128, self.dim))
        encoder.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder)

        decoder = []

        decoder.append(nn.Linear(self.dim, 128))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.Linear(128, (sidelength//8)*(sidelength//8)*16))
        decoder.append(nn.LeakyReLU())
        
        decoder.append(View(self.batch_size, 16, (sidelength//8), (sidelength//8)))

        decoder.append(nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.ConvTranspose2d(64, 3, 3, padding=1))
        decoder.append(nn.Sigmoid())
        
        decoder.append(nn.Conv2d(32, 64, 3, padding=1)) #8x8
        decoder.append(nn.BatchNorm2d(64))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.UpsamplingNearest2d(scale_factor=2))
        decoder.append(nn.Conv2d(64, 128, 3, padding=1)) # 16x16
        decoder.append(nn.BatchNorm2d(128))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.UpsamplingNearest2d(scale_factor=2))
        decoder.append(nn.Conv2d(128, 128, 3, padding=1)) #32x32
        decoder.append(nn.BatchNorm2d(64))
        decoder.append(nn.LeakyReLU())
        decoder.append(nn.Conv2d(128, 3, 3, padding=1)) #32x32
        decoder.append(nn.BatchNorm2d(3))
        decoder.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder)
        '''

        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )

        '''


    def train(self, steps, lr=0.001, images=False):
        loss_fn = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(self.model.parameters(), betas=[0.5, 0.999], lr=lr)

        for i in range(steps):
            # Train the Encoder via reconstruction MSE

            indices = np.random.choice(range(len(self.inputs)), size=self.batch_size)
            x_batch = self.inputs[indices]
            
            pred = self.model(x_batch)
            loss_re = loss_fn(pred, x_batch)

            loss_re.backward()
            optimizer.step()

            optimizer.zero_grad()

            if i % 100 == 0:
                print("Training Step: {}, loss: {}".format(i, loss_re.detach().cpu().numpy() / self.batch_size))

                test_indices = np.random.choice(range(len(self.test)), size=self.batch_size)

                x_test = self.test[test_indices]
                test_pred = self.model(x_test)
                test_loss = loss_fn(test_pred, x_test)

                print("Testing Step: {}, loss: {}".format(i, test_loss.detach().cpu().numpy() / self.batch_size))

                if images:
                    display_img(pred.detach().cpu().numpy()[0:16], self.path, show=False, shape=self.img_shape, channels_first=False, index=i//100)

                del test_loss
                del test_pred
                del x_test
            
            del loss_re
            del x_batch
            del pred
            


        self.save_weights("autoencoder")

    def encode(self, data):
        data = torch.Tensor(data).cuda()
        encodings = []

        data = data.view(self.input_shape)

        for i in range(len(data)//self.batch_size):
            x_batch = data[i*self.batch_size:(i+1)*self.batch_size]
            pred_batch = self.encoder(x_batch).detach().cpu().numpy()

            if len(encodings) == 0:
                encodings = pred_batch
            else:
                encodings = np.concatenate((encodings, pred_batch), 0)

        return encodings

    def decode(self, data):
        data = torch.Tensor(data).cuda()
        decodings = []

        for i in range(len(data)//self.batch_size):
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

        print("Evaluating the Autoencoder. Loss:")
        print(loss)

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




