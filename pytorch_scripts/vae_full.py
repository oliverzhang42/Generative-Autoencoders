#https://github.com/pytorch/examples/blob/master/vae/main.py

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from utils import *


from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, 30)
        self.fc32 = nn.Linear(512, 30)
        self.fc4 = nn.Linear(30, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 784)

    def encode(self, x):
        #h1 = nn.LeakyReLU()(self.fc1(x))
        #h2 = nn.LeakyReLU()(self.fc2(h1))

        h1 = nn.LeakyReLU()(nn.BatchNorm1d(512)(self.fc1(x)))
        h2 = nn.LeakyReLU()(nn.BatchNorm1d(512)(self.fc2(h1)))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        #h4 = nn.LeakyReLU()(self.fc4(z))
        #h5 = nn.LeakyReLU()(self.fc5(h4))

        h4 = nn.LeakyReLU()(nn.BatchNorm1d(512)(self.fc4(z)))
        h5 = nn.LeakyReLU()(nn.BatchNorm1d(512)(self.fc5(h4)))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return x

import pudb; pudb.set_trace()

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=[0.5, 0.999])


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def loss_extra(recon_x, x):
    layer1_real = model.fc1(x.view(-1, 784))
    layer1_fake = model.fc1(recon_x)
    
    loss = F.mse_loss(layer1_fake, layer1_real, reduction='sum')
    
    return loss
    

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward(retain_graph=False)

        '''
        for param in model.fc1.parameters():
            param.requires_grad=False

        for param in model.fc21.parameters():
            param.requires_grad=False

        for param in model.fc22.parameters():
            param.requires_grad=False

        extra_loss = loss_extra(recon_batch, data)
        extra_loss.backward()

        for param in model.fc1.parameters():
            param.requires_grad=True

        for param in model.fc21.parameters():
            param.requires_grad=True

        for param in model.fc22.parameters():
            param.requires_grad=True
        '''

        train_loss += loss.item()
        #train_loss += extra_loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item() #+ loss_extra(recon_batch, data).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, 20):
        train(epoch)
        test(epoch)

        with torch.no_grad():
            sample = torch.randn(64, 30).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')