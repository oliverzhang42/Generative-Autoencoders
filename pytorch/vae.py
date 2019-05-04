from __future__ import print_function
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import View

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

kwargs = {'num_workers': 1, 'pin_memory': True}
transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize(64),
        transforms.ToTensor(),
])
data_dir = 'celebA'
dset = datasets.ImageFolder(data_dir, transform)
    
dset_size = len(dset)
train_size = 7*dset_size//10

train_set, test_set = torch.utils.data.random_split(dset, (train_size, dset_size-train_size))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

shape = (3, 64, 64)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.dim = 100
        self.create_conv_model()

    def create_conv_model(self):
        # https://github.com/1Konny/WAE-pytorch/blob/master/model.py

        H = 512
        input_shape = (3, 64, 64)

        assert input_shape[1] == input_shape[2], "We assume the images are square. (Also, channels are first)"
        sidelength = input_shape[1]

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
            View(-1, 1024*4*4)
        )

        self.means = nn.Linear(1024*4*4, self.dim)
        self.variance = nn.Linear(1024*4*4, self.dim)

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

    def encode(self, x):
        h1 = self.encoder(x)
        return self.means(h1), self.variance(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.decoder(z)
        return h3

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=3e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
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
            data = data.cuda()
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 3, 64, 64)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    model.load_state_dict(torch.load("vae_model"))

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        torch.save("vae_model", model.state_dict())

        with torch.no_grad():
            sample = torch.randn(100, 100)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, 64, 64),
                       'results/sample_' + str(epoch) + '.png')

    with torch.no_grad():
        for i in range(100):
            sample = torch.randn(100, 100)
            sample = model.decode(sample).cpu()
            torch.save("wae_images{}".format(i), sample)