import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class AE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, hidden_dim=400, use_cuda=True):
        super(AE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(hidden_dim)
        self.decoder = Decoder(hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            # self.encoder.cuda()
            # self.decoder.cuda()

    def forward(self, x):
        z = self.encoder.forward(x)
        x_out = self.decoder.forward(z)
        return x_out

class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(37*37*32, hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(hidden_dim, 37*37*32)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU())

    def forward(self, z):
        out = self.fc(z)
        out = out.reshape(-1, 32, 37, 37)
        out = self.layer1(out)
        out = self.layer2(out)
        return out