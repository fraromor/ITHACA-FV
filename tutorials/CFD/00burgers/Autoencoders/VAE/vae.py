import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import Iterable

import pyro
import pyro.distributions as dist

from clock import *
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 29, 29]             408
       BatchNorm2d-2            [-1, 8, 29, 29]              16
               ELU-3            [-1, 8, 29, 29]               0
            Conv2d-4           [-1, 16, 14, 14]           3,216
       BatchNorm2d-5           [-1, 16, 14, 14]              32
               ELU-6           [-1, 16, 14, 14]               0
            Conv2d-7             [-1, 32, 6, 6]          12,832
       BatchNorm2d-8             [-1, 32, 6, 6]              64
               ELU-9             [-1, 32, 6, 6]               0
           Conv2d-10             [-1, 64, 3, 3]          32,832
      BatchNorm2d-11             [-1, 64, 3, 3]             128
              ELU-12             [-1, 64, 3, 3]               0
           Linear-13                    [-1, 4]           2,308
           Linear-14                  [-1, 576]           2,880
  ConvTranspose2d-15             [-1, 32, 7, 7]          51,232
      BatchNorm2d-16             [-1, 32, 7, 7]              64
              ELU-17             [-1, 32, 7, 7]               0
  ConvTranspose2d-18           [-1, 16, 15, 15]          12,816
      BatchNorm2d-19           [-1, 16, 15, 15]              32
              ELU-20           [-1, 16, 15, 15]               0
  ConvTranspose2d-21            [-1, 8, 29, 29]           3,208
      BatchNorm2d-22            [-1, 8, 29, 29]              16
              ELU-23            [-1, 8, 29, 29]               0
  ConvTranspose2d-24            [-1, 2, 60, 60]             578
      BatchNorm2d-25            [-1, 2, 60, 60]               4
================================================================
Total params: 122,666
Trainable params: 122,666
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 0.65
Params size (MB): 0.47
Estimated Total Size (MB): 1.15
----------------------------------------------------------------
"""
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 60, 60]           1,216
       BatchNorm2d-2           [-1, 16, 60, 60]              32
              ReLU-3           [-1, 16, 60, 60]               0
         MaxPool2d-4           [-1, 16, 30, 30]               0
            Conv2d-5           [-1, 32, 30, 30]          12,832
       BatchNorm2d-6           [-1, 32, 30, 30]              64
              ReLU-7           [-1, 32, 30, 30]               0
         MaxPool2d-8           [-1, 32, 15, 15]               0
            Linear-9                    [-1, 4]          28,804
             Tanh-10                    [-1, 4]               0
           Linear-11                 [-1, 7200]          36,000
  ConvTranspose2d-12           [-1, 16, 30, 30]           8,208
      BatchNorm2d-13           [-1, 16, 30, 30]              32
             ReLU-14           [-1, 16, 30, 30]               0
  ConvTranspose2d-15            [-1, 3, 60, 60]           1,731
      BatchNorm2d-16            [-1, 3, 60, 60]               6
             ReLU-17            [-1, 3, 60, 60]               0
================================================================
Total params: 88,925
Trainable params: 88,925
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.04
Forward/backward pass size (MB): 2.77
Params size (MB): 0.34
Estimated Total Size (MB): 3.15
----------------------------------------------------------------
"""
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 60, 60]             816
       BatchNorm2d-2           [-1, 16, 60, 60]              32
              ReLU-3           [-1, 16, 60, 60]               0
         MaxPool2d-4           [-1, 16, 30, 30]               0
            Conv2d-5           [-1, 32, 30, 30]          12,832
       BatchNorm2d-6           [-1, 32, 30, 30]              64
              ReLU-7           [-1, 32, 30, 30]               0
         MaxPool2d-8           [-1, 32, 15, 15]               0
            Linear-9                    [-1, 4]          28,804
          Sigmoid-10                    [-1, 4]               0
           Linear-11                 [-1, 7200]          36,000
  ConvTranspose2d-12           [-1, 16, 30, 30]           8,208
      BatchNorm2d-13           [-1, 16, 30, 30]              32
             ReLU-14           [-1, 16, 30, 30]               0
  ConvTranspose2d-15            [-1, 2, 60, 60]           1,154
      BatchNorm2d-16            [-1, 2, 60, 60]               4
             ReLU-17            [-1, 2, 60, 60]               0
================================================================
Total params: 87,946
Trainable params: 87,946
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 2.69
Params size (MB): 0.34
Estimated Total Size (MB): 3.05
----------------------------------------------------------------
"""
DIM = 2
DOMAIN_SIZE = 60

# define a PyTorch module for the VAE
class VAE(nn.Module):
    def __init__(self,
                 hidden_dim=400,
                 use_cuda=True,
                 domain_size=DOMAIN_SIZE,
                 scale=(-1, 1),
                 mean=0,
                 nor=None):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = DeepEncoder(hidden_dim, domain_size)
        self.decoder = DeepDecoder(hidden_dim, domain_size,
                                       self.encoder.hl, scale, mean)
        self.scale = scale
        self.ds = domain_size
        self.nor = nor
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            # self.encoder.cuda()
            # self.decoder.cuda()
        self.use_cuda = use_cuda
        self.hidden_dim = hidden_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            # print("model x", x.shape)
            # plot_snapshot((x.detach().cpu().numpy()).reshape(20, 2, 60, 60), 10)
            # plot_snapshot(x, 10)
            z_loc = torch.zeros(x.shape[0], self.hidden_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.hidden_dim, dtype=x.dtype, device=x.device)
            # print("normal", z_loc.shape, z_scale.shape, z_loc, z_scale)
            # sample from prior (value will be sampled by guide when computing
            # the ELBO)
            assert torch.all( z_scale >= 0.), "scale is not positive"
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # print("model z", z.shape, z)
            # decode the latent code z
            loc_img, loc_scale = self.decoder.forward(z)
            # print("model loc", loc_img.shape, loc_scale.shape)
            # plot_snapshot(self.nor.rescale(x, x.device).detach().cpu().numpy(), 10)
            # score against actual images
            # plot_snapshot((self.nor.rescale(x, x.device)-loc_img.reshape(-1,
            # 2, 60, 60)).detach().cpu().numpy(), 0)
            assert torch.all( loc_scale >= 0.), "scale is not positive"
            pyro.sample("obs", dist.Normal(loc_img, loc_scale).to_event(1), obs=self.nor.rescale(x, x.device).reshape(-1, 2*self.ds**2))
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        # print("guide x", x.shape, x)
        # plot_snapshot((x.detach().cpu().numpy()).reshape(20, 2, 60, 60), 10)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # print("loc guide", z_scale)#z_loc.shape, z_scale.shape, z_loc, z_scale)
            # sample the latent code z
            assert torch.all( z_scale >= 0.), "scale is not positive "+ str(z_scale)
            latent = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # print("latent", latent.shape, latent)

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img, loc_scale = self.decoder(z)
        return loc_img, loc_scale

class ShallowEncoder(nn.Module):
    def __init__(self, hidden_dim, domain_size):
        super(ShallowEncoder, self).__init__()
        self.ds = domain_size
        z_dim = self.ds**2

        # setup the three linear transformations used
        self.fc1 = nn.Linear(2*self.ds**2, z_dim)
        self.fc21 = nn.Linear(z_dim, hidden_dim)
        self.fc22 = nn.Linear(z_dim, hidden_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 2*self.ds**2)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale

class ShallowDecoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length, scale, mean):
        super(ShallowDecoder, self).__init__()
        self.ds = domain_size
        self.hl = hidden_length
        self.scale = scale
        self.mean = mean
        z_dim = self.ds**2

        self.fc1 = nn.Sequential(nn.Linear(hidden_dim, z_dim), nn.ELU())
        self.fc21 = nn.Sequential(nn.Linear(z_dim, 2*self.ds**2), nn.ELU())

    def forward(self, z):
        hidden = self.fc1(z)
        loc_img = self.fc21(hidden)
        return loc_img

class DeepEncoder(nn.Module):
    def __init__(self, hidden_dim, domain_size):
        super(DeepEncoder, self).__init__()
        self.ds = domain_size
        self.hl = int(self.eval_size())
        # print(self.hl)
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, stride=2, padding=1), nn.ELU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8,16, kernel_size=5, stride=2, padding=1),
            nn.ELU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.ELU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU())

        self.fc = nn.Sequential(nn.Linear(64 * (self.hl)**2, hidden_dim), nn.ELU())
        self.sigma_net = nn.Sequential(nn.Linear(2 * self.ds**2, hidden_dim), nn.Softmax())

    def forward(self, x):
        # print("enc x", x)
        # print("wieght", self.layer1[0].weight)
        out = self.layer1(x)
        # print("layer1", out.size(), out)
        out = self.layer2(out)
        # print("layer2", out.size(), out)
        out = self.layer3(out)
        # print("layer3", out.size(), out)
        out = self.layer4(out)
        # print("layer4", out.size(), out)
        out = out.reshape(out.size(0), -1)
        # print("layer5: ", out.shape, out)
        out = self.fc(out)
        # print("latent enc: ", out.shape, out)
        return out, torch.exp(self.sigma_net(x.reshape(-1, 2*self.ds**2)))

    def eval_size(self):
        convlayer = lambda x: np.floor((x  - 5 + 2) / 2 + 1)
        lastconvlayer = lambda x: np.floor((x  - 4 + 2) / 2 + 1)
        # print(convlayer(self.ds))
        # print(convlayer(convlayer(self.ds)))
        # print(convlayer(convlayer(convlayer(self.ds))))
        return lastconvlayer(convlayer(convlayer(convlayer(self.ds))))

class Decoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length, scale):
        super(Decoder, self).__init__()
        self.ds = domain_size
        self.hl = hidden_length
        self.scale = scale

        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32 * (self.hl)**2))#, nn.ReLU())
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 2, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(2), nn.ReLU())

    def forward(self, z):
        out = self.fc(z)
        out = out.reshape(-1, 32, self.hl, self.hl)
        out = self.layer1(out)
        out = self.layer2(out).reshape(-1, self.ds * self.ds * 2)
        return out*(self.scale[1]-self.scale[0])+self.scale[0]

class DeepDecoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length, scale, mean):
        super(DeepDecoder, self).__init__()
        self.ds = domain_size
        self.hl = hidden_length
        self.scale = scale
        self.mean = mean

        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64 * (self.hl)**2), Swish())
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1), Swish())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32,16, kernel_size=5, stride=2, padding=1), Swish())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2), Swish())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(8, 2, kernel_size=6, stride=2, padding=1))
        self.sigma_net = nn.Sequential(nn.Linear(hidden_dim, 2 * (self.ds)**2 ))

    def forward(self, z):
        out = self.fc(z)
        out = out.reshape(-1, 64, self.hl, self.hl)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)

        out *= 0.5 * (self.scale[1] - self.scale[0])
        out += 0.5 * (self.scale[1] + self.scale[0])
        out += self.mean
        out = out.reshape(-1, self.ds * self.ds * 2)

        return nn.functional.relu(out), torch.exp(self.sigma_net(z))

def regularizerl2(model, device, factor=0.01):
    l2_lambda = factor
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return l2_lambda * l2_reg

def regularizerl1(model, device, factor=0.01):
    l1_lambda = factor
    l1_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l1_reg += torch.norm(param, 1)
    return l1_lambda * l1_reg

class Normalize(object):
    def __init__(self, snap, center_fl=True, scale_fl=True):
        self.n_total = snap.shape[0]
        self._center_fl = center_fl
        self._scale_fl = scale_fl

        snap_tmp = self.framesnap(snap)

        if self._center_fl:
            # self._mean = np.mean(snap_tmp, axis=0, keepdims=True)
            self._mean = snap_tmp[:1, :, :, :]
            # plot_snapshot(self._mean, 0, title="mean")

        if self._scale_fl:
            print("max, min snap before centering: ", np.max(snap_tmp),
                  np.min(snap_tmp))
            if self._center_fl:
                snap_tmp = snap_tmp - self._mean
            self._max_sn = np.max(snap_tmp)
            self._min_sn = np.min(snap_tmp)
            # self._std = np.std(snap_tmp, axis=0, keepdims=True )

    @staticmethod
    def framesnap(snap):
        # reshape as (train_samples, channel, y, x)
        return snap.T.reshape(-1, 3, DOMAIN_SIZE, DOMAIN_SIZE)[:, :DIM, :, :]

    @staticmethod
    def frame2d(snap):
        # reshape as (train_samples, channel, y, x)
        return snap.reshape(-1, 2, DOMAIN_SIZE, DOMAIN_SIZE)

    @staticmethod
    def vectorize2d(snap):
        return snap.reshape(-1, DIM * DOMAIN_SIZE**2)

    @staticmethod
    def vectorize3d(snap):
        return snap.reshape(-1, DIM * DOMAIN_SIZE**3)

    def center(self, snap):
        return np.mean(snap, axis=0)

    def scale(self, snap, device=None):
        assert len(
            snap.shape) == 4, "snapshots to be scaled must be in frame format"

        if self._center_fl:
            if device:
                mean = torch.from_numpy(self._mean).to(device,
                                                       dtype=torch.float)
                # std = torch.from_numpy(self._std).to(device, dtype=torch.float)
            else:
                mean = self._mean
                # std = self._std

            snap = snap - mean

        if self._scale_fl:
            snap = snap - 0.5 * (self._min_sn + self._max_sn)
            snap = snap * 2 / (self._max_sn - self._min_sn)
            # snap /= std
            assert np.max(snap) <= 1.0, "Error in scale " + str(np.max(snap))
            assert np.min(snap) >= -1.0, "Error in scale " + str(np.min(snap))
        return snap

    def rescale(self, snap, device=None):
        assert len(snap.shape
                   ) == 4, "snapshots to be rescaled must be in frame format"

        if self._scale_fl:
            snap = snap * (self._max_sn - self._min_sn) / 2
            snap = snap + 0.5 * (self._min_sn + self._max_sn)
            # snap *= std

        if self._center_fl:
            if device:
                mean = torch.from_numpy(self._mean).to(device,
                                                       dtype=torch.float)
                # std = torch.from_numpy(self._std).to(device, dtype=torch.float)
            else:
                mean = self._mean
                # std = self._std
            snap = snap + mean

        return snap

    @property
    def max_sn(self):
        return self._max_sn

    @property
    def min_sn(self):
        return self._min_sn

    # def std(self, device=None):
    #     if device:
    #         return torch.from_numpy(self._std).to(device, dtype=torch.float)
    #     else:
    #         return self._std

    def mean(self, device=None):
        if self._center_fl:
            if device:
                return torch.from_numpy(self._mean).to(device,
                                                       dtype=torch.float)
            else:
                return self._mean


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + 'checkpoint'
    torch.save(state["model"], f_path)
    # state["model"].save(f_path+str("_vae"))
    state["optimizer"].save(f_path+str("_optim"))
    if is_best:
        best_fpath = best_model_dir + 'best_model.pt'
        torch.save(state["model"], best_fpath)



def load_ckp(checkpoint_fpath, model, optimizer=None, optim_path=None):
    model.load(checkpoint_fpath)
    if optimizer and optim_path:
        optimizer.load(optim_path)
    return model, optimizer


def plot_snapshot(frame, idx, idx_coord=0, title=""):
    m = frame.shape[2]
    x, y = np.meshgrid(np.arange(m), np.arange(m))
    z = frame[idx, idx_coord, x, y]
    plt.figure(figsize=(7, 6))
    plt.title(title)
    pl = plt.contourf(x, y, z)
    # v1 = np.linspace(0, 0.1, 100)
    # plt.clim(0., 0.1)
    cb = plt.colorbar(pl, fraction=0.046, pad=0.04)  #, ticks=v1)
    plt.show()
    # cb.ax.tick_params(labelsize='large')
    # cb.ax.set_yticklabels(["{:2.5f}".format(i) for i in v1])
    # plt.savefig("data/snap_"+str(idx))
    # plt.close()


def plot_two(snap,
             snap_reconstruct,
             idx,
             epoch,
             idx_coord=0,
             title='bu',
             save=True):
    domain_size = snap.shape[2]
    x, y = np.meshgrid(np.arange(domain_size), np.arange(domain_size))
    if isinstance(idx, Iterable):
        z = [snap[n, idx_coord, x, y] for n in idx]
        z_reconstruct = [snap_reconstruct[n, idx_coord, x, y] for n in idx]
    else:
        z = [snap[idx, idx_coord, x, y]]
        z_reconstruct = [snap_reconstruct[idx, idx_coord, x, y]]

    fig, axes = plt.subplots(3, len(z), figsize=(len(z) * 4, 10))
    fig.suptitle(title)
    if len(z) > 2:
        for i, image in enumerate(z):
            im = axes[0, i].contourf(x, y, image)
            fig.colorbar(im, ax=axes[0, i])
        for i, image in enumerate(z_reconstruct):
            im_ = axes[1, i].contourf(x, y, image)
            fig.colorbar(im_, ax=axes[1, i])
        for i, image in enumerate(z):
            A = z > 0
            axes[2, i].spy(A)
    elif len(z) == 1:
        for i, image in enumerate(z):
            im = axes[0].contourf(x, y, image)
            fig.colorbar(im, ax=axes[0])
        for i, image in enumerate(z_reconstruct):
            im_ = axes[1].contourf(x, y, image)
            fig.colorbar(im_, ax=axes[1])
        for i, image in enumerate(z):
            A = image > 0
            axes[2].spy(A)
    if save:
        plt.savefig('./data/' + title + "_" + str(epoch) + '.png')
        plt.close()
    else:
        plt.show()


def plot_compare(snap, snap_reconstruct, n_train, idx_coord=0, n_samples=5):
    domain_size = snap.shape[2]
    x, y = np.meshgrid(np.arange(domain_size), np.arange(domain_size))
    index_list = np.random.randint(0, n_train, n_samples)
    z = [snap[n, idx_coord, x, y] for n in index_list]
    z_reconstruct = [snap_reconstruct[n, idx_coord, x, y] for n in index_list]

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 4, 20))
    fig.suptitle("comparison of snapshots and reconstructed snapshots")
    for i, image in enumerate(z):
        axes[0, i].contourf(x, y, image)
    for i, image in enumerate(z_reconstruct):
        axes[1, i].contourf(x, y, image)
    plt.show()


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
