import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import clock
from collections.abc import Iterable
import shutil

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
            Conv2d-1            [-1, 8, 28, 28]             408
       BatchNorm2d-2            [-1, 8, 28, 28]              16
             Swish-3            [-1, 8, 28, 28]               0
            Conv2d-4           [-1, 16, 14, 14]           1,168
       BatchNorm2d-5           [-1, 16, 14, 14]              32
             Swish-6           [-1, 16, 14, 14]               0
            Conv2d-7             [-1, 32, 7, 7]           4,640
       BatchNorm2d-8             [-1, 32, 7, 7]              64
             Swish-9             [-1, 32, 7, 7]               0
           Conv2d-10             [-1, 64, 4, 4]          18,496
      BatchNorm2d-11             [-1, 64, 4, 4]             128
            Swish-12             [-1, 64, 4, 4]               0
           Conv2d-13            [-1, 128, 3, 3]          32,896
      BatchNorm2d-14            [-1, 128, 3, 3]             256
            Swish-15            [-1, 128, 3, 3]               0
           Linear-16                    [-1, 4]           4,612
           Linear-17                 [-1, 1152]           5,760
            Swish-18                 [-1, 1152]               0
  ConvTranspose2d-19             [-1, 64, 4, 4]          32,832
      BatchNorm2d-20             [-1, 64, 4, 4]             128
            Swish-21             [-1, 64, 4, 4]               0
  ConvTranspose2d-22             [-1, 32, 7, 7]          18,464
      BatchNorm2d-23             [-1, 32, 7, 7]              64
            Swish-24             [-1, 32, 7, 7]               0
  ConvTranspose2d-25           [-1, 16, 14, 14]           8,208
      BatchNorm2d-26           [-1, 16, 14, 14]              32
            Swish-27           [-1, 16, 14, 14]               0
  ConvTranspose2d-28            [-1, 8, 29, 29]           1,160
      BatchNorm2d-29            [-1, 8, 29, 29]              16
            Swish-30            [-1, 8, 29, 29]               0
  ConvTranspose2d-31            [-1, 2, 60, 60]             258
      BatchNorm2d-32            [-1, 2, 60, 60]               4
================================================================
Total params: 129,642
Trainable params: 129,642
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 0.71
Params size (MB): 0.49
Estimated Total Size (MB): 1.24
----------------------------------------------------------------
initial latent variable shape :  [[ 92.20896149  63.95618057  40.81492615 -33.28034592]]
"""
DIM = 2
DOMAIN_SIZE = 60

class PruningGrads(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
            mask = t.grad != 0.
            # print("MASK", torch.Tensor(mask), t.grad)
            return mask.long()

def prune_unstructured(module, name):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> foobar_unstructured(m, name='bias')
    """
    PruningGrads.apply(module, name)
    return module

class AE(nn.Module):
    def __init__(self,
                 hidden_dim=400,
                 domain_size=DOMAIN_SIZE,
                 scale=(-1, 1),
                 mean=0):
        super(AE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(hidden_dim, domain_size)
        self.decoder = Decoder(hidden_dim, domain_size, scale, mean)

    def forward(self, x):
        z = self.encoder.forward(x)
        x_out = self.decoder.forward(z)
        return x_out


class Encoder(nn.Module):
    def __init__(self, hidden_dim, domain_size):
        super(Encoder, self).__init__()
        self.ds = domain_size

        self.layer1 = nn.Sequential(nn.Linear(7200, 14400),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(14400, 4),nn.ReLU())

    def forward(self, x):
        x = x.reshape(-1, self.ds * self.ds * 2)
        # print(x.shape)
        out = self.layer1(x)
        # print(x.shape)
        out = self.layer2(out)
        # print(out.shape)
        return out


class Decoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, scale, mean):
        super(Decoder, self).__init__()
        self.ds = domain_size
        self.scale = scale
        self.mean = mean
        # print("scale", scale)

        self.layer1 = nn.Sequential(nn.Linear(4, 14400),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(14400, 7200),nn.ReLU())

    def forward(self, z):
        # print(z.shape)
        out = self.layer1(z)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = out.reshape(-1, 2, self.ds, self.ds)
        # print(self.scale[0])
        # print(self.scale[1])
        out = out * 0.5 * (self.scale[1] - self.scale[0])
        out = out + 0.5 * (self.scale[1] + self.scale[0])
        out = out + self.mean
        out = out.reshape(-1, self.ds * self.ds * 2)
        return torch.nn.functional.relu(out)



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
            self._mean = np.mean(snap_tmp, axis=0, keepdims=True)
            # self._mean = snap_tmp[:1, :, :, :]
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
    f_path = checkpoint_dir + 'checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + 'best_model.pt'
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer=None):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


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
