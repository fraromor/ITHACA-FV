import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.modules.activation import ELU
from collections.abc import Iterable
import shutil

DIM = 4
DOMAIN_SIZE = 64


class AEFrequency(nn.Module):
    def __init__(self, hidden_dim=400):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = FrequencyEncoder(hidden_dim)
        self.decoder = FrequencyDecoder(hidden_dim)

    def forward(self, x):
        z = self.encoder.forward(x)
        x_out = self.decoder.forward(z)
        return x_out


class FrequencyEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        layers = []
        for i in range(12, 3, -1):
            layers.append(nn.Linear(2**i, 2**(i-1)))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(2**3, hidden_dim))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FrequencyDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        layers = []
        layers.append(nn.Linear(hidden_dim, 2**3))
        layers.append(nn.ReLU())

        for i in range(3, 12, 1):
            layers.append(nn.Linear(2**(i), 2**(i+1)))
            if i<11:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



class AE(nn.Module):
    def __init__(self, hidden_dim=400):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = DeepEncoder(hidden_dim)
        self.decoder = DeepDecoderTrans(hidden_dim)

    def forward(self, x):
        z = self.encoder.forward(x)
        x_out = self.decoder.forward(z)
        return x_out

class DeepEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ELU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            nn.ELU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.ELU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ELU())
        self.fc1 = nn.Sequential(nn.Linear(64 * 6**2, 64), nn.ELU())
        self.fc2 = nn.Sequential(nn.Linear(64, hidden_dim), nn.ELU())

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        # print(out.size())
        out = self.fc2(out)
        # print(out.size())
        return out

class DeepDecoderTrans(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()


        self.fc1 = nn.Sequential(nn.Linear(hidden_dim, 64),
                                nn.ELU())
        self.fc2 = nn.Sequential(nn.Linear(64, 64* 7**2),
                                nn.ELU())
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.ELU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
            nn.ELU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1),
            nn.ELU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(16, 2, kernel_size=4, stride=1, padding=1),
            )

    def forward(self, z):
        out = self.fc1(z)
        # print(out.size())
        out = self.fc2(out)
        # print(out.size())
        out = out.reshape(-1, 64, 7, 7)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = out.reshape(-1, 2, 64, 64)
        # print(out.size())
        return out



class DeepDeepEncoder(nn.Module):
    def __init__(self, hidden_dim, domain_size):
        super().__init__()
        self.ds = domain_size
        self.hl = 3
        # print(self.hl)
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=0),
            # nn.BatchNorm2d(8),
            nn.ELU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d16),
            nn.ELU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(32),
            nn.ELU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(32),
            nn.ELU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2),
            # nn.BatchNorm2d(64),
            nn.ELU())
        self.fc = nn.Sequential(nn.Linear(128 * 3**2, hidden_dim))  #, Swish())

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = self.layer5(out)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        # print(out.size())
        return out

    def eval_size(self):
        convlayer = lambda x: np.floor((x - 5 + 2) / 2 + 1)
        lastconvlayer = lambda x: np.floor((x - 4 + 2) / 2 + 1)
        # print(convlayer(self.ds))
        # print(convlayer(convlayer(self.ds)))
        # print(convlayer(convlayer(convlayer(self.ds))))
        return lastconvlayer(convlayer(convlayer(convlayer(self.ds))))


class DeepDecoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length):
        super().__init__()
        self.ds = domain_size
        self.hl = 3

        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64 * (self.hl)**2),
                                nn.ELU())
        self.layer0 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1), nn.ELU())
        self.layer1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), nn.ELU())
        self.layer2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=2), nn.ELU())
        self.layer3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1), nn.ELU())

    # @clock.clock
    def forward(self, z):
        # print("LATENT", z.shape, z)
        out = self.fc(z)
        out = out.reshape(-1, 64, self.hl, self.hl)
        # print(out.size())
        out = self.layer0(out)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        return out


class DeepDeepDecoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length):
        super(DeepDeepDecoder, self).__init__()
        self.ds = domain_size
        self.hl = 3

        self.fc = nn.Sequential(nn.Linear(hidden_dim, 128 * (self.hl)**2),
                                nn.ELU())
        self.layer0 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ELU())
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(32),
            nn.ELU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            nn.ELU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=0),
            # nn.BatchNorm2d(8),
            nn.ELU())
        self.layer4 = nn.Sequential(
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(8, 4, kernel_size=6, stride=2, padding=0),
            # nn.BatchNorm2d(2),
        )

    # @clock.clock
    def forward(self, z):
        # print("LATENT", z.shape, z)
        out = self.fc(z)
        out = out.reshape(-1, 128, self.hl, self.hl)
        # print(out.size())
        out = self.layer0(out)
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        return out


class ShallowDecoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length, scale):
        super(ShallowDecoder, self).__init__()
        self.ds = domain_size
        self.hl = hidden_length
        self.scale = scale

        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, 32 * (self.hl)**2),
                                    nn.BatchNorm1d(32 * (self.hl)**2),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 2, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(2), nn.ReLU())
        # self.layer2 = nn.Sequential(nn.Linear(32 * (self.hl)**2, 3 * (self.ds)**2), nn.BatchNorm1d(3 * (self.ds)**2), nn.ReLU())
        # self.layer3 = nn.Sequential(nn.Linear(hidden_dim, 32 * (self.hl)**2)), nn.BatchNorm1d(32 * (self.hl)**2)), nn.ReLU())

    def forward(self, z):
        out = self.layer1(z)
        out = out.reshape(-1, 32, self.hl, self.hl)
        out = self.layer2(out).reshape(-1, self.ds * self.ds * 2)
        return out * (self.scale[1] - self.scale[0]) + self.scale[0]


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
            # snap = (snap-self._min_sn)/(self._max_sn-self._min_sn)
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
            # snap = snap * (self._max_sn - self._min_sn) + self._min_sn
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
