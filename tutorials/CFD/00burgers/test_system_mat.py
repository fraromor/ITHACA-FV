import numpy as np
import matplotlib.pyplot as plt
import torch
from convae import *

def plot_c(spam, title=""):
    x, y = np.meshgrid(np.arange(60), np.arange(60))
    torch_frame_py = spam.T.reshape(4, 2, 60, 60)
    z0 = torch_frame_py[0, 0, x, y]
    z1 = torch_frame_py[1, 0, x, y]
    z2 = torch_frame_py[2, 0, x, y]
    z3 = torch_frame_py[3, 0, x, y]

    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(title)

    # plt.subplot(2, 2, 1)
    pl0 = axs[0, 0].contourf(x, y, z0)
    cb0 = plt.colorbar(pl0, fraction=0.046, pad=0.04, ax=axs[0,0])  #, ticks=v1)

    # plt.subplot(2, 2, 2)
    pl1 = axs[0, 1].contourf(x, y, z1)
    cb1 = plt.colorbar(pl1, fraction=0.046, pad=0.04, ax=axs[0,1])  #, ticks=v1)

    # plt.subplot(2, 2, 3)
    pl2 = axs[1, 0].contourf(x, y, z2)
    cb2 = plt.colorbar(pl2, fraction=0.046, pad=0.04, ax=axs[1,0])  #, ticks=v1)

    # plt.subplot(2, 2, 4)
    pl3 = axs[1, 1].contourf(x, y, z3)
    cb3 = plt.colorbar(pl3, fraction=0.046, pad=0.04, ax=axs[1,1])  #, ticks=v1)

    plt.savefig('./data/' + title.replace(" ", "") + '.png')
    # plt.close()
    plt.show()

# Device configuration
device = torch.device('cuda')
print("device is: ", device)

init = np.load("./Autoencoders/ConvolutionalAe/latent_initial_4.npy")
print(init)
np.save("./Autoencoders/ConvolutionalAe/latent_initial_4_float64.npy", init.astype(np.float64))

mass = np.load("mass.npy")
print("mass", mass[:2, :2])
plt.spy(mass[:70, :70])
plt.show()

div = np.load("div.npy")
print("div", div[:10, :10])
plt.spy(div[:100, :100])
plt.show()

diff = np.load("diff.npy")
print("diff", diff[:70, :70])
plt.spy(diff[:70, :70])
plt.show()