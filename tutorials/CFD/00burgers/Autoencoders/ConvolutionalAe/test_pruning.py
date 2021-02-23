import torch
import numpy as np
import torch.nn.utils.prune as prune
from torchsummary import summary
import torch.nn as nn

import time
import matplotlib.pyplot as plt
import argparse
from lstm import *
from convae import *

"""
CPU
Results:  0.0007992267608642578 0.003635058403015137 0.0006339812278747559 0.0036092233657836915 0.0006192541122436524 0.0036711573600769045
Results std:  0.0018208605908938426 0.0002670496192128683 9.048460265008887e-05
0.00023760945442909186 9.250801382441944e-05 0.0003400380395581786
GPU
Results:  0.0027509403228759767 0.003593730926513672 0.0007230257987976075 0.003582329750061035 0.0007891273498535157 0.003551943302154541
Results std:  0.02058271404175449 0.00026978082780839226 5.535019764741927e-05 0.00016335241867486484 0.000159090540983393 0.00010243224811197903
"""
class AEsingleoutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        # create the encoder and decoder networks
        self.model = model

    def forward(self, x):
        z = self.model(x)[0, 0]
        return z

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

class AEShallow(nn.Module):
    def __init__(self,
                 hidden_dim=400,
                 domain_size=DOMAIN_SIZE,
                 scale=(-1, 1),
                 mean=0):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(hidden_dim, domain_size)
        self.decoder = Decoder(hidden_dim, domain_size, self.encoder.hl, scale, mean)

    def forward(self, x):
        z = self.encoder.forward(x)
        x_out = self.decoder.forward(z)
        return x_out


class Encoder(nn.Module):
    def __init__(self, hidden_dim, domain_size):
        super(Encoder, self).__init__()
        self.ds = domain_size
        self.hl = 52#int(self.eval_size())

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16), nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32 * (self.hl)**2, hidden_dim),
                                nn.Sigmoid())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out

    def eval_size(self):
        convlayer = lambda x: np.floor((x + 2 * 2 - 5) / 1 + 1)
        poolayer = lambda x: np.floor((x - 2) / 2 + 1)
        return poolayer(convlayer(poolayer(convlayer(self.ds))))


class Decoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length, scale, mean):
        super(Decoder, self).__init__()
        self.ds = domain_size
        self.hl = hidden_length
        self.scale = scale

        self.fc = nn.Sequential(nn.Linear(hidden_dim,
                                          32 * (15)**2))  #, nn.ReLU())
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 2, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(2), nn.ReLU())

    def forward(self, z):
        out = self.fc(z)
        out = out.reshape(-1, 32,15,15)
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out.reshape(-1, self.ds * self.ds * 2)
        return out * (self.scale[1] - self.scale[0]) + self.scale[0]

def toeplitz_mult_ch(kernel, input_size):
    """Compute toeplitz matrix for 2d conv with multiple in and out channels.
    Args:
        kernel: shape=(n_out, n_in, H_k, W_k)
        input_size: (n_in, H_i, W_i)"""

    kernel_size = kernel.shape
    output_size = (kernel_size[0], input_size[1] - (kernel_size[1]-1), input_size[2] - (kernel_size[2]-1))
    T = np.zeros((output_size[0], int(np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))

    for i,ks in enumerate(kernel):  # loop over output channel
        for j,k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_1_ch(k, input_size[1:])
            T[i, :, j, :] = T_k

    T.shape = (np.prod(output_size), np.prod(input_size))

    return T

def main(args):
    WM_PROJECT = "../../"
    HIDDEN_DIM = args.latent_dim
    DOMAIN_SIZE = 60
    DIM = 2

    # Device configuration
    device = torch.device('cpu')
    print("device is: ", device)

    # snapshots have to be clipped before
    snap_vec = np.load(WM_PROJECT + "npSnapshots.npy")
    assert np.min(snap_vec) >= 0., "Snapshots should be clipped"

    n_total = snap_vec.shape[1]
    n_train = n_total-n_total//6

    # scale the snapshots
    nor = Normalize(snap_vec, center_fl=True)
    snap_framed = nor.framesnap(snap_vec)
    snap_scaled = nor.scale(snap_framed)
    snap_torch = torch.from_numpy(snap_scaled).to("cpu", dtype=torch.float)
    print("snapshots shape", snap_scaled.shape)
    print("Min max after scaling: ", np.min(snap_scaled), np.max(snap_scaled))

    # load autoencoder
    model = AE(
        HIDDEN_DIM,
        scale=(nor.min_sn, nor.max_sn),
        mean=nor.mean(device),
        domain_size=DOMAIN_SIZE).to(device)

    model.load_state_dict(torch.load("./model_"+str(args.latent_dim)+".ckpt"))
    model.eval()
    # summary(model, input_size=(DIM, DOMAIN_SIZE, DOMAIN_SIZE))
    # print(list(model.named_parameters()))

    zero = []
    first = []
    second = []
    third = []
    fourth = []
    fifth = []
    for i in range(100):
        init = time.time()
        output = model.decoder(torch.ones(1, 4).to(device))
        zero.append(time.time()-init)
        print("zero ", zero[i])

        mod_0 = torch.jit.trace(model.decoder, torch.ones(1, 4).to(device))

        init = time.time()
        mod_0(torch.ones(1, 4).to(device))
        first.append(time.time()-init)
        print("first ", first[i])

        for param in model.parameters():
            # print(param)
            mask = param != 0
            # print(mask.shape)
            param = torch.ones(mask.shape)
            param[~mask] = 0
            # print(param.shape)

        model.eval()
        init = time.time()
        output = model.decoder(torch.ones(1, 4).to(device))
        second.append(time.time()-init)
        print("second ", second[i])
        mod_1 = torch.jit.trace(model.decoder, torch.ones(1, 4).to(device))

        init = time.time()
        mod_1(torch.ones(1, 4).to(device))
        third.append(time.time()-init)
        print("third ", third[i])

        print("out shape: ", output.shape, output[0, 0].shape)
        (output[0, 0]).backward()

        acc = 0
        for name, module in model.decoder.named_modules():
            print("name", name, module)
            if isinstance(module, torch.nn.ConvTranspose2d):
                # print("OK conv")
                if module.weight.grad is not None:
                    prune_unstructured(module, name='weight')
                if module.bias.grad is not None:
                    prune_unstructured(module, name='bias')
            elif isinstance(module, torch.nn.Linear):
                if module.weight.grad is not None:
                    prune_unstructured(module, name='weight')
                if module.bias.grad is not None:
                    prune_unstructured(module, name='bias')

        # print("END PRUNING")
        # for name, item in list(model.named_buffers()):
        #     print(name)
        # print(module._forward_pre_hooks)
        # print(list(model.named_buffers()))

        output = model.decoder(torch.ones(1, 4).to(device))
        # print("test", model.decoder.layer0[0].weight, model.decoder.layer0[0].weight_orig)

        for name, module in model.decoder.named_modules():
            # print("name", name, module)
            if isinstance(module, torch.nn.ConvTranspose2d):
                # print("OK conv")
                # if module.weight.grad is not None:
                    # prune_unstructured(module, name='weight')
                prune.remove(module, 'weight')
                # if module.bias.grad is not None:
                    # prune_unstructured(module, name='bias')
                prune.remove(module, 'bias')
            elif isinstance(module, torch.nn.Linear):
                # if module.weight.grad is not None:
                    # prune_unstructured(module, name='weight')
                prune.remove(module, 'weight')
                # if module.bias.grad is not None:
                    # prune_unstructured(module, name='bias')
                prune.remove(module, 'bias')

        for param in model.parameters():
            if param is not None:
                acc += torch.sum(param != 0.)
            param = param.to_sparse()
            print(param)

        print("total non zero weights: ", acc, "percentage: ", 100*acc/68000)

        # model = AEsingleoutput(model.decoder)
        model.eval()

        with torch.no_grad():
            init = time.time()
            output = model.decoder(torch.ones(1, 4).to(device))
            fourth.append(time.time()-init)
            print("four ", fourth[i])
            mod_2 = torch.jit.trace(model.decoder, torch.ones(1, 4).to(device))

            init = time.time()
            doit = torch.ones(1, 4).to(device)
            mod_2(doit)
            fifth.append(time.time()-init)
            print("five ", fifth[i])

    print("Results: ", np.mean(np.array(zero)), np.mean(np.array(first)), np.mean(np.array(second)), np.mean(np.array(third)), np.mean(np.array(fourth)), np.mean(np.array(fifth)))
    print("Results std: ", np.std(np.array(zero)), np.std(np.array(first)), np.std(np.array(second)), np.std(np.array(third)), np.std(np.array(fourth)), np.std(np.array(fifth)))

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n',
                        '--num_epochs',
                        default=10000,
                        type=int,
                        help='number of training epochs')
    parser.add_argument('-lr',
                        '--learning_rate',
                        default=5.0e-3,
                        type=float,
                        help='learning rate')
    parser.add_argument('-batch',
                        '--batch_size',
                        default=20,
                        type=int,
                        help='batch')
    parser.add_argument('-dim',
                        '--latent_dim',
                        default=4,
                        type=int,
                        help='latent dim')
    parser.add_argument('-device',
                        '--device',
                        default='True',
                        type=str,
                        help='whether to use cuda')
    parser.add_argument('-i',
                        '--iter',
                        default=5,
                        type=int,
                        help='epoch when visualization runs')
    args = parser.parse_args()

    main(args)