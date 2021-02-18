import numpy as np
from scipy.optimize import least_squares
import scipy.linalg as lg
from scipy.sparse import diags
from convae import *
import torch
def plot_c(spam, title=""):
    x, y = np.meshgrid(np.arange(60), np.arange(60))
    torch_frame_py = spam.T.reshape(8, 2, 60, 60)
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

class Residual(object):
    def __init__(self, decoder, init, mu, snap, device):
        self.decoder = decoder
        self.mu = mu
        self.ref = snap
        self.device = device

        self.prev = self.ref * self.mu
        self.g0 = decoder(init).squeeze().detach().cpu().numpy()
        self.latent = init.squeeze().detach().cpu().numpy()

    def update(self, x0):
        self.prev = self.forward(x0)
        self.latent = x0

    def forward(self, x):
        g = self.decoder(torch.from_numpy(x).to(self.device, dtype=torch.float)).squeeze().detach().cpu().numpy()
        return g - self.g0 + self.ref * self.mu

    def __call__(self, x):
        # print("x trial: ", x)
        output = self.forward(x)

        u_ = output[:3600].reshape(60, 60)
        v_ = output[3600:].reshape(60, 60)
        u = u_[1:59, 1:59].reshape(-1)
        v = v_[1:59, 1:59].reshape(-1)
        # print("shapes: ", u.shape, v.shape)

        dx = dy = 1/59
        dt = 0.001

        M_b = diags([np.ones(60-2), -1* np.ones(60-3)], [0, -1]).toarray()
        N_b = np.diag(np.ones(60-2))
        # print("shapes: ", M_b.shape, N_b.shape)

        M = (1/dx)*lg.block_diag(*(M_b for i in range(60-2)))
        N = (1/dy)*lg.block_diag(*(N_b for i in range(60-2)))
        # print("shapes: ", M.shape, N.shape)

        e1 = np.zeros((1, 60-2))
        e1[0, 0] = 1

        b_ux1 = (1/dx)*np.kron(u.reshape(58, 58)[0, :], e1).squeeze()
        b_uy1 = (1/dx)*np.kron(e1, u.reshape(58, 58)[:, 0]).squeeze()
        b_vx1 = (1/dy)*np.kron(v.reshape(58, 58)[0, :], e1).squeeze()
        b_vy1 = (1/dy)*np.kron(e1, v.reshape(58, 58)[:, 0]).squeeze()
        # print("shapes: ", b_ux1.shape, b_vx1.shape, b_vy1.shape, b_uy1.shape)


        ut = - u * (M.dot(u) - b_ux1) - v * (N.dot(u) - b_uy1)
        vt = - u * (M.dot(v) - b_vx1) - v * (N.dot(v) - b_vy1)
        # print("product: ", M.dot(u).shape)

        f = np.zeros((2, 60, 60))
        f[0, 1:59, 1:59] = ut.reshape(58, 58)
        f[1, 1:59, 1:59] = vt.reshape(58, 58)

        xt = self.prev + dt * f.reshape(-1)
        # print(xt.shape)
        return xt

# Device configuration
device = torch.device('cuda')
print("device is: ", device)
WM_PROJECT = "../../"

# snapshots have to be clipped before
snap_init = np.load(WM_PROJECT + "npSnapshots.npy")
assert np.min(snap_init) >= 0., "Snapshots should be clipped"

# true_latent = np.load(WM_PROJECT+"snapshotsConvAeTrueProjection.npy")
nonintr = np.load(WM_PROJECT+"nonIntrusiveCoeffConvAe.npy")
print("shaoes:", nonintr.shape)
n_total = snap_init.shape[1]
n_train = n_total-n_total//6

# scale the snapshots
nor = Normalize(snap_init, center_fl=True)
snap_framed = nor.framesnap(snap_init)
snap_vector = nor.vectorize2d(snap_framed)

# snap_scaled = nor.scale(snap_framed)
# snap_torch = torch.from_numpy(snap_scaled).to("cpu", dtype=torch.float)

inputs = torch.from_numpy(np.load("latent_initial_8.npy")).to(device, dtype=torch.float)
decoder = torch.jit.load("./decoder_gpu_8.pt")
print("latent init: ", inputs)
init = decoder(inputs).squeeze()
residual = Residual(decoder, inputs, 1.1360751, snap_vector[0, :], device)

# reproduce in pytorch
inputs_torch = inputs
print("inputs shape", inputs_torch.shape)
inputs_repeated = inputs_torch.repeat(3600, 1).requires_grad_(True)
print("inputs shape", inputs_repeated.shape)
grad_output = torch.eye(7200).to(device, dtype=torch.float)
print("ATTE", grad_output.type())
output = decoder(inputs_repeated.to(device, dtype=torch.float))

jac_torch = torch.zeros([7200, 8])
for i in range(2):
    output.backward(grad_output[i*3600:(i+1)*3600, :], retain_graph=True)
    jac_torch[i*3600:(i+1)*3600, :] = torch.autograd.grad(output, inputs_repeated, grad_output[i*3600:(i+1)*3600, :], retain_graph=True, create_graph=True)[0]
    # jac_torch[i*100:(i+1)*100, :] = inputs_repeated.grad.data
print("pytorch shape", jac_torch.shape)

#plot from torch
plot_c(jac_torch.detach().numpy(), "Decoder Jacobian from pytorch jit scripted")
del inputs_repeated
del output
del grad_output
del jac_torch

for i in range(1000):
    res = least_squares(residual, residual.latent)
    print("time: ", i, "new x: ", res.x, "norm: ", np.linalg.norm(res.fun))
    print("nonintr: ", nonintr[i, :])
    residual.update(res.x)
    if i%100 == 0:
        plot_snapshot(nor.frame2d(decoder(torch.from_numpy(nonintr[i, :]).to(device, dtype=torch.float)).detach().cpu().numpy()), 0)

plot_snapshot(nor.frame2d(residual.prev), 0)