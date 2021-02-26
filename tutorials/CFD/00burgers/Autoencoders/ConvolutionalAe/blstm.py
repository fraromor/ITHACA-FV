import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pyro.poutine as poutine
from convae import *
torch.set_default_tensor_type('torch.DoubleTensor')

WM_PROJECT = "../../"
HIDDEN_DIM = 2
DOMAIN_SIZE = 60
DIM = 2

# We'll ue this helper to check our models are correct.
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)

class ReducedCoeffsTimeSeries(torch.nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=HIDDEN_DIM,
                 hidden_dim=600,
                 n_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim,
                                  hidden_dim,
                                  n_layers,
                                  batch_first=True)

        self.encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ELU(), nn.Linear(hidden_dim//2, output_dim))

        # self.lstm.weight_hh_l0 = pyro.nn.PyroSample(pyro.distributions.Normal(loc=torch.zeros_like(self.lstm.weight_hh_l0), scale=torch.ones_like(self.lstm.weight_hh_l0)))

        # for name, param in self.named_pyro_params():
        #     print (name)
        #     self.state_dict()[name] = pyro.nn.PyroSample(pyro.distributions.Normal(loc=torch.zeros_like(param),
        #     scale=torch.ones_like(param)))

    def forward(self, inputs):
        z = self.lstm(inputs)[0]
        mean = self.encoder(z)
        return mean

def bayes_model(x, y, bnnlstm, decoder):
    mean = bnnlstm(x).reshape(-1, HIDDEN_DIM)
    scale = pyro.sample("sigma", dist.Uniform(0, 0.5))
    rec_snap = decoder(mean.reshape(-1, HIDDEN_DIM))

    obs = pyro.sample("obs", dist.Normal(rec_snap, scale).to_event(1), obs=y.reshape(-1, DIM*DOMAIN_SIZE**2))

    # print("Obs", scale.item(), "\n", z.detach().cpu().numpy()[0, :], "\n", z1.detach().cpu().numpy()[0, :],"\n", y.reshape(-1).detach().cpu().numpy()[:4])
    # z = dist.Normal(mean.reshape(-1, 4), scale).to_event(1)
    # print("test", dist.Normal(mean.reshape(-1, 4), scale).to_event(1).shape)

    # with pyro.plate("data", mean.shape[0], dim=-2):
    #     z = dist.Normal(mean.reshape(-1, 4), scale).to_event(1)
    #     obs = pyro.sample("obs", dist.Normal(mean.reshape(-1, 4), scale).to_event(1), obs=y.reshape(-1, 4))

    # print(z.shape, z.batch_shape, z.event_shape)
    return obs

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
ae = AE(
    HIDDEN_DIM,
    scale=(nor.min_sn, nor.max_sn),
    mean=nor.mean(device),
    domain_size=DOMAIN_SIZE).to(device)

ae.load_state_dict(torch.load("./model_"+str(HIDDEN_DIM)+".ckpt"))
ae.eval()

model = ReducedCoeffsTimeSeries(
                 input_dim=2,
                 output_dim=HIDDEN_DIM,
                 hidden_dim=600,
                 n_layers=2)

pyro.nn.module.to_pyro_module_(model)

# Now we can attempt to be fully Bayesian:
for m in model.modules():
    for name, value in list(m.named_parameters(recurse=False)):
        setattr(m, name, pyro.nn.PyroSample(prior=dist.Normal(0, 0.001)
                                                .expand(value.shape)
                                                .to_event(value.dim())))

print("TRACE")
with poutine.trace() as tr:
    model(torch.ones([5, 2001, 2]))

trace = tr.get_trace()
trace.compute_log_prob()
print(trace.format_shapes())

# load training inputs
array=[]
with open("../../ITHACAoutput/Offline/Training/mu_samples_mat.txt") as f:
    for i, line in enumerate(f):
        array.append([*line.split()])

array_ = [[float(elem) for elem in item] for item in array]
x = np.array(array_)
print("inputs shape: ", x.shape)

nl_red_coeff = torch.from_numpy(np.load("nl_red_coeff.npy")).to(device)
print("nl_red_coeff", nl_red_coeff.shape)
# pyro_model = pyro.nn.PyroModule(model)

# for name, param in model.named_pyro_params():
#     print (name)#, param.data)
#     param = pyro.nn.PyroSample(pyro.distributions.Normal(loc=torch.zeros_like(param), scale=torch.ones_like(param)))

# print("store")
# for param in pyro.get_param_store():
#     print(param)

# for name, param in model.named_pyro_params():
#     print(param)

guide = pyro.infer.autoguide.AutoDiagonalNormal(bayes_model)

adam = pyro.optim.Adam({"lr": 0.1})
svi = SVI(bayes_model, guide, adam, loss=Trace_ELBO())

pyro.clear_param_store()
num_iterations = 1000

# LSTM
input_dim = x.shape[1]
hidden_dim = HIDDEN_DIM
print("HIDDEN DIM: ", hidden_dim)
n_layers = 2
n_train = 10000
n_train_params = 6
n_time_samples = 2001
val_list=[]

# dataloader
x = torch.from_numpy(x.reshape(n_train_params, n_time_samples, x.shape[1])).to(torch.device("cpu"))
output = nor.vectorize2d(snap_torch).reshape(6, 2001, 7200)
validation = nl_red_coeff.reshape(n_train_params, n_time_samples, HIDDEN_DIM).to(torch.device("cpu"))

# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
# for i in range(4):
#     for j in range(i, 4):
#         axes[i, j].scatter(nl_red_coeff[:, i], nl_red_coeff[:, j])
# plt.show()

val_input = x[4, :, :].unsqueeze(0).to(torch.device("cpu"))
x = torch.cat([x[:4, :, :], x[5:, :, :]]).to(torch.device("cpu"))

val_output  = validation[4, :, :].unsqueeze(0).to(torch.device("cpu"))
output = torch.cat([output[:4, :, :], output[5:, :, :]]).to(torch.device("cpu"))

print("train in out:", x.shape, output.shape)
print("test in out:", val_input.shape, val_output.shape)

for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(x.to(torch.device("cpu")), output.to(torch.device("cpu")), model, ae.decoder)

    if j % 2 == 0:
        guide_trace = poutine.trace(guide).get_trace(val_input[:].to(torch.device("cpu")))
        val_forwarded = poutine.replay(model, guide_trace)(val_input[:].to(torch.device("cpu")))
        # val_forwarded = guide(val_input[:])
        # print("guide", val_forwarded.shape)
        val_error = np.max(np.abs((val_forwarded.reshape(-1).detach().cpu().numpy
        ()-val_output.reshape(-1).detach().cpu().numpy())))
        val_list.append(val_error)
        # print("Obs\n", val_forwarded.reshape(-1).detach().cpu().numpy
        # ()[:4], "\n", val_output.reshape(-1).detach().cpu().numpy()[:4])
        print("[iteration %04d] loss: %.4f Validation loss: %.12f" % (j + 1, loss / (2001*5), val_error))