import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
import torch
from torch._C import dtype
import torch.nn as nn
from torchsummary import summary
import argparse
from convae import *
import clock
import time
import sys
import GPy
from athena import rrmse

DOMAIN_SIZE = 60
DIM = 2  # number of components of velocity field
WM_PROJECT = "../../"


def main(args):
    NUM_EPOCHS = args.num_epochs
    HIDDEN_DIM = args.latent_dim
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    LOAD = eval(args.load)
    SAVE = eval(args.save)
    checkpoint_dir = "./checkpointDEIM/"
    model_dir = "./modelDEIM/"

    # Device configuration
    device = torch.device('cpu')
    print("device is: ", device, "latent dim is: ", HIDDEN_DIM)

    # load deim indices
    deim_idx = np.load(WM_PROJECT + "indices.npy").squeeze()
    print("original deim indexes, with repetitions: ", deim_idx.shape)
    lis = []
    for item in list(deim_idx):
        if item >= 0: lis.append(item)
    deim_idx = np.array(lis)
    print(
        "original deim indexes, with repetitions and without repeated B indices: ",
        deim_idx.shape)

    deim_idx_set = set()
    for id in range(deim_idx.shape[0]):
        if np.isnan(deim_idx[id]) >= 0:
            deim_idx_set.add(int(deim_idx[id]))
    print("len of set of deim indexes, without repetitions: ",
          deim_idx_set.__len__())
    deim_idx_set_xy = set()
    for item in deim_idx_set:
        deim_idx_set_xy.add(int(item))
        deim_idx_set_xy.add(int(item) + 3600)
    print("xy deim indexes length: ", deim_idx_set_xy.__len__())

    # create map from set indices to mdeim indices
    Mconvert = torch.zeros((2 * deim_idx.shape[0], deim_idx_set_xy.__len__()))
    sortedIndices = sorted(deim_idx_set)
    for i, id_ in enumerate(sortedIndices):
        # print(i, id_)
        for j, idx in enumerate(list(deim_idx)):
            # print("inside", j, idx)
            if idx == id_:
                # print("j: ", j, j+deim_idx.shape[0])
                Mconvert[j, i] = 1
                Mconvert[j + deim_idx.shape[0], i] = 1

    # snapshots have to be clipped before
    snap_vec = np.load(WM_PROJECT + "npSnapshots.npy")
    assert np.min(snap_vec) >= 0., "Snapshots should be clipped"
    print("snap vec: ", snap_vec.shape)
    n_total = snap_vec.shape[1]
    n_train = n_total - n_total // 10

    # scale the snapshots
    nor = Normalize(snap_vec, center_fl=True)
    snap_framed = nor.framesnap(snap_vec)
    snap_scaled = nor.scale(snap_framed)
    snap_torch = torch.from_numpy(snap_scaled).to("cpu", dtype=torch.float)
    print("snapshots shape", snap_scaled.shape)
    print("Min max after scaling: ", np.min(snap_scaled), np.max(snap_scaled))

    # load autoencoder
    model = AE(HIDDEN_DIM,
               scale=(nor.min_sn, nor.max_sn),
               mean=nor.mean(device),
               domain_size=DOMAIN_SIZE,
               use_cuda=args.device).to(device, dtype=torch.float)

    checkpoint = torch.load("./model/best_model.pt")
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(torch.load('model_' + str(HIDDEN_DIM) + '.ckpt'))
    model.eval()
    # summary(model, input_size=(DIM, DOMAIN_SIZE, DOMAIN_SIZE))

    # evaluate hidden variables
    nl_red_coeff = model.encoder.forward(snap_torch).detach().cpu().numpy()
    print("latent coeff shape: ", nl_red_coeff.shape)
    snap_rec = model.forward(snap_torch).detach().cpu().numpy()
    print("snap rec shape: ", snap_rec.shape)

    n_total = snap_rec.shape[0]
    n_val = n_total // 20
    n_train = n_total - n_val
    print("n train , n val. ", n_train, n_val)

    mask = np.zeros_like(snap_rec)
    for id in deim_idx_set_xy:
        mask[:, id] = 1
    mask = mask == 1
    print("mask: ", mask[:, 913], mask.shape)
    snap_rect = snap_rec[mask].reshape(12006, len(deim_idx_set_xy))
    print("test", snap_rec[mask].shape)
    print("restricted snap rec: ", snap_rect.shape)
    snap_rect = snap_rect
    # for i in range(100, len(deim_idx_set_xy)):
    #     plt.plot(range(n_total), snap_rect[:, i])
    # plt.show()

    # GPR
    # nl_red_coeff = nl_red_coeff.detach().cpu().numpy()
    # nl_red_coeff_tr, nl_red_coeff_val = nl_red_coeff[:n_train],  nl_red_coeff[:n_val]
    # snap_rec_tr, snap_rec_val = snap_rec[:n_train],  snap_rec[:n_val]
    # gpr = GPy.models.GPRegression(nl_red_coeff_tr, snap_rec_tr)
    # gpr.optimize_restarts(10)
    # predicted = gpr.predict(nl_red_coeff_val)[0]
    # print("GPR error: ", rrmse(predicted, snap_rec_val))

    # # Data loader
    # class burgers2D(torch.utils.data.Dataset):
    #     def __init__(self, data):
    #         self.data = data
    #     def __getitem__(self, key):
    #         return self.data[key]
    #     def __len__(self):
    #         return self.data

    device = torch.device('cuda')
    print("in, out: ", nl_red_coeff.shape, snap_rect.shape)
    nl_red_coeff_ = torch.Tensor(nl_red_coeff)
    snap_rect_ = torch.Tensor(snap_rect)
    train_snap, val_torch = torch.utils.data.random_split(
        list(zip(nl_red_coeff_, snap_rect_)), [n_train, n_total - n_train])

    print("train loader and val loader length: ", len(train_snap),
          len(val_torch))

    train_loader = torch.utils.data.DataLoader(dataset=train_snap,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    # validation set normalization
    # print(len(val_torch.indices), len(val_torch.dataset))
    val_np, val_out_np = zip(*(val_torch.dataset))
    val_np = torch.stack(val_np)
    val = val_np[val_torch.indices]
    # print("debug:", val_np.shape)
    val_out_np = torch.stack(val_out_np).detach().cpu().numpy()
    val_out_np = val_out_np[val_torch.indices]
    print("val_np.shape: ", val_out_np.shape)
    val_norm = np.max(val_out_np)
    print("validation set norm", val_norm)

    # start model
    model = DeimSurrogate(args.latent_dim, 100,
                          deim_idx_set_xy.__len__()).to(device,
                                                        dtype=torch.float)

    # Loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500, 1500])

    # load model
    if LOAD is True:
        # ckp_path = "./checkpointDEIM/checkpoint.pt"
        ckp_path = "./modelDEIM/best_model.pt"
        model, _, start_epoch = load_ckp(ckp_path, model)
        print("loaded", start_epoch)

    if SAVE is True:
        # Save the model checkpoint
        torch.save(model.state_dict(),
                   'modelDEIM_' + str(HIDDEN_DIM) + '.ckpt')

        inp = torch.ones(HIDDEN_DIM).reshape(1, -1)
        model.to(torch.device("cpu"))
        start = time.time()
        model(inp.to(torch.device("cpu"), dtype=torch.float))
        print("Elapsed CPU: ", (time.time() - start) * 1000, " ms")

        model.to(torch.device("cuda"))
        start = time.time()
        model(inp.to(torch.device("cuda"), dtype=torch.float))
        print("Elapsed GPU: ", (time.time() - start) * 1000, " ms")

        mod_gpu = torch.jit.script(model)
        mod_cpu = torch.jit.script(model.to(torch.device("cpu")))
        mod_gpu.save('modelDEIM_gpu_' + str(HIDDEN_DIM) + '.pt')

        inp = torch.ones(HIDDEN_DIM).reshape(1, -1)
        mod_cpu.to(torch.device("cpu"))
        start = time.time()
        mod_cpu(inp.to(torch.device("cpu"), dtype=torch.float))
        print("Elapsed script CPU: ", (time.time() - start) * 1000, " ms")

        mod_gpu.to(torch.device("cuda"))
        start = time.time()
        mod_gpu(inp.to(torch.device("cuda"), dtype=torch.float))
        print("Elapsed script GPU: ", (time.time() - start) * 1000, " ms")

        class SaveMDEIM(nn.Module):
            def __init__(self, matrix, mdeim_model, device):
                super().__init__()
                self.device = device
                self.mat = matrix.to(self.device, dtype=torch.float)
                self.mdeim_model = mdeim_model.to(self.device)

            def forward(self, x):
                out = self.mdeim_model(x).to(self.device)
                # print("out", out.shape, self.mat.shape)
                res = torch.matmul(out, self.mat.T).to(self.device)
                # print("res", res.shape)
                return res

        saveMDEIM = SaveMDEIM(Mconvert.to(torch.device("cuda")),
                              model.to(torch.device("cuda")),
                              torch.device("cuda"))
        sm_gpu = torch.jit.script(saveMDEIM)
        sm_gpu.save('modelDEIM_gpu_' + str(HIDDEN_DIM) + '.pt')
        saveMDEIM_ = SaveMDEIM(Mconvert, model, torch.device("cpu"))
        sm_cpu = torch.jit.script(saveMDEIM_.to(torch.device("cpu")))

        inp = torch.ones(HIDDEN_DIM).reshape(1, -1)
        sm_cpu.to(torch.device("cpu"))
        start = time.time()
        sm_cpu(inp.to(torch.device("cpu"), dtype=torch.float))
        print("Elapsed script full CPU: ", (time.time() - start) * 1000, " ms")

        sm_gpu.to(torch.device("cuda"))
        start = time.time()
        sm_gpu(inp.to(torch.device("cuda"), dtype=torch.float))
        print("Elapsed script full GPU: ", (time.time() - start) * 1000, " ms")

        # test errors
        saveMDEIM.to(torch.device("cpu"))
        res = saveMDEIM(
            nl_red_coeff_.to(torch.device("cpu"), dtype=torch.float))
        snap_rect_repeated = torch.matmul(snap_rect_.to(dtype=torch.float),
                                          Mconvert.T.to(dtype=torch.float))
        print("shapes: ", res.shape, snap_rect_repeated.shape)
        print(
            "RMS: ",
            torch.max(
                res.to(torch.device("cuda")) -
                snap_rect_repeated.to(torch.device("cuda"))))
        print("saved")
        sys.exit()

    loss_list = []
    val_list = []
    test_list = []
    loss = 1
    norm_average_loss = len(train_loader)
    best = 1000000
    start_epoch = 1
    it = 0
    loss_val = 1
    val_l2 = []
    # Train the model
    start = time.time()
    factor = 0.000001
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_loss = 0
            # i=0
            for snap, snap_out in train_loader:
                # print("step: ", i, snap)
                # i+=1
                snap = snap.to(device, dtype=torch.float)
                outputs = model(snap)
                # print("shapes outputs, rec: ", snap_out.shape, outputs.shape)

                loss = criterion(outputs, snap_out.to(
                    device, dtype=torch.float))  # + regularizerl1(
                # model, device, factor=factor
                # )  # + regularizerl2(model, device, factor=0.0005)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()

            # scheduler.step()
            mean_epoch_loss = epoch_loss / norm_average_loss
            loss_list.append(mean_epoch_loss)

            # validation loss evaluation
            # print("val.shape: ", val.shape)
            val_cuda = val.to(device, dtype=torch.float)
            outputs_val = model(val_cuda).detach().cpu().numpy()
            del val_cuda
            diff = np.abs(outputs_val - val_out_np)
            # print("diff, norm: ", np.max(diff), val_norm)
            loss_val = np.max(diff) / val_norm
            index_val = np.argmax(diff.reshape(-1)) % diff.shape[1]
            val_list.append(loss_val)
            l2_val = np.linalg.norm(outputs_val - val_out_np) / val_norm
            val_l2.append(l2_val)

            # test error evaluation
            # snap_true_cuda = snap_true_torch[:].to(device, dtype=torch.float)
            # snap_true_rec = nor.frame2d(
            #     model(snap_true_cuda)).detach().cpu().numpy()
            # del snap_true_cuda
            # err = np.abs(snap_true_rec - nor.rescale(snap_true_scaled))
            # error_proj = np.linalg.norm(nor.vectorize2d(err), axis=1)
            # error_proj_max = np.max(nor.vectorize2d(err), axis=1)

            # # relative test errors L2 and max
            # error_proj = error_proj / test_norm
            # error_proj_max = error_proj_max / test_max_norm

            # error_max_mean = np.max(error_proj_max)
            # error_mean = np.mean(error_proj)
            # error_max = np.max(error_proj)
            # error_min = np.min(error_proj)

            # test_list.append(error_mean)

            # if loss_val < 0.0015: optimizer.param_groups[0]['lr'] =
            # LEARNING_RATE * 0.1

            # if epoch > 500 and loss_val < 0.001:
            #     break

            # plt.ion()
            if epoch % args.iter == 0:
                print(
                    'Epoch [{}/{}], Time: {:.2f} s, Loss: {:.10f}\n Validation, Loss: {:.6f}, {:.6f}\n'
                    .format(epoch, NUM_EPOCHS,
                            time.time() - start, mean_epoch_loss, loss_val,
                            l2_val))
                #         , "regularizers l1 l2: ",
                # regularizerl1(model, device, factor=factor).item(),
                # regularizerl2(model, device, factor=factor).item())

                start = time.time()

                plt.plot(outputs_val[:, index_val], label="predicted")
                plt.plot(val_out_np[:, index_val], label="true")
                plt.plot(outputs_val[:, index_val] - val_out_np[:, index_val],
                         label="error")
                plt.legend()
                plt.tight_layout()
                plt.savefig('./data/' + "compare" + "_" + str(epoch) + '.png')
                plt.close()

                # save checkpoints
                if loss_val < best:
                    is_best = True
                    best = loss_val
                    it = 0
                    print("BEST CHANGED")
                else:
                    is_best = False
                    it += 1
                    if (
                            loss_val < 0.14 or epoch > 200
                    ) and it > 300 and optimizer.param_groups[0]['lr'] > 2e-5:
                        optimizer.param_groups[0]['lr'] *= 0.5
                        print("LR CHANGED: ", optimizer.param_groups[0]['lr'])
                        it = 0
                    else:
                        if it > 50 and epoch > 10 and optimizer.param_groups[
                                0]['lr'] > 2e-5:
                            optimizer.param_groups[0]['lr'] *= 0.5
                            print("LR CHANGED: ",
                                  optimizer.param_groups[0]['lr'])
                            it = 0

                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
                is_best = False

                # # loss plot
                # loss_plot = torch.norm(outputs-snapshot.reshape(-1,
                # 2*DOMAIN_SIZE**2), p=ORDER, dim=0).reshape(1, 2, DOMAIN_SIZE,
                # DOMAIN_SIZE).detach().cpu().numpy() loss_plot_ =
                # (torch.max(torch.abs(outputs-snapshot.reshape(-1,
                # 2*DOMAIN_SIZE**2)), dim=0)[0]).reshape(1, 2, DOMAIN_SIZE,
                # DOMAIN_SIZE).detach().cpu().numpy() plot_snapshot(loss_plot,
                # 0) plot_two(loss_plot, loss_plot_, [0], epoch, title="loss")

                # reconstruction plot plt.show()
                # # plot_snapshot(max_sn*outputs.detach().cpu().numpy().reshape((-1, DIM, DOMAIN_SIZE, DOMAIN_SIZE)), 0)
                # plot_snapshot(max_sn*snapshot.detach().cpu().numpy().reshape((-1,
                # DIM, DOMAIN_SIZE, DOMAIN_SIZE)), 0) plt.show()

                # error plot plt.plot(range(epoch//args.iter),
                # np.log10(loss_list[::args.iter]))
                # plt.savefig('./train_cae.png') plt.draw() plt.pause(0.05)
                # plt.show()

        plt.subplot(3, 1, 1)
        plt.plot(range(1, len(loss_list) + 1), np.log10(loss_list))
        plt.ylabel('train ave L2^2')  # training average L2 squared error

        plt.subplot(3, 1, 2)
        plt.plot(range(1, len(val_list) + 1), np.log10(val_list))
        plt.ylabel('val max rel err')  # validation max relative error

        plt.subplot(3, 1, 3)
        plt.plot(range(1, len(test_list) + 1), np.log10(test_list))
        plt.xlabel('epochs')
        plt.ylabel('test L2 rel err')  # test L2 relative error

        plt.show()

        # Save the model checkpoint
        torch.save(model.state_dict(), 'model_' + str(HIDDEN_DIM) + '.ckpt')
        summary(model, input_size=[(HIDDEN_DIM)])

        # Save the initial value of the latent variable
        # initial = model.encoder(snaps_torch[:1, :, :, :].to(
        #     device, dtype=torch.float)).detach().to(torch.device('cpu'),
        #                                             dtype=torch.float).numpy()
        # print("initial latent variable shape : ", initial)
        # np.save("latent_initial_" + str(HIDDEN_DIM) + ".npy", initial)

        # # Save decoder
        # model.decoder.to(device)
        # sm = torch.jit.script(model.decoder)
        # sm.save('decoder_gpu_' + str(HIDDEN_DIM) + '.pt')

        # device = 'cpu' model.decoder.to(device) sm =
        # torch.jit.script(model.decoder) sm.save('decoder_' + str(HIDDEN_DIM) +
        # '.pt')

    except KeyboardInterrupt:
        plt.subplot(3, 1, 1)
        plt.plot(range(1, len(loss_list) + 1), np.log10(loss_list))
        plt.ylabel('train ave L2^2')

        plt.subplot(3, 1, 2)
        plt.plot(range(1, len(val_list) + 1), np.log10(val_list))
        plt.ylabel('val max rel err')

        plt.subplot(3, 1, 3)
        plt.plot(range(1, len(test_list) + 1), np.log10(test_list))
        plt.xlabel('epochs')
        plt.ylabel('test L2 rel err')
        plt.savefig("Loss" + str(time.ctime()).replace(" ", "") + ".png")

        # save checkpoints
        if loss_val < best:
            is_best = True
            best = loss_val
        else:
            is_best = False

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
        print("Saved checkpoint")

        # Save the initial value of the latent variable
        # initial = model.encoder(snaps_torch[:1, :, :, :].to(
        #     device,
        #     dtype=torch.float)).detach().to(torch.device('cpu'),
        #                                     dtype=torch.float).numpy()
        # print("initial latent variable shape : ", initial)
        # np.save("latent_initial_" + str(HIDDEN_DIM) + ".npy", initial)


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n',
                        '--num_epochs',
                        default=1000,
                        type=int,
                        help='number of training epochs')
    parser.add_argument('-lr',
                        '--learning_rate',
                        default=1.0e-3,
                        type=float,
                        help='learning rate')
    parser.add_argument('-batch',
                        '--batch_size',
                        default=100,
                        type=int,
                        help='learning rate')
    parser.add_argument('-dim',
                        '--latent_dim',
                        default=4,
                        type=int,
                        help='learning rate')
    parser.add_argument('-device',
                        '--device',
                        default='True',
                        type=str,
                        help='whether to use cuda')
    parser.add_argument('-load',
                        '--load',
                        default='False',
                        type=str,
                        help='whether to load the model')
    parser.add_argument('-save',
                        '--save',
                        default='False',
                        type=str,
                        help='whether to save the model')
    parser.add_argument('-i',
                        '--iter',
                        default=2,
                        type=int,
                        help='epoch when visualization runs')
    args = parser.parse_args()

    main(args)