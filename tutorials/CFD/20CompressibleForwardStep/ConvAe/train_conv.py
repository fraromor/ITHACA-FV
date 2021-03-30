import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.modules.loss import HingeEmbeddingLoss
from torchsummary import summary
import argparse
from convae import *
import clock
import time

DOMAIN_SIZE = 64
DIM = 4  # number of fields
WM_PROJECT = "../"


def plot_compare_spectra(snap,
                         snap_rec,
                         snap_error,
                         idx,
                         epoch,
                         idx_coord=0,
                         title='bu',
                         save=True):

    snap = snap.reshape(-1, 4, 64**2)
    snap_rec = snap_rec.reshape(-1, 4, 64**2)
    snap_error = snap_error.reshape(-1, 4, 64**2)

    dict = {'0': "U0", '1': "U1", '2': "rho", '3': "e"}
    for i in range(4):
        plt.plot(snap_rec[idx, i, :], label="Reconstructed"+str(dict[str(i)]))
        # plt.plot(snap_error[idx, :], label="max rel error")
        plt.plot(snap[idx, i, :], label="True"+str(dict[str(i)]), alpha=0.5)
    plt.legend()
    plt.title(str(np.max(snap_error)))

    if save:
        plt.savefig('./data/' + title + "_" + str(epoch) + '.png')
        plt.close()
    else:
        plt.show()


def plot_spectra(snapshots):
    snap = snapshots.reshape(-1, 4, 64**2)
    means = np.mean(snap, axis=0)
    stds = np.std(snap, axis=0)

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)
    dict = {'0': "U0", '1': "U1", '2': "rho", '3': "e"}
    for i in range(4):
        ax.plot(means[i, :], label=dict[str(i)])
        ax.fill_between(range(64**2),
                        means[i, :] - stds[i, :],
                        means[i, :] + stds[i, :],
                        alpha=0.2)

    plt.xlabel("compressed snapshot component from 1 to 64^2")
    plt.ylabel("value of the component")
    plt.title("Spectral analysis of the compressed snapshots")
    plt.legend()
    plt.show()

    idxs = iter([100 * i for i in range(1, 11)])
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(2):
        for j in range(5):
            ix = next(idxs)
            axes[i, j].plot(snap[i*(1+j), 0, :]-snap[i*j, 0, :], label="U0_" + str(ix))
            # axes[i, j].plot(snap[0, 0, :], label="U0_0", alpha=0.1)
            axes[i, j].legend()
    plt.show()


def load_evals(path):
    array = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i > 1:
                array.append(*line.split(','))

    array_ = [float(item) for item in array]
    return np.array(array_)


def main(args):
    NUM_EPOCHS = args.num_epochs
    HIDDEN_DIM = args.latent_dim
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    ORDER = 4
    LOAD = eval(args.load)
    SAVE = eval(args.save)
    checkpoint_dir = "./checkpoint/"
    model_dir = "./model/"

    # Device configuration
    device = torch.device('cuda' if eval(args.device) else 'cpu')
    print("device is: ", device, "latent dim is: ", HIDDEN_DIM)

    # snapshots have to be clipped before
    snapshots = np.load("../compressedSnap.npy").reshape(-1, 4, 64, 64)
    test_snapshots = np.load("../compressedSnapTest.npy").reshape(
        -1, 4, 64, 64)
    n_snap = snapshots.shape[0]
    validation_perecentage = 0.1
    # plot_spectra(snapshots)

    print("snapshots shapes: ", snapshots.shape)
    print("test snapshots shapes: ", test_snapshots.shape)

    # load eigenvalues
    eigU0 = load_evals(
        "../ITHACAoutput/POD/Eigenvalues_U.component(0)")[:64**2]
    eigU1 = load_evals(
        "../ITHACAoutput/POD/Eigenvalues_U.component(1)")[:64**2]
    eigrho = load_evals("../ITHACAoutput/POD/Eigenvalues_rho")[:64**2]
    eige = load_evals("../ITHACAoutput/POD/Eigenvalues_e")[:64**2]

    # scale w.r.t. inverse eigenvalues
    # evals = np.hstack((eigU0, eigU1, eigrho, eige)).reshape(1, 4, 64, 64)
    # print("ev", eigU0[-5:])
    # norm_evals = evals / np.sum(evals.reshape(4, 64**2), axis=1).reshape(
    #     1, 4, 1, 1)
    # snapshots = snapshots / evals
    # test_snapshots = test_snapshots / evals

    evals = np.clip(
        np.hstack((eigU0, eigU1, eigrho, eige)).reshape(1, 4, 64, 64), 1e-9,
        None)
    print("ev", eigU0[-5:])
    # snapshots[:, :, :, :-1] = snapshots[:, :, :, :-1] / evals[:, :, :, :-1]
    # test_snapshots[:, :, :, :-1] = test_snapshots[:, :, :, :-1] / evals[:, :, :, :-1]
    # snapshots[:, :, :, -1] = 0
    # test_snapshots[:, :, :, -1] = 0
    # norm_evals = evals / np.sum(evals.reshape(4, 64**2), axis=1).reshape(
    #     1, 4, 1, 1)

    # mean normalization
    # snapshots = snapshots - snapshots[0]
    # test_snapshots = test_snapshots - snapshots[0]
    # plot_spectra(snapshots)
    # plot_spectra(test_snapshots)

    # linear scaling
    # snapshots = snapshots * np.arange(64**2, 0, -1).reshape(1, 1, 64, 64)
    # test_snapshots = test_snapshots * np.arange(64**2, 0, -1).reshape(
    #     1, 1, 64, 64)
    # plot_spectra(snapshots)

    # normalize channel wise
    min_sn = np.min(snapshots.reshape(-1, 4 * 64 * 64).T.reshape(4, -1),
                    axis=1)
    max_sn = np.max(snapshots.reshape(-1, 4 * 64 * 64).T.reshape(4, -1),
                    axis=1)
    print("min, max train check: ", min_sn, max_sn)

    snapshots = (snapshots - min_sn.reshape(1, -1, 1, 1)) / (
        max_sn.reshape(1, -1, 1, 1) - min_sn.reshape(1, -1, 1, 1))
    test_snapshots = (test_snapshots - min_sn.reshape(1, -1, 1, 1)) / (
        max_sn.reshape(1, -1, 1, 1) - min_sn.reshape(1, -1, 1, 1))

    min_sn_after = np.min(snapshots.reshape(-1, 4 * 64 * 64).T.reshape(4, -1),
                          axis=1)
    max_sn_after = np.max(snapshots.reshape(-1, 4 * 64 * 64).T.reshape(4, -1),
                          axis=1)
    min_sn_test = np.min(test_snapshots.reshape(-1,
                                                4 * 64 * 64).T.reshape(4, -1),
                         axis=1)
    max_sn_test = np.max(test_snapshots.reshape(-1,
                                                4 * 64 * 64).T.reshape(4, -1),
                         axis=1)

    print("min, max train after check: ", min_sn_after, max_sn_after)
    print("min, max test check: ", min_sn_test, max_sn_test)
    # plot_spectra(snapshots)
    # plot_spectra(test_snapshots)

    # normalize component wise
    # min_sn = np.zeros((4, 64**2))
    # max_sn = np.zeros((4, 64**2))
    # for i in range(64**2):
    #     min_sn[:, i] = np.min(snapshots.reshape(-1, 4 * 64 * 64).T.reshape(
    #         4, 64**2, -1)[:, i, :],
    #                           axis=1)
    #     max_sn[:, i] = np.max(snapshots.reshape(-1, 4 * 64 * 64).T.reshape(
    #         4, 64**2, -1)[:, i, :],
    #                           axis=1)
    # print("min, max train check: ", min_sn.shape, max_sn.shape)

    # snapshots = (snapshots - min_sn.reshape(1, 4, 64, 64)) / (
    #     max_sn.reshape(1, 4, 64, 64) - min_sn.reshape(1, 4, 64, 64))
    # test_snapshots = (test_snapshots - min_sn.reshape(1, 4, 64, 64)) / (
    #     max_sn.reshape(1, 4, 64, 64) - min_sn.reshape(1, 4, 64, 64))

    # min_sn_after = np.zeros((4, 64**2))
    # max_sn_after = np.zeros((4, 64**2))
    # for i in range(64**2):
    #     min_sn_after[:,
    #                  i] = np.min(snapshots.reshape(-1, 4 * 64 * 64).T.reshape(
    #                      4, 64**2, -1)[:, i, :],
    #                              axis=1)
    #     max_sn_after[:,
    #                  i] = np.max(snapshots.reshape(-1, 4 * 64 * 64).T.reshape(
    #                      4, 64**2, -1)[:, i, :],
    #                              axis=1)
    # min_sn_test = np.zeros((4, 64**2))
    # max_sn_test = np.zeros((4, 64**2))
    # for i in range(64**2):
    #     min_sn_test[:, i] = np.min(test_snapshots.reshape(
    #         -1, 4 * 64 * 64).T.reshape(4, 64**2, -1)[:, i, :],
    #                                axis=1)
    #     max_sn_test[:, i] = np.max(test_snapshots.reshape(
    #         -1, 4 * 64 * 64).T.reshape(4, 64**2, -1)[:, i, :],
    #                                axis=1)

    # print("min, max train after check: ", np.min(min_sn_after),
    #       np.max(max_sn_after))
    # print("min, max test check: ", min_sn_test, max_sn_test)
    # plot_spectra(snapshots)
    # plot_spectra(test_snapshots)

    # Data loader nested linear
    # train_snap, val_torch = torch.utils.data.random_split(
    #     torch.from_numpy(snapshots.reshape(-1, 4, 64**2)[:, 0, :]),
    #     [9000, n_snap - 9000])

    # Data loader convolutional
    train_snap, val_torch = torch.utils.data.random_split(
        torch.from_numpy(snapshots), [18000, n_snap - 18000])
    del snapshots

    train_loader = torch.utils.data.DataLoader(dataset=train_snap,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    # validation set normalization
    val_np = val_torch[:].detach().cpu().numpy()
    # val_norm = np.max(nor.rescale(val_np))
    # print("validation set norm", val_norm)

    # test set normalization nested linear
    # snap_test_torch = torch.from_numpy(test_snapshots[:, 0, :, :].reshape(
    #     -1, 64**2))
    # test_np = snap_test_torch[:].detach().cpu().numpy()
    snap_test_torch = torch.from_numpy(test_snapshots).to("cpu")
    test_np = snap_test_torch[:].detach().cpu().numpy()
    del test_snapshots

    # convolutional ae
    print("CUDA: ",
          torch.cuda.memory_reserved(device) / 1000000,
          torch.cuda.memory_allocated(device) / 1000000)
    model = AE(HIDDEN_DIM).to(device)
    summary(model, input_size=(DIM, DOMAIN_SIZE, DOMAIN_SIZE))

    print("CUDA: ",
          torch.cuda.memory_reserved(device) / 1000000,
          torch.cuda.memory_allocated(device) / 1000000)

    # nested linear ae
    # model = AEFrequency(HIDDEN_DIM).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500, 1500])

    # load model
    if LOAD is True:
        # ckp_path = "./checkpoint/checkpoint.pt"
        ckp_path = "./model/best_model.pt"
        model, _, start_epoch = load_ckp(ckp_path, model)
        print("loaded", start_epoch)

    if SAVE is True:
        # Save the model checkpoint
        torch.save(model.state_dict(), 'model_' + str(HIDDEN_DIM) + '.ckpt')

        sm = torch.jit.script(model)
        sm.save('model_gpu_' + str(HIDDEN_DIM) + '.pt')
        # Save the initial value of the latent variable
        initial = model.encoder(snap_torch[:1, :, :, :].to(
            device,
            dtype=torch.float)).detach().to(torch.device('cpu'),
                                            dtype=torch.double).numpy()
        print("initial latent variable shape : ", initial)
        np.save("latent_initial_" + str(HIDDEN_DIM) + ".npy", initial)

        # Save decoder
        model.decoder.to(device)
        sm = torch.jit.script(model.decoder)
        sm.save('decoder_gpu_' + str(HIDDEN_DIM) + '.pt')

        # reproduce in pytorch
        # inputs_torch = torch.from_numpy(initial)#.transpose(0, 1)
        # print("inputs shape", inputs_torch.shape)
        # inputs_repeated = inputs_torch.repeat(3600, 1).requires_grad_(True)
        # print("inputs shape", inputs_repeated.shape)
        # grad_output = torch.eye(7200).to(device, dtype=torch.float)
        # print("ATTE", grad_output.type())
        # decoder = torch.jit.load("./decoder_gpu_4.pt")
        # output1 = model.decoder(inputs_repeated.to(device, dtype=torch.float))
        # output = decoder(inputs_repeated.to(device, dtype=torch.float))
        print("saved")

    loss_list = []
    val_list = []
    test_list = []
    loss = 1
    norm_average_loss = len(train_loader)
    best = 3000
    start_epoch = 1
    it = 0
    loss_val = 1

    # weighted loss
    weights = lambda theta: norm_evals + theta * (np.ones(norm_evals.shape) -
                                                  norm_evals)

    # Train the model
    start = time.time()
    factor = 0.
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_loss = 0

            for snap in train_loader:

                snap = snap.to(device, dtype=torch.float)
                outputs = model(snap)
                loss = criterion(outputs, snap)  #+ regularizerl1(
                #   model, device, factor=factor
                # )  # + regularizerl2(model, device, factor=0.0005)
                # loss = torch.sum(
                #     torch.from_numpy(weights(epoch / NUM_EPOCHS)).to(device) *
                #     (outputs - snap)**2)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()

            # scheduler.step()
            mean_epoch_loss = epoch_loss / norm_average_loss
            loss_list.append(mean_epoch_loss)

            # validation loss evaluation
            val_cuda = val_torch[:].to(device, dtype=torch.float)
            outputs_val = model(val_cuda).detach().cpu().numpy()
            del val_cuda
            diff_val = np.abs(outputs_val - val_np)
            # print("diff", diff.shape)
            index_val = np.argmax(diff_val.reshape(
                -1, 4 * DOMAIN_SIZE**2)) // (4 * DOMAIN_SIZE**2)
            loss_val = np.max(diff_val)
            val_list.append(loss_val)

            # test error evaluation
            test_cuda = snap_test_torch[:].to(device, dtype=torch.float)
            outputs_test = model(test_cuda).detach().cpu().numpy()
            del test_cuda
            diff_test = np.abs(outputs_test - test_np) / max_sn_test.reshape(
                1, 4, 1, 1)
            index_test = np.argmax(diff_test.reshape(
                -1, 4 * DOMAIN_SIZE**2)) // (4 * DOMAIN_SIZE**2)
            loss_test = np.max(diff_test)
            test_list.append(loss_test)

            # if loss_val < 0.0015:
            # optimizer.param_groups[0]['lr'] = LEARNING_RATE * 0.1

            if epoch > 500 and loss_val < 0.001:
                break

            # plt.ion()
            if epoch % args.iter == 0:
                print(
                    'Epoch [{}/{}], Time: {:.2f} s, Loss: {:.10f}\n Validation Loss: {:.6f} Test Loss: {:.6f}'
                    .format(epoch, NUM_EPOCHS,
                            time.time() - start, mean_epoch_loss, loss_val,
                            loss_test), "regularizers l1 l2: ",
                    regularizerl1(model, device, factor=factor).item(),
                    regularizerl2(model, device, factor=factor).item())

                start = time.time()

                #validation plot where validation loss is worst
                # index_list = torch.sort(index, descending=True)[1]
                plot_compare_spectra(val_np,
                                     outputs_val,
                                     outputs_val - val_np,
                                     index_val,
                                     epoch,
                                     title="validation")
                plot_compare_spectra(test_np,
                                     outputs_test,
                                     outputs_test - test_np,
                                     index_test,
                                     epoch,
                                     title="test")

                # save checkpoints
                if mean_epoch_loss < best:
                    is_best = True
                    best = mean_epoch_loss
                    it = 0
                    print("BEST CHANGED")
                else:
                    is_best = False
                    it += 1
                    if (it > 200 and epoch > 300
                            and optimizer.param_groups[0]['lr'] > 1e-5) or (
                                it > 50 and epoch > 5 and epoch < 300
                                and optimizer.param_groups[0]['lr'] > 1e-5):
                        # optimizer.param_groups[0]['lr'] *= 0.5
                        print("LR CHANGED: ", optimizer.param_groups[0]['lr'])
                        it = 0

                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
                is_best = False

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
        summary(model, input_size=(DIM, DOMAIN_SIZE, DOMAIN_SIZE))

        # Save the initial value of the latent variable
        initial = model.encoder(snaps_torch[:1, :, :, :].to(
            device,
            dtype=torch.float)).detach().to(torch.device('cpu'),
                                            dtype=torch.double).numpy()
        print("initial latent variable shape : ", initial)
        np.save("latent_initial_" + str(HIDDEN_DIM) + ".npy", initial)

        # Save decoder
        model.decoder.to(device)
        sm = torch.jit.script(model.decoder)
        sm.save('decoder_gpu_' + str(HIDDEN_DIM) + '.pt')

        # device = 'cpu'
        # model.decoder.to(device)
        # sm = torch.jit.script(model.decoder)
        # sm.save('decoder_' + str(HIDDEN_DIM) + '.pt')

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
                        default=20,
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
