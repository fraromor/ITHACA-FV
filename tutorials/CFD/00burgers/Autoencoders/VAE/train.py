import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
import argparse
from vae import *
import clock
import time

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam

DOMAIN_SIZE = 60
DIM = 2  # number of components of velocity field
WM_PROJECT = "../../"


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
    snap_vec = np.load(WM_PROJECT + "npSnapshots.npy")
    assert np.min(snap_vec) >= 0., "Snapshots should be clipped"

    # specify how many samples should be used for training and validation
    n_total = snap_vec.shape[1]
    n_train = n_total-n_total//10
    print("Dimension of validation set: ", n_total-n_train)

    # scale the snapshots
    nor = Normalize(snap_vec, center_fl=True)
    snap_framed = nor.framesnap(snap_vec)
    print("add constant solution: ", snap_framed.shape)
    snap_scaled = nor.scale(snap_framed)
    snaps_torch = torch.from_numpy(snap_scaled)
    print("snapshots shape", snap_scaled.shape)
    print("Min max after scaling: ", np.min(snap_scaled), np.max(snap_scaled))
    # for i in range(12000):
    #     plot_snapshot(nor.rescale(snaps_torch, snaps_torch.device).detach().cpu().numpy(), i)

    # Test snapshots
    snap_true_vec = np.load(WM_PROJECT + "npTrueSnapshots.npy")
    snap_true_scaled = nor.scale(nor.framesnap(snap_true_vec))
    np.save(WM_PROJECT + "npTrueSnapshots_framed.npy", nor.vectorize2d(snap_true_scaled))
    print("saved snapshots for consistency test libtorch", nor.vectorize2d(snap_true_scaled).shape)
    snap_true_torch = torch.from_numpy(snap_true_scaled)
    test_norm = np.linalg.norm(snap_true_vec, axis=0, keepdims=False)
    test_max_norm = np.max(snap_true_vec, axis=0, keepdims=False)
    print("test snapshots shape: ", snap_true_scaled.shape)
    print("max and min L2 norm", np.max(test_norm), np.min(test_norm))
    print("Min max after scaling: ", np.min(snap_true_scaled), np.max(snap_true_scaled))


    # Data loader
    train_snap, val_torch = torch.utils.data.random_split(
        snaps_torch, [n_train, n_total - n_train])
    # train_snap = train_snap[:]
    # # print("train_snap size", train_snap.shape)
    # train_snap = torch.cat((train_snap, torch.zeros((1 ,DIM, DOMAIN_SIZE, DOMAIN_SIZE))), axis=0)
    # print("train_snap size", train_snap.shape)

    train_loader = torch.utils.data.DataLoader(dataset=train_snap,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    # validation set normalization
    val_np = val_torch[:].detach().cpu().numpy()
    val_norm = np.max(nor.rescale(val_np))
    print("validation set norm", val_norm)

    # clear param store
    pyro.clear_param_store()

    # start model
    vae = VAE(
        HIDDEN_DIM,
        scale=(nor.min_sn, nor.max_sn),
        mean=nor.mean(device),
        domain_size=DOMAIN_SIZE,
        use_cuda=args.device,
        nor=nor).to(device)

    # Loss and optimizer
    optimizer = ClippedAdam({"lr":LEARNING_RATE})
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500, 1500])

    # load model
    if LOAD is True:
        ckp_path = "./checkpoint/checkpoint_vae"
        optim_path = "./checkpoint/checkpoint_optim"
        # ckp_path = "./model/best_model.pt"
        vae, optimizer = load_ckp(ckp_path, vae, optimizer, optim_path=optim_path)

    # if SAVE is True:
    #     # Save the model checkpoint
    #     torch.save(model.state_dict(), 'model_' + str(HIDDEN_DIM) + '.ckpt')

    #     sm = torch.jit.script(model)
    #     sm.save('model_gpu_' + str(HIDDEN_DIM) + '.pt')
    #     # Save the initial value of the latent variable
    #     initial = model.encoder(snaps_torch[:1, :, :, :].to(
    #         device,
    #         dtype=torch.float)).detach().to(torch.device('cpu'),
    #                                         dtype=torch.double).numpy()
    #     print("initial latent variable shape : ", initial)
    #     np.save("latent_initial_" + str(HIDDEN_DIM) + ".npy", initial)

    #     # Save decoder
    #     model.decoder.to(device)
    #     sm = torch.jit.script(model.decoder)
    #     sm.save('decoder_gpu_' + str(HIDDEN_DIM) + '.pt')

        # reproduce in pytorch
        # inputs_torch = torch.from_numpy(initial)#.transpose(0, 1)
        # print("inputs shape", inputs_torch.shape)
        # inputs_repeated = inputs_torch.repeat(3600, 1).requires_grad_(True)
        # print("inputs shape", inputs_repeated.shape)
        # grad_output = torch.eye(7200).to(device, dtype=torch.float)
        # print("ATTE", grad_output.type())
        # decoder = torch.jit.load("./decoder_gpu_8.pt")
        # output1 = model.decoder(inputs_repeated.to(device, dtype=torch.float))
        # output = decoder(inputs_repeated.to(device, dtype=torch.float))
        # print("saved")

    loss_list = []
    val_list = []
    test_list = []
    diag_list = []
    loss = 1
    norm_average_loss = len(train_loader)
    best = 1000
    start_epoch = 1
    it = 0
    loss_val = 1
    TEST_FREQUENCY = 5

    # Train the model
    start = time.time()
    try:
        # training loop
        for epoch in range(NUM_EPOCHS):
            # initialize loss accumulator
            epoch_loss = 0.
            # do a training epoch over each mini-batch x returned
            # by the data loader
            for x in train_loader:
                # if on GPU put mini-batch into CUDA memory
                x = x.to(device, dtype=torch.float)
                # do ELBO gradient and accumulate loss
                epoch_loss += svi.step(x)

            mean_epoch_loss = total_epoch_loss_train = epoch_loss / norm_average_loss
            loss_list.append(-mean_epoch_loss)
            # print("[epoch %03d] average training loss: %.4f" % (epoch, mean_epoch_loss))

            # if epoch % TEST_FREQUENCY == 0:
            #     # report test diagnostics
            #     # initialize loss accumulator
            #     diag_loss = 0.
            #     # do a training epoch over each mini-batch x returned
            #     # by the data loader
            #     for x in train_loader:
            #         # if on GPU put mini-batch into CUDA memory
            #         x = x.to(device, dtype=torch.float)
            #         # do ELBO gradient and accumulate loss
            #         diag_loss += svi.step(x)
            #     mean_epoch_loss_diag = total_epoch_loss_train = epoch_loss / norm_average_loss
            #     diag_list.append(-mean_epoch_loss_diag)
                # print("[epoch %03d] average test loss: %.4f" % (epoch, mean_epoch_loss_diag))

            # validation loss evaluation
            val_cuda = val_torch[:].to(device, dtype=torch.float)
            outputs_val = vae.reconstruct(val_cuda)[0].detach().cpu().numpy()
            del val_cuda
            diff = np.abs(nor.frame2d(outputs_val) - nor.rescale(val_np))
            index = np.argmax(nor.vectorize2d(diff)) // (DOMAIN_SIZE**2 * DIM)
            loss_val = np.max(diff) / val_norm
            val_list.append(loss_val)

            # test error evaluation
            snap_true_cuda = snap_true_torch[:].to(device, dtype=torch.float)
            snap_true_rec = nor.frame2d(
                vae.reconstruct(snap_true_cuda)[0]).detach().cpu().numpy()
            del snap_true_cuda
            err = np.abs(snap_true_rec - nor.rescale(snap_true_scaled))
            error_proj = np.linalg.norm(nor.vectorize2d(err), axis=1)
            error_proj_max = np.max(nor.vectorize2d(err), axis=1)


            # relative test errors L2 and max
            error_proj = error_proj / test_norm
            error_proj_max = error_proj_max / test_max_norm

            error_max_mean = np.max(error_proj_max)
            error_mean = np.mean(error_proj)
            error_max = np.max(error_proj)
            error_min = np.min(error_proj)

            test_list.append(error_mean)

            # if loss_val < 0.0015:
                # optimizer.param_groups[0]['lr'] = LEARNING_RATE * 0.1

            if epoch > 500 and loss_val < 0.001:
                break

            # plt.ion()
            if epoch % args.iter == 0:
                print(
                    'Epoch [{}/{}], Time: {:.2f} s, Loss: {:.10f}\n Validation, Loss: {:.6f}, Test, Loss: {:.6f}, {:.6f}, {:.6f}, max {:.6f}\n'
                    .format(epoch, NUM_EPOCHS,
                            time.time() - start, mean_epoch_loss, loss_val,
                            error_mean, error_max, error_min, error_max_mean))

                start = time.time()

                # validation plot where validation loss is worst
                # index_list = torch.sort(index, descending=True)[1]
                plot_two(nor.frame2d(outputs_val),
                         nor.frame2d(outputs_val)-nor.rescale(val_np), index,
                         epoch,
                         title="validation")


                # save checkpoints
                if mean_epoch_loss < best:
                    is_best = True
                    best = mean_epoch_loss
                    it = 0
                    print("BEST CHANGED")
                else:
                    is_best=False
                    it += 1
                    if it > 8 and epoch> 5:
                        optimizer.param_groups[0]['lr'] *= 0.5
                        print("LR CHANGED")
                        it = 0

                checkpoint = {
                    'epoch': epoch,
                    'model': vae,
                    'optimizer': optimizer
                }
                save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
                is_best=False

                # # loss plot
                # loss_plot = torch.norm(outputs-snapshot.reshape(-1, 2*DOMAIN_SIZE**2), p=ORDER, dim=0).reshape(1, 2, DOMAIN_SIZE, DOMAIN_SIZE).detach().cpu().numpy()
                # loss_plot_ = (torch.max(torch.abs(outputs-snapshot.reshape(-1, 2*DOMAIN_SIZE**2)), dim=0)[0]).reshape(1, 2, DOMAIN_SIZE, DOMAIN_SIZE).detach().cpu().numpy()
                # plot_snapshot(loss_plot, 0)
                # plot_two(loss_plot, loss_plot_, [0], epoch, title="loss")

                # reconstruction plot
                # plt.show()
                # # plot_snapshot(max_sn*outputs.detach().cpu().numpy().reshape((-1, DIM, DOMAIN_SIZE, DOMAIN_SIZE)), 0)
                # plot_snapshot(max_sn*snapshot.detach().cpu().numpy().reshape((-1, DIM, DOMAIN_SIZE, DOMAIN_SIZE)), 0)
                # plt.show()

                # error plot
                # plt.plot(range(epoch//args.iter), np.log10(loss_list[::args.iter]))
                # plt.savefig('./train_cae.png')
                # plt.draw()
                # plt.pause(0.05)
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
            is_best=False

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)


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
