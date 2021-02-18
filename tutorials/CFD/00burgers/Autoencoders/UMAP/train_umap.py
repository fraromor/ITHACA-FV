from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
tf.keras.backend.clear_session()
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

def plot_snapshot(frame, idx, idx_coord=0, title=""):
    m = frame.shape[2]
    x, y = np.meshgrid(np.arange(m), np.arange(m))
    # print("FRAME: ", frame.shape)
    z = frame[idx, x, y, idx_coord]
    plt.figure(figsize=(7, 6))
    plt.title(title)
    pl = plt.contourf(x, y, z)
    cb = plt.colorbar(pl, fraction=0.046, pad=0.04)
    plt.show()

def plot_correlations(latent_data, latent_dim):
    fig, axes = plt.subplots(latent_dim, latent_dim, figsize=(10, 10))
    for i in range(latent_dim):
        for j in range(i, latent_dim):
            axes[i, j].scatter(latent_data[:, i], latent_data[:, j])
    plt.show()

def relativeMax(snap, rec):
    max_norm = np.max(np.abs(snap))
    abs_err = np.max(np.abs(snap-rec))
    print("MAX NORM: ", max_norm, "MAX ABS ERR: ", abs_err)
    return abs_err/max_norm

def relativeL2(snap, rec):
    l2_norm = np.linalg.norm(snap.reshape(-1, 2*60**2), axis=1)
    l2_abs_err = np.linalg.norm((snap-rec).reshape(-1, 2*60**2), axis=1)
    # print("L2 NORM: ", l2_norm, "L2 ABS ERR: ", l2_abs_err)
    return l2_abs_err/l2_norm

def train(snap_scaled, n_train, n_val):
    with tf.device('/CPU:0'):
        snap_shuffled = tf.random.shuffle(snap_scaled, seed=None, name=None)
        snap_train, snap_val = tf.split(snap_shuffled, [n_train, n_val], axis=0, num=None, name='split')
        print("train, val: ", snap_train.shape, snap_val.shape)

        dims = (60, 60, 2)
        n_components = 4

        encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=dims),
        tf.keras.layers.Conv2D(
            filters=8, kernel_size=3, strides=(2, 2), activation="elu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=(2, 2), activation="elu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation="elu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="elu", padding="same"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation="elu"),
        tf.keras.layers.Dense(units=256, activation="elu"),
        tf.keras.layers.Dense(units=n_components),
        ])
        encoder.summary()

        decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_components)),
        tf.keras.layers.Dense(units=256, activation="elu"),
        tf.keras.layers.Dense(units=4 * 4 * 256, activation="elu"),
        tf.keras.layers.Reshape(target_shape=(4, 4, 256)),
        tf.keras.layers.UpSampling2D((2)),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, padding="same", activation="elu"
        ),
        tf.keras.layers.UpSampling2D((2)),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, padding="same", activation="elu"
        ),
        tf.keras.layers.UpSampling2D((2)),
        tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, activation="elu"
        ),
        tf.keras.layers.UpSampling2D((2)),
        tf.keras.layers.Conv2D(
            filters=2, kernel_size=3, padding="same", activation="relu"
        ),

        ])

        decoder.summary()

        # train
        keras_fit_kwargs = {"callbacks": [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=10**-2,
            patience=10,
            verbose=1,
        )
        ]}

        # NB UMAP base class parameters are passed as keyvalued arguments, see umap_.py
        embedder = ParametricUMAP(
        encoder=encoder,
        decoder=decoder,
        batch_size=50,
        dims=dims,
        parametric_reconstruction= True,
        reconstruction_validation=snap_val,
        verbose=True,
        autoencoder_loss = True,
        loss_report_frequency=10,
        keras_fit_kwargs = keras_fit_kwargs,
        n_training_epochs=10,
        n_neighbors=30,
        n_components=n_components,
        )

        embedding = embedder.fit_transform(snap_train)
        embedder.save('./embedder')

        print(embedder._history)
        fig, ax = plt.subplots()
        ax.semilogy(embedder._history['loss'])
        ax.set_ylabel('log MSE')
        ax.set_xlabel('Epoch')

def predict(snap_train, mean, max_sn, min_sn):
    from umap.parametric_umap import load_ParametricUMAP
    embedder = load_ParametricUMAP('./embedder', verbose=True)

    embedder.parametric_model.summary()
    print(embedder.embedding_.shape)
    fig, ax = plt.subplots()
    ax.semilogy(embedder._history['loss'])
    ax.set_ylabel('log MSE')
    ax.set_xlabel('Epoch')

    plot_correlations(embedder.embedding_, 4)

    latent = embedder.encoder(snap_train.reshape(-1, 60, 60, 2))
    rec = embedder.decoder(latent).numpy()

    rec = rec * (max_sn-min_sn) + min_sn
    rec = rec + mean
    rec = np.clip(rec, a_min = 0, a_max=None)

    snap_train = snap_train.reshape(-1, 60, 60, 2) * (max_sn-min_sn) + min_sn
    snap_train = snap_train + mean

    print(latent.shape, rec.shape)
    # plot_snapshot(rec, 1)
    # plot_snapshot(snap_train, 1)
    # for i in range(4):
    #     plot_snapshot(snap_train-rec, i*500, title="snap {} - reconstructed snap {}".format(i*500, i*500))
    print("maximum rel error", relativeMax(snap_train.reshape(-1, 60, 60, 2), rec))
    err_l2 = relativeL2(snap_train.reshape(-1, 60, 60, 2), rec)
    plt.semilogy(range(1, err_l2.shape[0]+1), err_l2)
    plt.ylim([1e-4,1e-0])
    plt.grid(True, which="both")
    plt.xlabel("time instants [s]")
    plt.ylabel(" log10 relative L2 error")
    plt.title("Projection error of test sample for reduced Burgers' PDE\n reduced dimension is 4")
    plt.show()
    # print("L2 rel error", relativeL2(snap_train.reshape(-1, 60, 60, 2), rec))

if __name__ == '__main__':
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    WM_PROJECT = "../../"
    snap_vec = np.load(WM_PROJECT + "npSnapshots.npy")
    assert np.min(snap_vec) >= 0., "Snapshots should be clipped"
    print(snap_vec[:, ::5].shape)
    # np.save("npSnapshots.npy", snap_vec[:, ::5])

    # specify how many samples should be used for training and validation
    n_total = snap_vec.shape[1]
    n_train = n_total-n_total//10
    n_val = n_total-n_train
    print("Dimension of validation set: ", n_total-n_train)

    # scale the snapshots
    to_b_scaled = snap_vec.T.reshape(-1, 3, 60, 60)[:, :2, :, :]
    # plot_snapshot(to_b_scaled, 2500)

    mean = np.mean(to_b_scaled, axis=0, keepdims=True)#to_b_scaled[:1, :, :, :]
    # plot_snapshot(mean, 0, title="mean")
    print("max, min snap before scaling: ", np.max(to_b_scaled), np.min(to_b_scaled))
    snap = to_b_scaled - mean
    max_sn = np.max(snap)
    min_sn = np.min(snap)

    snap = snap - 0.5 * (min_sn + max_sn)
    snap_scaled = snap * 2 / (max_sn - min_sn)
    snap_scaled = snap_scaled.reshape(-1, 2, 60**2).transpose((0, 2, 1)).reshape(-1, 2*60**2)

    # snap_scaled = (snap-min_sn)/(max_sn-min_sn)
    # snap_scaled = snap_scaled.reshape(-1, 2, 60**2).transpose((0, 2, 1)).reshape(-1, 2*60**2)

    print("snapshots shape", snap_scaled.shape)
    print("Min max after scaling tf: ", np.min(snap_scaled), np.max(snap_scaled))
    # plot_snapshot(snap_scaled, 2500)

    # train(snap_scaled[:n_total, :], n_train, n_val)
    snap_true_vec = np.load(WM_PROJECT + "npTrueSnapshots.npy")
    assert np.min(snap_true_vec) >= 0., "Snapshots should be clipped"
    # np.save("npTrueSnapshots.npy", snap_true_vec[:, ::5])

    to_b_scaled_true = snap_true_vec.T.reshape(-1, 3, 60, 60)[:, :2, :, :]
    snap_true = to_b_scaled_true - mean

    snap_true = snap_true - 0.5 * (min_sn + max_sn)
    snap_true_scaled = snap_true * 2 / (max_sn - min_sn)
    snap_true_scaled = snap_true_scaled.reshape(-1, 2, 60**2).transpose((0, 2, 1)).reshape(-1, 2*60**2)

    # snap_true_scaled = (snap_true-min_sn)/(max_sn-min_sn)
    # snap_true_scaled = snap_true_scaled.reshape(-1, 2, 60**2).transpose((0, 2, 1))

    predict(snap_true_scaled, mean.reshape(-1, 2, 60**2).transpose((0, 2, 1)).reshape(-1, 60, 60, 2), max_sn , min_sn)

    # process_eval = multiprocessing.Process(target=train, args=(snap_scaled, n_train, n_val))
    # process_eval.start()
    # process_eval.join()