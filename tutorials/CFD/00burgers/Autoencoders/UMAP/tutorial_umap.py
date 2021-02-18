from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.device('/GPU:0'):
    dims = (28, 28, 1)
    n_components = 2

    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=dims),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=n_components),
    ])
    encoder.summary()

    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_components)),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=7 * 7 * 256, activation="relu"),
        tf.keras.layers.Reshape(target_shape=(7, 7, 256)),
        tf.keras.layers.UpSampling2D((2)),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, padding="same", activation="relu"
        ),
        tf.keras.layers.UpSampling2D((2)),
        tf.keras.layers.Conv2D(
            filters=1, kernel_size=3, padding="same", activation="relu"
        ),

    ])

    decoder.summary()

    (train_images, Y_train), (test_images, Y_test) = mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], -1))/255.
    test_images = test_images.reshape((test_images.shape[0], -1))/255.
    validation_images = test_images.reshape((test_images.shape[0], -1))/255.
    print("VAL ", validation_images.shape, train_images.shape, )

    keras_fit_kwargs = {"callbacks": [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=10**-2,
            patience=10,
            verbose=1,
        )
    ]}

    embedder = ParametricUMAP(
        encoder=encoder,
        decoder=decoder,
        # batch_size=,
        dims=dims,
        parametric_reconstruction= True,
        reconstruction_validation=validation_images,
        verbose=True,
        autoencoder_loss = False,
        loss_report_frequency=1,
        keras_fit_kwargs = keras_fit_kwargs,
        n_training_epochs=1
    )
    print("LOSS", dir(embedder.parametric_model))
    embedding = embedder.fit_transform(train_images)
    embedder.save('/your/path/here')

    print(embedder._history)
    fig, ax = plt.subplots()
    ax.plot(embedder._history['loss'])
    ax.set_ylabel('Cross Entropy')
    ax.set_xlabel('Epoch')

    latent = embedding(train_images)
    print(latent.shape)

    plt.scatter(latent[:, 0], latent[:, 1], c=Y_train)
    plt.show()