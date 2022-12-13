

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import numpy as np
    import tensorflow as tf

    import context
    from sknotlearn.datasets import load_MNIST, load_CIFAR10
    from tensorno.tuner import tune_LWTA_architecture

    # x_train, y_train, x_test, y_test = load_MNIST()
    x_train, y_train, x_test, y_test = load_CIFAR10()

    # Flattening image arrays.
    x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=0.75, random_state=42
    )

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                     patience=5,
                                                     verbose=1,
                                                     restore_best_weights=True)

    tuner = tune_LWTA_architecture(
        layer_type="channel_out",
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        isregression=False,
        layer_choices=[2, 3, 4, 5],
        node_choices=[8, 16, 32, 64],
        group_choices=[4, 8, 16, 32],
        builder_kwargs=dict(
            dropout_rate=0.1,
            # lmbda=1e-4
        ),
        tuner_kwargs=dict(
            max_epochs=30,
            hyperband_iterations=1,
            project_name="CIFAR10_do_es",
            # overwrite=True
        ),
        search_kwargs=dict(
            epochs=30,
            callbacks=[early_stopper, ]
        )
    )

    tuner.results_summary(num_trials=3)
