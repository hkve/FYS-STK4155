

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import numpy as np
    import tensorflow as tf

    import context
    from sknotlearn.datasets import load_MNIST, load_CIFAR10, load_EPL
    from tensorno.tuner import tune_LWTA_architecture

    # x_train, y_train, x_test, y_test = load_MNIST()
    # x_train, y_train, x_test, y_test = load_CIFAR10()
    dataset = load_EPL(encoded=True)
    y = dataset.y.to_numpy()
    labels = np.array(["w", "d", "l"])
    y = np.array([np.where(y_ == range(len(labels)), 1, 0) for y_ in y], dtype=int)
    x = dataset.x
    x = x.astype(float)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=5/6,
                                                        shuffle=False)

    # Flattening image arrays.
    # x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=0.75, random_state=42
    )

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                     patience=15,
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
            dropout_rate=0.25,
            lmbda=1e-4
        ),
        tuner_kwargs=dict(
            max_epochs=150,
            hyperband_iterations=3,
            project_name="EPL_do_l2_es",
            overwrite=True
        ),
        search_kwargs=dict(
            epochs=150,
            callbacks=[early_stopper, ]
        )
    )

    tuner.results_summary(num_trials=3)
