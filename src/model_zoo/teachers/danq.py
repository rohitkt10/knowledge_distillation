import tensorflow as tf

__all__ = ['get_model']

def get_model(
    input_shape=(1000, 4),
    num_classes=919,
    activation=tf.nn.relu,
    first_activation=None,
    kernel_initializer="glorot_normal",
    logits_only=False,
    name="danq",
):
    """Return `tf.keras` implementation of DanQ.
    This model is a translation of the Keras 0.2.0 code, available at
    https://github.com/uci-cbcl/DanQ/blob/98341e0e2eeb4b8ec12632b93fb09c563605451a/DanQ_train.py
    Notes
    -----
    - Paper: https://doi.org/10.1093/nar/gkw226
    - Code: https://github.com/uci-cbcl/DanQ
    From supplementary note of manuscript:
    > Detailed specifications of the architectures and hyperparameters of the DanQ
    > models used in this study. Numbers to the right of the forward slash indicate
    > values unique to the DanQ-JASPAR model, a larger model in which about half of the
    > convolution kernels are initialized with motifs from the JASPAR database.
    >
    > Model Architecture:
    > 1. Convolution layer (320/1024 kernels. Window size: 26/30. Step size: 1.)
    > 2. Pooling layer (Window size: 13/15. Step size: 13/15.)
    > 3. Bi-directional long short term memory layer (320/512 forward and 320/512
    >      backward
    >      LSTM neurons)
    > 4. Fully connected layer (925 neurons)
    > 5. Sigmoid output layer
    >
    > Regularization Parameters:
    >
    > Dropout proportion (proportion of outputs randomly set to 0):
    > - Layer 2: 20%
    > - Layer 3: 50%
    > - All other layers: 0%
    """
    layers = [
        tf.keras.layers.Conv1D(
                            filters=320,
                            kernel_size=26,
                            strides=1,
                            padding="valid",
                            kernel_initializer=kernel_initializer,
                            name="layer1/convolution",
                            input_shape=input_shape,
                            ),
        tf.keras.layers.Activation(
            first_activation or activation, name="layer1/activation"
                            ),
        tf.keras.layers.MaxPool1D(pool_size=13, strides=13, name="layer1/maxpool"),
        tf.keras.layers.Dropout(0.2, name="layer1/dropout"),
        tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units=320, return_sequences=True),
                                        backward_layer=tf.keras.layers.LSTM(
                                        units=320, return_sequences=True, go_backwards=True
                                        ),
                                    merge_mode="concat",
                                    name="bidirectional",
                                ),
        tf.keras.layers.Dropout(0.5, name="bidirectional/dropout"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(925, name="fully-connected"),
        tf.keras.layers.Activation(activation, name="fully-connected/activation"),
        tf.keras.layers.Dense(num_classes, name="logits"),
            ]
    if not logits_only:
        layers.append(tf.keras.layers.Activation('sigmoid'), name='probabilities')

    model = tf.keras.Sequential(layers=layers, name=name)
    return model
