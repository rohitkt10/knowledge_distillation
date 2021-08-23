import tensorflow as tf

__all__ = ['get_model']

def get_model(
    input_shape=(1000, 4),
    num_classes=919,
    activation=tf.nn.relu,
    first_activation=None,
    kernel_initializer="glorot_normal",
    logits_only=False,
    name="deepsea",
):
    """Return `tf.keras` implementation of DeepSEA.
    The model in this file is a translation from the original Lua/Torch code.
    """
    padding = "valid"
    layers = [
        # Layer 1.
        tf.keras.layers.Conv1D(
            filters=320,
            kernel_size=8,
            kernel_initializer=kernel_initializer,
            padding=padding,
            name="layer1/convolution",
            input_shape=input_shape,
        ),
        tf.keras.layers.Activation(
            first_activation or activation, name="layer1/activation"
        ),
        tf.keras.layers.MaxPool1D(pool_size=4, strides=4, name="layer1/maxpool"),
        tf.keras.layers.Dropout(0.2, name="layer1/dropout"),
        # Layer 2.
        tf.keras.layers.Conv1D(
            filters=480,
            kernel_size=8,
            kernel_initializer=kernel_initializer,
            padding=padding,
            name="layer2/convolution",
        ),
        tf.keras.layers.Activation(activation, name="layer2/activation"),
        tf.keras.layers.MaxPool1D(pool_size=4, strides=4, name="layer2/maxpool"),
        tf.keras.layers.Dropout(0.2, name="layer2/dropout"),
        # Layer 3.
        tf.keras.layers.Conv1D(
            filters=960,
            kernel_size=8,
            kernel_initializer=kernel_initializer,
            padding=padding,
            name="layer3/convolution",
        ),
        tf.keras.layers.Activation(activation, name="layer3/activation"),
        tf.keras.layers.Dropout(0.5, name="layer3/dropout"),
        tf.keras.layers.Flatten(),
        # Layer 4.
        tf.keras.layers.Dense(
            num_classes, kernel_initializer=kernel_initializer, name="layer4/dense"
        ),
        tf.keras.layers.Activation(activation, name="layer4/activation"),
        # Layer 5.
        tf.keras.layers.Dense(
            num_classes, kernel_initializer=kernel_initializer, name="layer5/dense"
        ),
        tf.keras.layers.Activation(tf.keras.activations.linear, name="logits"),
            ]
    if not logits_only:
        layers = layers + [
                tf.keras.layers.Activation(tf.keras.activations.sigmoid, name="probabilities")
                            ]
    model = tf.keras.Sequential(layers=layers, name=name)
    return model
