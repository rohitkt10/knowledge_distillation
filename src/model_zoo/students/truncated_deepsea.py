import tensorflow as tf
from tensorflow import keras as tfk
from ..teachers import deepsea

__all__ = ['get_model']

def get_model(
        input_shape=(1000, 4),
        num_classes=919,
        logits_only=True,
        truncation_factor=1.,
        first_activation=None,
        activation="relu",
        name="truncated_deepsea"
                ):
    truncation_factor = float(truncation_factor)
    if truncation_factor == 1:
        return deepsea.get_model(
                        input_shape=input_shape,
                        num_classes=num_classes,
                        logits_only=logits_only,
                        activation=activation,
                        first_activation=first_activation,
                        name=name)

    else:
        x = tfk.layers.Input(input_shape, name="input")

        # Layer 1.
        y = tfk.layers.Conv1D(
                        filters=int(truncation_factor*320),
                        kernel_size=8,
                        kernel_initializer="glorot_normal",
                        padding="valid",
                        input_shape=input_shape,
                        name="layer1/convolution"
                            )(x)

        y = tfk.layers.Activation(
            first_activation or activation, name="layer1/activation"
                                )(y)
        y = tfk.layers.MaxPool1D(pool_size=4, name="layer1/maxpool")(y)
        y = tfk.layers.Dropout(0.2, name="layer1/dropout")(y)

        # Layer 2.
        y = tfk.layers.Conv1D(
                        filters=int(truncation_factor*480),
                        kernel_size=8,
                        kernel_initializer="glorot_normal",
                        padding="valid",
                        name="layer2/convolution",
                            )(y)
        y = tfk.layers.Activation(activation, name="layer2/activation")(y)
        y = tfk.layers.MaxPool1D(pool_size=4, name="layer2/maxpool")(y)
        y = tfk.layers.Dropout(0.2, name="layer2/dropout")(y)

        # Layer 3.
        y = tfk.layers.Conv1D(
                            filters=int(truncation_factor*960),
                            kernel_size=8,
                            kernel_initializer="glorot_normal",
                            padding="valid",
                            name="layer3/convolution",
                                )(y)
        y = tfk.layers.Activation(activation, name="layer3/activation")(y)
        y = tfk.layers.Dropout(0.5, name="layer3/dropout")(y)
        y = tfk.layers.Flatten()(y)

        # Layer 4.
        y = tfk.layers.Dense(
            int(truncation_factor*num_classes), kernel_initializer="glorot_normal", name="layer4/dense"
                            )(y)
        y = tfk.layers.Activation(activation, name="layer4/activation")(y)

        # Layer 5.
        y = tfk.layers.Dense(
            num_classes, kernel_initializer="glorot_normal", name="logits"
        )(y)
        if not logits_only:
            y = tfk.layers.Activation("sigmoid", name="probabilities")(y)

        # final model
        model = tfk.Model(inputs=x, outputs=y, name=name)
        return model
