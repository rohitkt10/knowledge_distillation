import tensorflow as tf
from tensorflow import keras as tfk

__all__ = ['get_model']

def get_model(
        input_shape=(1000, 4),
        num_labels=919,
        activation='relu',
        logits_only=False,
        name="deepbind",
            ):
    # input layer
    x = tfk.layers.Input(shape=input_shape, name='input')

    # 1st block
    y = tfk.layers.Conv1D(
                        filters=64,
                        padding='same',
                        kernel_size=24,
                        use_bias=False,
                        kernel_regularizer=tfk.regularizers.l2(1e-6)
                        )(x)
    y = tfk.layers.Activation('relu')(y)  # rectification
    y = tfk.layers.Lambda(lambda x : tf.reduce_max(x, axis=1))(y)  # max pooling
    y = tfk.layers.Dropout(0.3)(y)  # dropout
    y = tfk.layers.Dense(32, activation="relu")(y)
    y = tfk.layers.Dense(num_labels, name="logits")(y)
    if not logits_only:
        y = tfk.layers.Activation("sigmoid")(y)
    model = tfk.Model(inputs=x, outputs=y, name=name)
    return model
