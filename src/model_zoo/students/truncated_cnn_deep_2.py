import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

def get_model(
        input_shape=(1000, 4),
        num_classes=919,
        logits_only=False,
        truncation_factor=0.5,
        l2=None,
        name="cnn_deep"
            ):
    if l2:
        l2 = tfk.regularizers.l2(l2)
    x = tfkl.Input(input_shape, name='input')
    y = tfkl.Conv1D(int(76*truncation_factor), 19, padding='same', kernel_regularizer=l2)(x)
    y = tfkl.BatchNormalization()(y)
    y = tfkl.Activation('relu')(y)
    y = tfkl.MaxPool1D(5)(y)
    y = tfkl.Dropout(truncation_factor*0.2)(y)

    y = tfkl.Conv1D(int(153*truncation_factor), 9, padding='same', kernel_regularizer=l2)(y)
    y = tfkl.BatchNormalization()(y)
    y = tfkl.Activation('relu')(y)
    y = tfkl.MaxPool1D(5)(y)
    y = tfkl.Dropout(truncation_factor*0.2)(y)

    y = tfkl.Conv1D(int(307*truncation_factor), 7, padding='same', kernel_regularizer=l2)(y)
    y = tfkl.BatchNormalization()(y)
    y = tfkl.Activation('relu')(y)
    y = tfkl.MaxPool1D(5)(y)
    y = tfkl.Dropout(truncation_factor*0.2)(y)

    y = tfkl.Flatten()(y)
    y = tfkl.Dense(int(614*truncation_factor), kernel_regularizer=l2)(y)
    y = tfkl.Dropout(truncation_factor*0.5)(y)
    y = tfkl.BatchNormalization()(y)
    y = tfkl.Activation('relu')(y)
    y = tfkl.Dense(num_classes, name="logits")(y)

    if not logits_only:
        y = tfkl.Activation('sigmoid')(y)

    model = tfk.Model(inputs=x, outputs=y, name=name)
    return model
