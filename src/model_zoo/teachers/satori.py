import tensorflow as tf
from tensorflow import keras as tfk

__all__ = ['get_model']

def get_model(
        input_shape=(1000, 4),
        num_classes=919,
        activation=tf.nn.relu,
        logits_only=False,
        name="satori",
        num_heads=8,
        key_dim=64,
            ):
    # input layer
    x = tfk.layers.Input(input_shape)

    # 1st conv layer
    y = tfk.layers.Conv1D(filters=1000, kernel_size=13, padding='same', kernel_regularizer=tfk.regularizers.l2(1e-6),)(x)
    y = tfk.layers.BatchNormalization()(y)
    y = tfk.layers.Activation(activation)(y)
    y = tfk.layers.MaxPool1D(5)(y)
    y = tfk.layers.Dropout(0.2)(y)

    # bidirectional LSTM
    lstm = tfk.layers.LSTM(units=200, return_sequences=True, return_state=True, kernel_regularizer=tfk.regularizers.l2(1e-6))
    lstm_back = tfk.layers.LSTM(units=200, return_sequences=True, return_state=True, go_backwards=True, kernel_regularizer=tfk.regularizers.l2(1e-6))
    bidlstm = tfk.layers.Bidirectional(layer=lstm, backward_layer=lstm_back, merge_mode='concat')
    y = bidlstm(y)[0]
    y = tfk.layers.MaxPool1D(2)(y)
    y = tfk.layers.Dropout(0.2)(y)


    # MHA layer
    mha = tfk.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    y = mha(y,y,y)

    # dense layer to produce latent features
    y = tfk.layers.Dense(512, kernel_regularizer=tfk.regularizers.l2(1e-6))(y)
    y = tfk.layers.Activation("relu")(y)
    y = tfk.layers.MaxPool1D(2)(y)
    y = tfk.layers.Dropout(0.2)(y)
    y = tf.reduce_mean(y, axis=1)

    # final projection to task labels
    y = tfk.layers.Dense(num_classes, name="logits", kernel_regularizer=tfk.regularizers.l2(1e-6))(y)
    if not logits_only:
        y = tfk.layers.Activation("sigmoid", name="probabilities")(y)

    # final model
    model = tfk.Model(x, y, name=name)
    return model
