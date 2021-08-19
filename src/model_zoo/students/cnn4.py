import tensorflow as tf 
from tensorflow import keras as tfk

__all__ = ['get_model']

def get_model(
        input_shape=(1000, 4),
        num_labels=919,
        activation='relu', 
        name="cnn_25"
        ):
    """
    Conv1d -> Maxpool4 -> Conv1D -> Maxpool4 -> Dense128 -> Output.  
    """

    # input layer
    x = tfk.layers.Input(shape=input_shape, name='input')
    
    # 1st block
    y = tfk.layers.Conv1D(
                filters=32, 
                padding='same', 
                kernel_size=19, 
                use_bias=False,
                kernel_regularizer=tfk.regularizers.l2(1e-6)
                    )(x)
    y = tfk.layers.BatchNormalization()(y)
    y = tfk.layers.Activation(activation)(y)
    y = tfk.layers.MaxPool1D(pool_size=4)(y)
    y = tfk.layers.Dropout(0.2)(y)
    
    # 2nd block
    y = tfk.layers.Conv1D(
                    filters=64, 
                    kernel_size=7, 
                    padding='same', 
                    kernel_regularizer=tfk.regularizers.l2(1e-6)
                    )(y)
    y = tfk.layers.BatchNormalization()(y)
    y = tfk.layers.Activation('relu')(y)
    y = tfk.layers.MaxPool1D(pool_size=4)(y) 
    y = tfk.layers.Dropout(0.2)(y)
    
    # flatten and dense layer
    y = tfk.layers.Flatten()(y)
    y = tfk.layers.Dense(128, kernel_regularizer=tfk.regularizers.l2(1e-6))(y)      
    y = tfk.layers.BatchNormalization()(y)
    y = tfk.layers.Activation('relu')(y)
    y = tfk.layers.Dropout(0.5)(y)
    
    # output layer 
    y = tfk.layers.Dense(num_labels, kernel_regularizer=tfk.regularizers.l2(1e-6), name='logits')(y) ## logits
    y = tfk.layers.Activation('sigmoid', name='probabilities')(y) ## final output 
    
    # final model
    model = tfk.Model(inputs=x, outputs=y, name=name)
    return model