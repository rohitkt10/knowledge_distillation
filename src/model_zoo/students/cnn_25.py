import tensorflow as tf 
from tensorflow import keras as tfk
from tensorflow.keras import layers, regularizers


def get_model(
		input_shape, 
		activation='relu', 
		name="cnn_25"
		):
    """
    Set up the first convolutional layer outside this function. 
    """

    # input layer
    x = layers.Input(shape=input_shape, name='input')
    
    # 1st block
    y = layers.Conv1D(
    			filters=24, 
    			padding='same', 
    			kernel_size=19, 
    			use_bias=False,
    			kernel_regularizer=regularizers.l2(1e-6)
    				)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(activation)(y)
    y = layers.MaxPool1D(pool_size=25)(y)
    y = layers.Dropout(0.2)(y)
    
    # 2nd block
    y = layers.Conv1D(
    				filters=128, 
    				kernel_size=7, 
    				padding='same', 
    				kernel_regularizer=regularizers.l2(1e-6)
    				)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.MaxPool1D(pool_size=2)(y) 
    y = layers.Dropout(0.2)(y)
    
    # flatten and dense  layer
    y = layers.Flatten()(y)
    y = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-6))(y)      
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Dropout(0.5)(y)
    
    # output layer 
    y = layers.Dense(1, name='logits')(y) ## logits
    y = layers.Activation('sigmoid', name='output')(y) ## final output 
    
    model = tfk.Model(inputs=x, outputs=y, name=name)
    
    return model