import tensorflow as tf 
from tensorflow import keras as tfk
from tensorflow.keras import layers, regularizers

def get_model(
		input_shape=(200, 4), 
		activation='relu',
		name='cnn_dist'):
	
	# input layer
	x = layers.Input(shape=input_shape)
	
	# block 1 
	y = layers.Conv1D(
				filters=24,
				padding='same',
				kernel_size=19,
				use_bias=False,
				kernel_regularizer=regularizers.l2(1e-6),
					)(x)
	y = layers.BatchNormalization()(y)
	y = layers.Activation(activation=activation)(y)
	y = layers.Dropout(0.1)(y)

	# block 2 
	y = layers.Conv1D(
				filters=32,
				padding='same',
				kernel_size=7,
				use_bias=False,
				kernel_regularizer=regularizers.l2(1e-6),
					)(y)
	y = layers.BatchNormalization()(y)
	y = layers.Activation(activation='relu')(y)
	y = layers.Dropout(0.2)(y)

	# block 3
	y = layers.Conv1D(
				filters=48,
				padding='valid',
				kernel_size=7,
				use_bias=False,
				kernel_regularizer=regularizers.l2(1e-6),
					)(y)
	y = layers.BatchNormalization()(y)
	y = layers.Activation(activation='relu')(y)
	y = layers.Dropout(0.3)(y)

	# block 4
	y = layers.Conv1D(
				filters=64,
				padding='valid',
				kernel_size=3,
				use_bias=False,
				kernel_regularizer=regularizers.l2(1e-6),
					)(y)
	y = layers.BatchNormalization()(y)
	y = layers.Activation(activation='relu')(y)
	y = layers.Dropout(0.4)(y)

	# max pool 
	y = layers.MaxPool1D(3)(y)

	# flatten and get logits 
	y = layers.Flatten()(y)
	y = layers.Dense(1, name='logits')(y)

	# get output 
	y = layers.Activation('sigmoid', name='output')(y)

	# put together full model 
	model = tfk.Model(inputs=x, outputs=y, name=name)
	return model
