"""
A script that puts down standard model compilation / training 
choices. 
"""

import tensorflow as tf
from tensorflow import keras as tfk
import os

__all__ = ['get_callbacks', 'get_compile_options']


def get_compile_options(baselr, output_type='binary',):
	"""
	Set up the model compile options. 
	"""
	output_type = output_type.lower().strip()
	assert output_type in ['binary', 'categorical', 'sparse_categorical']

	# set up the optimizer
	optimizer = tfk.optimizers.Adam(baselr)

	# set up the loss function
	if output_type == 'binary':
		loss = tfk.losses.BinaryCrossentropy(name='bce')
	if output_type == 'categorical':
		loss = tfk.losses.CategoricalCrossentropy(name='cce')
	if output_type == 'sparse_categorical':
		loss = tfk.losses.SparseCategoricalCrossentropy(name='cce')

	# set up the metrics
	if output_type == 'binary':
		acc = tfk.metrics.BinaryAccuracy(name='acc')
	else:
		acc = tfk.metrics.CategoricalAccuracy(name='acc')

	auroc = tfk.metrics.AUC(curve='ROC', name='auroc')
	aupr  = tfk.metrics.AUC(curve='PR', name='aupr')
	metrics = [acc, auroc, aupr]

	compile_options = {
				'loss':loss,
				'optimizer':optimizer,
				'metrics':metrics
				}
	return compile_options


def get_callbacks(monitor, ckptdir=None):
	"""
	Set up the callbacks. 
	"""
	es_callback = tfk.callbacks.EarlyStopping(monitor, patience=20, verbose=1, mode='max',)
	lr_callback = tfk.callbacks.ReduceLROnPlateau(monitor, patience=20, verbose=1, mode='max')
	csvlogger = tfk.callbacks.CSVLogger(os.path.join(ckptdir, "log.csv"))
	callbacks = [es_callback, lr_callback, csvlogger]

	if ckptdir:
		ckptdir = os.path.abspath(ckptdir)
		if not os.path.exists(ckptdir):
			os.makedirs(ckptdir)
		ckpt_callback = tfk.callbacks.ModelCheckpoint(
													filepath=os.path.join(ckptdir, "ckpt_epoch-{epoch:04d}"),
													monitor=monitor,
													verbose=1,
													mode='max',
													save_best_only=True,
													save_weights_only=False
														)
		callbacks.append(ckpt_callback)
	return callbacks