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


def get_callbacks(monitor, ckptdir=None, save_best_only=True, save_freq="epoch",
					early_stopping=True, reduce_lr_on_plateau=True, log_to_csv=True):
	"""
	Set up the callbacks.
	"""
	callbacks = []
	if early_stopping:
		es_callback = tfk.callbacks.EarlyStopping(monitor, patience=20, verbose=1, mode='max',)
		callbacks.append(es_callback)
	if reduce_lr_on_plateau:
		lr_callback = tfk.callbacks.ReduceLROnPlateau(monitor, patience=20, verbose=1, mode='max')
		callbacks.append(lr_callback)
	if log_to_csv:
		csvlogger = tfk.callbacks.CSVLogger(os.path.join(ckptdir, "log.csv"))
		callbacks.append(csvlogger)
	if ckptdir:
		ckptdir = os.path.abspath(ckptdir)
		if not os.path.exists(ckptdir):
			os.makedirs(ckptdir)
		ckpt_callback = tfk.callbacks.ModelCheckpoint(
													filepath=os.path.join(ckptdir, "ckpt_epoch-{epoch:04d}"),
													monitor=monitor,
													verbose=1,
													mode='max',
													save_best_only=save_best_only,
													save_weights_only=False,
													save_freq=save_freq,
														)
		callbacks.append(ckpt_callback)
	return callbacks
