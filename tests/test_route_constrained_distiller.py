import numpy as np, os, sys, pandas as pd
sys.path.append("..")

import tensorflow as tf
from tensorflow import keras as tfk

from src.utils.dataloaders import get_truncated_deepsea_dataset as dataloader
datadir = "../data/truncated_deepsea/"
dataset = dataloader(datadir)

from src.distillation_strategies import BasicDistiller as Distiller
from src.utils.callbacks import RouteConstrainedCallback
from src.model_zoo.students import truncated_cnn_deep
from src.model_zoo.teachers import cnn_deep

student = truncated_cnn_deep.get_model(truncation_factor=0.3, l2=1e-6, logits_only=True, num_classes=12)
teacher = cnn_deep.get_model(l2=1e-6, logits_only=True)
distiller = Distiller(student, teacher)

optimizer = tfk.optimizers.Adam(1e-3)
loss = tfk.losses.BinaryCrossentropy()
metrics = [
    tfk.metrics.BinaryAccuracy(name='acc'),
    tfk.metrics.AUC(curve='PR', name='aupr'),
    tfk.metrics.AUC(curve='ROC', name='auroc')
]
distiller.compile(optimizer=optimizer, loss=loss, metrics=metrics)

ckpts = [f"ckpt_epoch-{epoch:04d}" for epoch in [1, 2, 3, 4, 7, 9, 17]]
ckpts = [os.path.abspath(f"../pretrained_models/cnn_deep/{ckpt}") for ckpt in ckpts]
rc_callback = RouteConstrainedCallback(ckpts=ckpts, patience=2)
lr_callback = tfk.callbacks.ReduceLROnPlateau(monitor='val_aupr', patience=3, mode='max', min_lr=1e-7, )
callbacks = [rc_callback, lr_callback]
distiller.fit(dataset['train'].batch(128), validation_data=dataset['valid'].batch(128), epochs=10, \
              callbacks=callbacks, steps_per_epoch=10, validation_steps=3)
