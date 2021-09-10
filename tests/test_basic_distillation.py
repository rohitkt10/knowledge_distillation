import numpy as np, os, sys, pandas as pd
sys.path.append("..")

import tensorflow as tf
from tensorflow import keras as tfk

from src.distillation_strategies import BasicDistiller as Distiller
from src.utils.callbacks import RouteConstrainedCallback
from src.model_zoo.students import truncated_cnn_deep
from src.model_zoo.teachers import cnn_deep
from src.augmentations import RCAugmentation, MixupAugmentation, GaussianNoiseAugmentation, AugmentedModel
from src.utils.dataloaders import get_truncated_deepsea_dataset as dataloader
datadir = "../data/truncated_deepsea/"
dataset = dataloader(datadir)

ckpt = os.path.abspath("../pretrained_models/cnn_deep/ckpt_epoch-0017")
teacher = tfk.models.load_model(ckpt)
teacher = tfk.Model(
        inputs=teacher.inputs,
        outputs=teacher.layers[-2].output, name=teacher.name
                )

l2 = None
student = truncated_cnn_deep.get_model(truncation_factor=0.3, l2=l2, logits_only=True, num_classes=12)
#augmentations = [RCAugmentation(), MixupAugmentation(alpha=0.3), GaussianNoiseAugmentation(stddev=0.15)]
#student = AugmentedModel(student, augmentations)
teacher = tfk.Model(inputs=models[-1].inputs, outputs=models[-1].layers[-2].output, name=models[-1].name)
teacher.trainable = False
distiller = Distiller(student, teacher)

optimizer = tfk.optimizers.Adam(1e-2)
loss = tfk.losses.BinaryCrossentropy()
metrics = [
    tfk.metrics.BinaryAccuracy(name='acc'),
    tfk.metrics.AUC(curve='PR', name='aupr'),
    tfk.metrics.AUC(curve='ROC', name='auroc')
]
distiller.compile(optimizer=optimizer, loss=loss, metrics=metrics, temperature=0.1, alpha=0.1)

lr_callback = tfk.callbacks.ReduceLROnPlateau(monitor='val_aupr', patience=3, mode='max', min_lr=1e-7, )
callbacks = [lr_callback]
distiller.fit(dataset['train'].shuffle(10000).batch(128), \
              validation_data=dataset['valid'].shuffle(10000).batch(128), epochs=11, \
              callbacks=callbacks, steps_per_epoch=200, validation_steps=50)
