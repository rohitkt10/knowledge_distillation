"""
Basic distillation of the CNN deep architecture
pretrained with the truncated deepsea dataset.
"""

import os, sys
SRCPATH = os.path.abspath('..')
sys.path.append(SRCPATH)

import numpy as np, h5py
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras as tfk

from src import RCAugmentation, MixupAugmentation, GaussianNoiseAugmentation, AugmentedModel
from src.distillation_strategies import BasicDistiller as Distiller
from src.model_zoo.students import truncated_cnn_deep
from src.utils.callbacks import ModelEvaluationCallback
from src.utils.dataloaders import get_truncated_deepsea_dataset as dataloader

from pdb import set_trace as keyboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=64, help="Batch size", type=int)
    parser.add_argument("--epochs", default=100, help="Epochs", type=int)
    parser.add_argument("--steps_per_epoch", default=None, help="Number of steps per epoch", type=int)
    parser.add_argument("--factor", default=0.5, help="Truncation factor", type=float)
    parser.add_argument("--temperature", default=1., help="Temperature for generating soft labels", type=float)
    parser.add_argument("--alpha", default=1., help="Coefficient to apply to distillation loss", type=float)
    parser.add_argument("--mixup", default=0., type=float, help="Concentration parameter for beta distribution in mixup.")
    parser.add_argument("--gaussian_noise", default=0., type=float, help="standard deviation of input Gaussian noise")
    parser.add_argument("--rc", action="store_true", help="Add reverse compliment data augmentation")
    parser.add_argument("--no-rc", action="store_true", help="Do not add reverse compliment data augmentation")
    parser.add_argument("--exptname", default='expt', type=str, help="Name of the experiment directory.")
    args = parser.parse_args()

    # load the dataset
    datadir = "../data/truncated_deepsea"
    dataset = dataloader(datadir)

    # set up teacher model
    teacher = tfk.models.load_model(os.path.join("../pretrained_models/cnn_deep/rc/trial_0001/ckpt_epoch-0049"))
    teacher = tfk.Model(inputs=teacher.inputs, outputs=teacher.layers[-2].output, name=teacher.name)
    teacher.trainable = False

    # set up student model
    student = truncated_cnn_deep.get_model(
                                    truncation_factor=args.factor,
                                    l2=1e-6,
                                    logits_only=True,
                                    num_classes=12
                                        )
    augmentations = []
    if args.rc:
        augmentations.append(RCAugmentation())
    if args.mixup:
        augmentations.append(MixupAugmentation(alpha=args.mixup))
    if args.gaussian_noise:
        augmentations.append(GaussianNoiseAugmentation(stddev=args.gaussian_noise))
    if len(augmentations)>0:
        student = AugmentedModel(student, augmentations)

    # set up distiller
    distiller = Distiller(student, teacher)

    # compile model
    optimizer = tfk.optimizers.Adam(1e-2)
    loss = tfk.losses.BinaryCrossentropy()
    metrics = [
            tfk.metrics.BinaryAccuracy(name='acc'),
            tfk.metrics.AUC(curve='PR', name='aupr'),
            tfk.metrics.AUC(curve='ROC', name='auroc')
            ]
    distiller.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                temperature=args.temperature,
                alpha=args.alpha
                    )

    # set up callbacks
    RC = 'rc' if args.rc else 'no-rc'
    savedir = os.path.abspath(f"results/basic_distillation/cnn_deep_distillation/{args.exptname}/")
    trialnum = 1+len([f for f in os.listdir(savedir) if 'trial' in f])
    savedir = os.path.join(savedir, f"trial_{trialnum:04d}")
    hyperparams = f"factor={args.factor}/temperature={args.temperature}/alpha={args.alpha}"
    savedir = os.path.join(savedir, hyperparams)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    lr_callback = tfk.callbacks.ReduceLROnPlateau(monitor='val_aupr', \
                                        patience=4, mode='max', min_lr=1e-7, )
    es_callback = tfk.callbacks.EarlyStopping(monitor='val_aupr', patience=8, mode='max',verbose=1, )
    ckpt_callback = tfk.callbacks.ModelCheckpoint(os.path.join(savedir, "ckpt_epoch-{epoch:04d}"), monitor='val_aupr', mode='max', save_best_only=True, save_weights_only=False)
    eval_callback = ModelEvaluationCallback(
                            dataset['test'].batch(128),
                            filepath=os.path.join(savedir,"model_evaluation.csv")
                                            )
    csvlogger = tfk.callbacks.CSVLogger(os.path.join(savedir, "model_history.csv"),)
    callbacks = [lr_callback, es_callback, ckpt_callback, csvlogger, eval_callback]

    # fit the model
    distiller.fit(
        dataset['train'].repeat().shuffle(10000).batch(args.batch),
        validation_data=dataset['valid'].batch(128),
        epochs=args.epochs,
        callbacks=callbacks,
        steps_per_epoch=args.steps_per_epoch,
                )

if __name__ == '__main__':
    main()
