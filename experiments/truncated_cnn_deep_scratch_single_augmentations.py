"""
Train truncated CNN deep teacher model with truncated deepsea data with a single
type of augmentation or no augmentation.
"""
import numpy as np, os, sys, h5py, pandas as pd, argparse
from pdb import set_trace as keyboard
sys.path.append("..")
import src
from src.utils.dataloaders import get_truncated_deepsea_dataset as dataloader
from src.model_zoo.students import truncated_cnn_deep
from src.utils.callbacks import ModelEvaluationCallback
from src.augmentations import RCAugmentation, GaussianNoiseAugmentation, MixupAugmentation, AugmentedModel

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import metrics as tfkm

def main():
    # get keyboard arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=64, help="Batch size", type=int)
    parser.add_argument("--epochs", default=100, help="Epochs", type=int)
    parser.add_argument("--steps_per_epoch", default=None, help="Number of steps per epoch", type=int)
    parser.add_argument("--save_freq", default=1, help="Frequency (in epochs) with which to save checkpoints", type=int)
    parser.add_argument("--factor", default=0.5, help="Truncation factor", type=float)
    parser.add_argument("--rc", action="store_true", help="Add reverse compliment data augmentation")
    parser.add_argument("--no-rc", action="store_true", help="Do not add reverse compliment data augmentation")
    parser.add_argument("--gaussian_noise", default=0., type=float, help="standard deviation of input Gaussian noise")
    parser.add_argument("--mixup", default=0., type=float, help="Concentration parameter for beta distribution in mixup.")
    #parser.add_argument("--dir", default="no_augmentation", help="Name of the directory in which to save checkpoints and results")
    args = parser.parse_args()

    # load data
    datadir = "../data/truncated_deepsea/"
    dataset = dataloader(datadir)

    # define model
    model = truncated_cnn_deep.get_model(
                            input_shape=(1000, 4),
                            num_classes=12,
                            l2=1e-6,
                            truncation_factor=args.factor
                                        )

    augmentations = []
    dirname = "no_augmentation"
    RC = args.rc
    if RC:
        augmentations.append(RCAugmentation())
        dirname = "rc"
    if args.gaussian_noise:
        augmentations.append(GaussianNoiseAugmentation(stddev=args.gaussian_noise))
        dirname = f"gn={args.gaussian_noise}"
    if args.mixup:
        augmentations.append(MixupAugmentation(alpha=args.mixup))
        dirname = f"mixup={args.mixup}"
    assert len(augmentations) < 2, "Pass only 1 augmentation."
    if len(augmentations) > 0:
        model = AugmentedModel(model, augmentations)

    # compile model
    optimizer = tfk.optimizers.Adam(1e-2)
    loss = tfk.losses.BinaryCrossentropy()
    metrics = [tfkm.BinaryAccuracy(name='acc'), \
                tfkm.AUC(curve='ROC',name='auroc'), \
                tfkm.AUC(curve='PR',name='aupr')]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # set up callbacks
    savedir = os.path.abspath(f"results/truncated_cnn_deep/{dirname}/factor={args.factor}")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    lr_callback = tfk.callbacks.ReduceLROnPlateau(mode='max', monitor='val_aupr', \
                                            patience=4, min_lr=1e-7,)
    es_callback = tfk.callbacks.EarlyStopping(mode='max', monitor='val_aupr', \
                                            verbose=1, patience=8)
    ckpt_callback = tfk.callbacks.ModelCheckpoint(
                        os.path.join(savedir, "ckpt_epoch-{epoch:04d}"),\
                        monitor='val_aupr', mode='max', save_best_only=False, save_freq=args.save_freq, \
                        save_weights_only=False
                                    )
    eval_callback = ModelEvaluationCallback(
                            dataset['test'].batch(128),
                            filepath=os.path.join(savedir,"model_evaluation.csv")
                                            )
    csvlogger = tfk.callbacks.CSVLogger(os.path.join(savedir, "model_history.csv"),)
    callbacks = [lr_callback, ckpt_callback, eval_callback, csvlogger]

    # fit the model
    model.fit(
        dataset['train'].repeat().shuffle(10000).batch(args.batch),
        validation_data=dataset['valid'].batch(128),
        epochs=args.epochs,
        callbacks=callbacks,
        steps_per_epoch=args.steps_per_epoch,
                )

if __name__ == '__main__':
    main()
