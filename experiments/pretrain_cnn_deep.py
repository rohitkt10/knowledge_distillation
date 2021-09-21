"""
Pretrain CNN deep teacher model with truncated deepsea data.
"""
import numpy as np, os, sys, h5py, pandas as pd, argparse
sys.path.append("..")
import src
from src.utils.dataloaders import get_truncated_deepsea_dataset as dataloader
from src.model_zoo.teachers import cnn_deep
from src.utils.callbacks import ModelEvaluationCallback
from src.augmentations import RCAugmentation, AugmentedModel

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import metrics as tfkm

def main():
    # get keyboard arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=64, help="Batch size", type=int)
    parser.add_argument("--epochs", default=100, help="Epochs", type=int)
    parser.add_argument("--steps_per_epoch", default=None, help="Number of steps per epoch", type=int)
    parser.add_argument("--evaluate_steps", default=None, help="Number of evaluation steps", type=int)
    parser.add_argument("--save_freq", default=1, help="Frequency (in epochs) with which to save checkpoints", type=int)
    parser.add_argument("--rc", action="store_true", help="Add reverse compliment data augmentation")
    parser.add_argument("--no-rc", action="store_true", help="Do not add reverse compliment data augmentation")
    args = parser.parse_args()
    print(args.steps_per_epoch)

    # load data
    datadir = "../data/truncated_deepsea/"
    dataset = dataloader(datadir)

    # define model
    model = cnn_deep.get_model(num_classes=12, l2=1e-6)
    if args.rc:
        augmentations = [RCAugmentation()]
        model = AugmentedModel(model, augmentations)

    # compile model
    optimizer = tfk.optimizers.Adam(1e-2)
    loss = tfk.losses.BinaryCrossentropy()
    metrics = [tfkm.BinaryAccuracy(name='acc'), \
                tfkm.AUC(curve='ROC',name='auroc'), \
                tfkm.AUC(curve='PR',name='aupr')]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # set up save directory
    savedir = os.path.abspath("../pretrained_models/cnn_deep")
    if args.rc:
        savedir = os.path.join(savedir, "rc")
    else:
        savedir = os.path.join(savedir, "no_rc")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    trials = [f for f in os.listdir(savedir) if 'trial' in f]
    trial_number = len(trials) + 1
    savedir = os.path.join(savedir, f"trial_{trial_number:04d}")

    # set the callbacks
    lr_callback = tfk.callbacks.ReduceLROnPlateau(mode='max', monitor='val_aupr', \
                                            factor=0.2, patience=3, min_lr=1e-7)
    ckpt_callback = tfk.callbacks.ModelCheckpoint(
                        os.path.join(savedir, "ckpt_epoch-{epoch:04d}"),
                        monitor='val_aupr', mode='max', save_best_only=False, save_freq=args.save_freq, \
                        save_weights_only=False
                                    )
    eval_callback = ModelEvaluationCallback(
                            dataset['test'].batch(128),
                            filepath=os.path.join(savedir, "model_evaluation.csv"),
                            steps=args.evaluate_steps)
    csvlogger = tfk.callbacks.CSVLogger(os.path.join(savedir,"model_history.csv"))
    callbacks = [lr_callback, ckpt_callback, eval_callback, csvlogger]

    # fit model
    batchsize=args.batch
    steps_per_epoch = args.steps_per_epoch
    epochs = args.epochs
    model.fit(
        dataset['train'].repeat().shuffle(10000).batch(batchsize),
        epochs=epochs,
        validation_data=dataset['valid'].batch(batchsize),
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
            )

if __name__ == '__main__':
    main()
