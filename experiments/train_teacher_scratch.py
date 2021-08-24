import os, sys
SRCPATH = os.path.abspath('../')
MODELZOOPATH = os.path.abspath('../src/')
RESULTSDIR = os.path.abspath("./results/")
DATAROOTDIR = os.path.abspath("../data/")
sys.path.append(SRCPATH)
sys.path.append(MODELZOOPATH)

import numpy as np, h5py
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras as tfk
from src import RCAugmentation, MixupAugmentation, GaussianNoiseAugmentation, AugmentedModel
from src import utils
from src.utils import get_callbacks, get_compile_options
from pdb import set_trace as keyboard
from importlib import import_module, __import__
from datetime import datetime

DATADIRS = {
	'deepsea':os.path.join(DATAROOTDIR, 'deepsea')
}
DATALOADERS = {
	'deepsea':utils.get_deepsea_dataset,
}

def _get_model_and_compile(get_model_fn, get_model_fn_kwargs, augmentations, compile_options):
    model = get_model_fn(**get_model_fn_kwargs)
    if len(augmentations):
        model = AugmentedModel(model, augmentations)
    model.compile(**compile_options)
    return model

def testing(test_dataset, ckptdir, evaluate_options={}):
    # get the latest checkpoint directory
    ckpts=[f for f in os.listdir(ckptdir) if 'ckpt' in f]

    # get test metrics at all available checkpoints
    for ckpt in ckpts:
        suffix = ckpt.split("_")[-1]
        model=tfk.models.load_model(os.path.join(ckptdir, ckpt))
        res = model.evaluate(test_dataset, return_dict=True, **evaluate_options)
        res = pd.DataFrame(columns=list(res.keys()),data=[list(res.values())])
        res.to_csv(os.path.join(ckptdir, f"test_results_{suffix}.csv"))

def training(
        dataset,
        get_model_fn,
        get_model_fn_kwargs={},
        augmentations=[],
        ckptdir=None,
        save_freq=1,
        batch_size=64,
        monitor='val_aupr',
        fit_options={"epochs":1, "steps_per_epoch":None},
            ):
    # set up distributed training strategy
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    if num_gpus > 1:
        print("Multiple GPUs detected. Setting up mirrored strategy for training...")
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None

    # set up and compile the model
    if strategy:
        with strategy.scope():
            compile_options = get_compile_options(baselr=4e-3)
            model = _get_model_and_compile(get_model_fn, get_model_fn_kwargs, augmentations, compile_options)
    else:
        compile_options = get_compile_options(baselr=4e-3)
        model = _get_model_and_compile(get_model_fn, get_model_fn_kwargs, augmentations, compile_options)
    print("Model compiled...")

    # fit the model
    callbacks = get_callbacks(
                        monitor=monitor,
                        ckptdir=ckptdir,
                        save_best_only=False,
                        save_freq=save_freq,
                        early_stopping=False,
                            )

    print("Fitting model...")
    model.fit(
        dataset['train'].shuffle(10000).batch(batch_size),
        callbacks = callbacks,
        validation_data = dataset['valid'].shuffle(10000).batch(batch_size),
        **fit_options,
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="deepsea", help="dataset", type=str)
    parser.add_argument("--teacher", default="deepsea", help="Teacher Model", type=str)
    parser.add_argument("--ckpt", default=RESULTSDIR, help="Results directory", type=str)
    parser.add_argument("--batch", default=64, help="Batch size", type=int)
    parser.add_argument("--epochs", default=100, help="Epochs", type=int)
    parser.add_argument("--steps_per_epoch", default=None, help="Number of steps per epoch", type=int)
    parser.add_argument("--evaluate_steps", default=None, help="Number of evaluation steps", type=int)
    parser.add_argument("--save_freq", default=1, help="Epoch frequency with which to checkpoint model", type=int)
    args = parser.parse_args()

    # get the dataset
    datadir = DATADIRS[args.dataset]
    dataloader = DATALOADERS[args.dataset]
    dataset = dataloader(datadir=datadir)

    # get the teacher model file
    teacher = getattr(import_module("model_zoo.teachers"), args.teacher)

    # set up the checkpoint directory
    ckptdir = os.path.join(
                        args.ckpt,
                        "teacher_scratch",
                        args.dataset,
                        args.teacher,
                            )
    if not os.path.exists(ckptdir):
        trial = 1
    else:
        num_trials = len([f for f in os.listdir(ckptdir) if 'trial' in f])
        trial = num_trials + 1
    ckptdir = os.path.join(ckptdir, f'trial-{trial:05d}')
    os.makedirs(ckptdir)
    print("Results will be saved to :\n%s"%str(ckptdir))

    # get model.fit options
    batch_size = args.batch ## keeping this separate to stay compatible with tf.data.Dataset usage
    fit_options = {
            'epochs':args.epochs,
            'steps_per_epoch':args.steps_per_epoch,
            }

    # Train the model and save results
    training(
        dataset=dataset,
        get_model_fn = teacher.get_model,
        get_model_fn_kwargs={'logits_only':False},
        ckptdir=ckptdir,
        save_freq=args.save_freq,
        fit_options=fit_options,
            )

    # test the model and save test results
    res = testing(
            test_dataset=dataset['test'].shuffle(10000).batch(batch_size),
            ckptdir=ckptdir,
            evaluate_options={"verbose":1, "steps":args.evaluate_steps}
                )

if __name__ == '__main__':
    main()
