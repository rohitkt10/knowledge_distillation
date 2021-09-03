import os, sys
SRCPATH = os.path.abspath('../')
MODELZOOPATH = os.path.abspath('../src/')
RESULTSDIR = os.path.abspath("./results/basic_distillation")
DATAROOTDIR = os.path.abspath("../data/")
PRETRAINEDDIR = os.path.abspath("../pretrained_models")
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
from src.distillation_strategies import BasicDistiller, L2Distiller, L1Distiller
from src.model_zoo.students import truncated_deepsea
from src.utils import load_pretrained_teachers

from pdb import set_trace as keyboard
from importlib import import_module, __import__
from datetime import datetime

DATADIRS = {
	'deepsea':os.path.join(DATAROOTDIR, 'deepsea')
}
DATALOADERS = {
	'deepsea':utils.get_deepsea_dataset,
}
DISTILLER ={'kld':BasicDistiller, 'l2':L2Distiller, 'l1':L1Distiller}

def _get_model_and_compile(
					distiller_class,
					get_student_fn,
					get_student_fn_kwargs,
					get_teacher_fn,
					get_teacher_fn_kwargs,
					augmentations,
					compile_options,):
	student = get_student_fn(**get_student_fn_kwargs)
	if len(augmentations):
		student = AugmentedModel(student, augmentations)
	teacher = get_teacher_fn(**get_teacher_fn_kwargs)
	distiller = distiller_class(student=student, teacher=teacher)
	distiller.compile(**compile_options)
	return distiller

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default=0.5, help="Deepsea model truncation factor", type=float)
    parser.add_argument("--alpha", default=1., help="Coefficient to apply to distillation loss", type=float)
    parser.add_argument("--temperature", default=1., help="Temperature for generating soft labels", type=float)
    parser.add_argument("--ckpt", default=RESULTSDIR, help="Results directory", type=str)
    parser.add_argument("--batch", default=64, help="Batch size", type=int)
    parser.add_argument("--epochs", default=100, help="Epochs", type=int)
    parser.add_argument("--steps_per_epoch", default=None, help="Number of steps per epoch", type=int)
    parser.add_argument("--evaluate_steps", default=None, help="Number of steps during model evaluation", type=int)
    parser.add_argument("--distill_loss", default="kld", help="Distillation loss for soft targets", type=str)
    args = parser.parse_args()
    assert args.distill_loss.strip().lower() in ['kld', 'l1', 'l2']

    # set up the save directory (checkpoint location)
    ckptdir = os.path.join(
						args.ckpt,
                        "truncated_deepsea_distillation",
                        f"factor={args.factor}",
						f"temp={args.temperature}",
						f"alpha={args.alpha}",
                            )
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    # set up callbacks
    callbacks = [
                    tfk.callbacks.ReduceLROnPlateau(monitor="val_aupr", patience=10, verbose=1, mode='max',),
                    tfk.callbacks.EarlyStopping(monitor="val_aupr", patience=10, verbose=1, mode='max',),
                    tfk.callbacks.CSVLogger(os.path.join(ckptdir, "log.csv")),
                    tfk.callbacks.ModelCheckpoint(
											filepath=os.path.join(ckptdir, "ckpt_epoch-{epoch:04d}"),
											monitor="val_aupr",
											verbose=1,
											mode='max',
											save_best_only=True,
											save_weights_only=False,
											save_freq="epoch",
												)
                ]

    # get the dataset
    datadir = os.path.join(DATAROOTDIR, 'deepsea')
    dataloader = utils.get_deepsea_dataset
    dataset = dataloader(datadir=datadir)

    # set up the distiller
    augmentations = [RCAugmentation(), MixupAugmentation(alpha=0.2), GaussianNoiseAugmentation(stddev=0.1)]
    assert args.factor < 1. and args.factor > 0., "The truncation factor must be between 0 and 1."
    student = truncated_deepsea.get_model(truncation_factor=args.factor, logits_only=True)
    student = AugmentedModel(student, augmentations, subsample=True)
    teacher = load_pretrained_teachers.deepsea(ckptdir=os.path.join(PRETRAINEDDIR, 'deepsea'), logits_only=True, )
    distiller_class = DISTILLER[args.distill_loss]
    distiller = distiller_class(student, teacher)

    # compile the distiller
    metrics=[tfk.metrics.BinaryAccuracy('acc'), tfk.metrics.AUC(curve="ROC", name="auroc"), tfk.metrics.AUC(curve="PR", name='aupr')]
    distiller.compile(
            loss=tfk.losses.BinaryCrossentropy(),
            optimizer=tfk.optimizers.Adam(1e-3),
            alpha=args.alpha,
            temperature=args.temperature,
            metrics=metrics
                    )

    # fit the dataset
    distiller.fit(
            dataset['train'].shuffle(10000).batch(args.batch),
            validation_data=dataset['valid'].shuffle(10000).batch(args.batch),
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            callbacks=callbacks
                )

    # evaluate the performance of the distilled model
    ckpts=[f for f in os.listdir(ckptdir) if 'ckpt' in f]
    ckpts.sort()
    model=tfk.models.load_model(os.path.join(ckptdir, ckpts[-1]))
    model = tfk.Sequential([model, tfk.layers.Activation('sigmoid')])
    model.compile(metrics=metrics)
    res = model.evaluate(dataset['test'].batch(args.batch), return_dict=True, verbose=1, steps=args.evaluate_steps)
    res = pd.DataFrame(columns=list(res.keys()),data=[list(res.values())])
    res.to_csv(os.path.join(ckptdir, "test_results.csv"))
    return

if __name__=='__main__':
    main()
