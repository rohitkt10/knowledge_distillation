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

def testing(dataset_name, student_name, test_dataset, options={}):
	# get the latest trials directory
	ckptdir=os.path.join(RESULTSDIR, "student_scratch", dataset_name, student_name)
	trials = [f for f in os.listdir(ckptdir) if 'trial' in f]
	trials.sort() 

	# get the latest checkpoint directory 
	ckptdir=os.path.join(ckptdir, trials[-1])
	ckpts=[f for f in os.listdir(ckptdir) if 'ckpt' in f]
	ckpts.sort()

	# load the model and evaluate 
	model=tfk.models.load_model(os.path.join(ckptdir, ckpts[-1]))
	res = model.evaluate(test_dataset, return_dict=True, **options)
	res = pd.DataFrame(columns=list(res.keys()),data=[list(res.values())])
	res.to_csv(os.path.join(ckptdir, "test_results.csv"))


def training(
			dataset,
			get_model_fn, 
			get_model_fn_kwargs={},
			augmentations=[],
			ckptdir=None,
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
	print("Fitting model...")
	model.fit(
		dataset['train'].shuffle(10000).batch(batch_size), 
		callbacks = get_callbacks(monitor=monitor, ckptdir=ckptdir),
		validation_data = dataset['valid'].shuffle(10000).batch(batch_size),
		**fit_options,
			)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset",default="deepsea", help="dataset", type=str)
	parser.add_argument("--student", default="cnn4", help="Student Model", type=str)
	parser.add_argument("--ckpt", default=RESULTSDIR, help="Results directory", type=str)
	parser.add_argument("--batch", default=64, help="Batch size", type=int)
	parser.add_argument("--epochs", default=100, help="Epochs", type=int)
	parser.add_argument("--steps_per_epoch", default=None, help="Number of steps per epoch", type=int)
	parser.add_argument("--evaluate_steps", default=None, help="Number of evaluation steps", type=int)
	parser.add_argument("--rc", action="store_true", help="Add reverse compliment data augmentation")
	parser.add_argument("--no-rc", action="store_true", help="Do not add reverse compliment data augmentation")
	parser.add_argument("--mixup", default=0., type=float, help="Concentration parameter for beta distribution in mixup.")
	parser.add_argument("--gaussian_noise", default=0., type=float, help="standard deviation of input Gaussian noise")
	args = parser.parse_args()

	# get the dataset
	datadir = DATADIRS[args.dataset]
	dataloader = DATALOADERS[args.dataset]
	dataset = dataloader(datadir=datadir)

	# get the student model file 
	student = getattr(import_module("model_zoo.students"), args.student)

	# get the augmentations if any
	augmentations = []
	if args.rc:
		augmentations.append(RCAugmentation())
	if args.mixup:
		assert args.mixup > 0., "Mixup beta distribution concentration param must be positive"
		augmentations.append(MixupAugmentation(alpha=args.mixup))
	if args.gaussian_noise:
		assert args.gaussian_noise > 0, "Gaussian noise s.d. must be positive."
		augmentations.append(GaussianNoiseAugmentation(stddev=args.gaussian_noise))

	# set up the checkpoint directory
	ckptdir = os.path.join(args.ckpt, "student_scratch", args.dataset, args.student)
	if not os.path.exists(ckptdir):
		trial = 1
	else:
		num_trials = len([f for f in os.listdir(ckptdir) if 'trial' in f]) 
		trial = num_trials + 1
	ckptdir = os.path.join(ckptdir, f'trial-{trial:05d}')
	os.makedirs(ckptdir)

	# get model.fit options 
	batch_size = args.batch ## keeping this separate to stay compatible with tf.data.Dataset usage
	fit_options = {
	'epochs':args.epochs,
	'steps_per_epoch':args.steps_per_epoch,
	}

	# Train the model and save results
	training(
		dataset,
		get_model_fn = student.get_model,
		augmentations = augmentations, 
		ckptdir=ckptdir,
		fit_options=fit_options,
			)

	# test the model and save test results 
	res = testing(
			dataset_name=args.dataset, 
			student_name=args.student, 
			test_dataset=dataset['test'].shuffle(10000).batch(batch_size),
			options={"verbose":1, "steps":args.evaluate_steps}
				)

if __name__ == '__main__':
	main()