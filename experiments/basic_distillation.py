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

def testing(
		test_dataset,
		ckptdir,
		logits_only=True,
		evaluate_options={}
			):
	# get the latest checkpoint directory
	ckpts=[f for f in os.listdir(ckptdir) if 'ckpt' in f]
	ckpts.sort()

	# load the model
	model=tfk.models.load_model(os.path.join(ckptdir, ckpts[-1]))
	if logits_only:
		metrics = model.compiled_metrics._metrics[0]
		model = tfk.Sequential([model, tfk.layers.Activation('sigmoid')])
		model.compile(metrics=metrics)

	# evaluate the model
	res = model.evaluate(test_dataset, return_dict=True, **evaluate_options)
	res = pd.DataFrame(columns=list(res.keys()),data=[list(res.values())])
	res.to_csv(os.path.join(ckptdir, "test_results.csv"))

def training(
			dataset,
			distiller_class,
			get_student_fn,
			get_student_fn_kwargs,
			get_teacher_fn,
			get_teacher_fn_kwargs,
			augmentations=[],
			ckptdir=None,
			batch_size=64,
			monitor='val_aupr',
			distiller_compile_options ={'alpha':1., 'temperature':1.},
			fit_options={"epochs":1, "steps_per_epoch":None},
			):
	if not 'val' in monitor:
		monitor = "val_"+monitor

	# set up distributed training strategy
	num_gpus = len(tf.config.list_physical_devices('GPU'))
	if num_gpus > 1:
		print("Multiple GPUs detected. Setting up mirrored strategy for training...")
		strategy = tf.distribute.MirroredStrategy()
	else:
		strategy = None

	# set up and compile the model
	compile_options = get_compile_options(baselr=4e-3)
	compile_options.update(distiller_compile_options)

	if strategy:
		with strategy.scope():
			distiller = _get_model_and_compile(
									distiller_class, get_student_fn,
									get_student_fn_kwargs, get_teacher_fn,
									get_teacher_fn_kwargs, augmentations,
									compile_options
												)
	else:
		distiller = _get_model_and_compile(
								distiller_class, get_student_fn,
								get_student_fn_kwargs, get_teacher_fn,
								get_teacher_fn_kwargs, augmentations,
								compile_options
											)
	print("Distiller compiled...")

	# distill the model
	callbacks = get_callbacks(
						monitor=monitor,
						ckptdir=ckptdir,
						save_best_only=True,
						save_freq='epoch',
						early_stopping=True,
							)
	print("Fitting model...")
	distiller.fit(
			dataset['train'].shuffle(10000).batch(batch_size),
			callbacks = callbacks,
			validation_data = dataset['valid'].shuffle(10000).batch(batch_size),
			**fit_options,
				)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset",default="deepsea", help="dataset", type=str)
	parser.add_argument("--teacher", default="deepsea", help="Teacher Model", type=str)
	parser.add_argument("--student", default="cnn4", help="Student Model", type=str)
	parser.add_argument("--distill_loss", default="kld", help="Distillation loss for soft targets", type=str)
	parser.add_argument("--temperature", default=1., help="Temperature for generating soft labels", type=float)
	parser.add_argument("--alpha", default=1., help="Coefficient to apply to distillation loss", type=float)
	parser.add_argument("--ckpt", default=RESULTSDIR, help="Results directory", type=str)
	parser.add_argument("--batch", default=64, help="Batch size", type=int)
	parser.add_argument("--epochs", default=100, help="Epochs", type=int)
	parser.add_argument("--steps_per_epoch", default=None, help="Number of steps per epoch", type=int)
	parser.add_argument("--evaluate_steps", default=None, help="Number of evaluation steps", type=int)
	parser.add_argument("--monitor", default='aupr', help="Metric to monitor for model checkpointing.", type=str)
	parser.add_argument("--rc", action="store_true", help="Add reverse compliment data augmentation")
	parser.add_argument("--no-rc", action="store_true", help="Do not add reverse compliment data augmentation")
	parser.add_argument("--mixup", default=0., type=float, help="Concentration parameter for beta distribution in mixup.")
	parser.add_argument("--gaussian_noise", default=0., type=float, help="standard deviation of input Gaussian noise")
	args = parser.parse_args()
	assert args.teacher in  ['danq', 'deepsea']
	assert args.student in ['cnn4', 'cnn25']
	assert args.distill_loss.strip().lower() in ['kld', 'l1', 'l2']
	distiller_class = DISTILLER[args.distill_loss]

	# get the dataset
	datadir = DATADIRS[args.dataset]
	dataloader = DATALOADERS[args.dataset]
	dataset = dataloader(datadir=datadir)

	# get the student and teacher model file
	student = getattr(import_module("model_zoo.students"), args.student)
	teacher = getattr(import_module("utils.load_pretrained_teachers"), args.teacher)

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
	rc = "rc" if args.rc else "no-rc"
	mx = "mixup=%s"%str(args.mixup)
	gn = "gn=%s"%str(args.gaussian_noise)
	ckptdir = os.path.join(
						args.ckpt,
						f"dataset={args.dataset}",
						f"distill_loss={args.distill_loss}",
						f"teacher={args.teacher}",
						f"student={args.student}",
						f"temp={args.temperature}",
						f"alpha={args.alpha}",
						rc,
						mx,
						gn)
	if not os.path.exists(ckptdir):
		trial = 1
	else:
		num_trials = len([f for f in os.listdir(ckptdir) if 'trial' in f])
		trial = num_trials + 1
	ckptdir = os.path.join(ckptdir, f'trial-{trial:05d}')
	os.makedirs(ckptdir)
	print("Results will be saved to :\n%s"%str(ckptdir))

	# Train the model and save results
	training(
		dataset,
		distiller_class,
		get_student_fn = student.get_model,
		get_student_fn_kwargs = {'logits_only':True},
		get_teacher_fn = teacher,
		get_teacher_fn_kwargs = {'ckptdir':os.path.join(PRETRAINEDDIR, args.teacher), 'logits_only':True},
		augmentations = augmentations,
		ckptdir = ckptdir,
		batch_size = args.batch,
		monitor = args.monitor,
		distiller_compile_options = {'alpha':args.alpha, 'temperature':args.temperature},
		fit_options = {'epochs':args.epochs, 'steps_per_epoch':args.steps_per_epoch},
			)

	# test the model and save test results
	print("Evaluating distilled model...")
	testing(
		test_dataset=dataset['test'].shuffle(10000).batch(args.batch),
		ckptdir=ckptdir,
		evaluate_options={"verbose":1, "steps":args.evaluate_steps}
			)

if __name__=='__main__':
    main()
