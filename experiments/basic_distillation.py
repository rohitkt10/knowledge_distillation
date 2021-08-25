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
from src.distillation_strategies import BasicDistiller, EuclideanDistiller
from pdb import set_trace as keyboard
from importlib import import_module, __import__
from datetime import datetime

DATADIRS = {
	'deepsea':os.path.join(DATAROOTDIR, 'deepsea')
}
DATALOADERS = {
	'deepsea':utils.get_deepsea_dataset,
}

def main():
    keyboard()
    return

if __name__=='__main__':
    main()
