import tensorflow as tf
from tensorflow import keras as tfk
from ._distiller import Distiller
from ..augmentations import Augmentation, AugmentedModel
from ..utils.loss_functions import BinaryKLDivergence

class RouteConstrainedDistiller(Distiller):
    def train_step(self, data):
        return
