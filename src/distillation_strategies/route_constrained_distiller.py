import tensorflow as tf
from tensorflow import keras as tfk
from .basic_distiller import BasicDistiller
from ..augmentations import Augmentation, AugmentedModel
from ..utils.loss_functions import BinaryKLDivergence

class RouteConstrainedDistiller(BasicDistiller):
    def __init__(self, student, teacher=None, *args, **kwargs):
        super().__init__(student, teacher, *args, **kwargs)

    @property
    def teacher(self):
        return _teacher

    @teacher.setter
    def teacher(self, model):
        self._teacher = model
