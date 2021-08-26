import tensorflow as tf
from tensorflow import keras as tfk
from ._distiller import Distiller
from .basic_distiller import BasicDistiller
from ..augmentations import Augmentation, AugmentedModel

__all__ = ['L2Distiller']

class L2Distiller(BasicDistiller):
    """
    This class modifies the basic distillation framework
    of Hinton et al (2015) [1] by replacing the KL divergence
    loss for distillation with the squared l2 norm
    of the difference between the soft logits produced
    by the student and teacher network.


    REFERENCES:
    [1] Hinton, G., Vinyals, O., & Dean, J. (2015).
        Distilling the knowledge in a neural network.
        arXiv preprint arXiv:1503.02531.
    """
    def _set_distillation_loss(self,):
        self.distill_loss = tfk.losses.MeanSquaredError(name="distill_loss")
