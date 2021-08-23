import tensorflow as tf
from tensorflow.keras.losses import Loss, kl_divergence

__all__ = ['BinaryKLDivergence']

class BinaryKLDivergence(Loss):
    """
    Implementation of the KL Divergence loss for Bernoulli distributed
    variables. This implementation works for both single and multi label
    cases.
    """
    def __init__(self, name=None):
        super().__init__(name=name)

    def _convert_to_categorical(self, y):
        pos = y       # probability of positive class
        neg = 1. - y  # probability of negative class
        cat = tf.stack([pos, neg],axis=-1) # (batch size, numlabels, numcategories)
        return cat

    def call(self, y_true, y_pred):
        y_true = self._convert_to_categorical(y_true)
        y_pred = self._convert_to_categorical(y_pred)
        loss = kl_divergence(y_true, y_pred) ## shape = (batch, 1) or (batch,)
        loss = tf.reduce_mean(loss,)
        return loss
