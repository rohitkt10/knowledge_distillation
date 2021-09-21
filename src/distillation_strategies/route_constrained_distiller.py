import tensorflow as tf
from tensorflow import keras as tfk
from .basic_distiller import BasicDistiller
from ..augmentations import Augmentation, AugmentedModel
from ..utils.loss_functions import BinaryKLDivergence

class RouteConstrainedDistiller(BasicDistiller):
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, a):
        self._alpha = a
    
    def compile(self, optimizer, loss, metrics=None, alpha=1., temperature=1., *args, **kwargs):
        """
        Compile the distiller.

        We determine the distillation loss function from context i.e. the data
        loss function passed to this method. In Hinton et al., the distillation
        loss used is the KL divergence between the temperature modulated labels
        produced by the teacher and student networks. For convenience, please
        only pass an instance of a keras Loss class. For example, pass
        an instance of tfk.losses.BinaryCrossentropy instead of
        the tfk.losses.binary_crossentropy function.

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
                The optimizer to use to minimize the total loss (standard loss + distiller loss).
        loss : tf.keras.losses.Loss
                The loss function to use to measure the discrepancy between the
                student model predictions and the data labels.
        metrics : list of tf.keras.metrics.Metrics
        alpha : <float>
                The coefficient to scale the distillation loss (default: 1).
        temperature : <float>
                The temperature used to calculate soft labels for distillation (default: 1).
        Returns
        -------
        None
        """
        assert isinstance(loss, tfk.losses.Loss)
        self.student.compile(optimizer, loss, metrics, *args, **kwargs)
        super().compile(optimizer, loss, metrics, *args, **kwargs)
        self._alpha = alpha
        self.temperature = temperature

        # determine the distillation loss function and final layer activation
        self._set_distillation_loss()
        if 'binary' in loss.__class__.__name__.lower():
            self.last_actfn = tf.math.sigmoid
            self.last_actfn_kwargs = {}
        elif 'categorical' in loss.__class__.__name__.lower():
            self.last_actfn = tf.nn.softmax
            self.last_actfn_kwargs = {'axis':-1}
        else:
            raise ValueError("Inappropriate loss function passed.")

    def _set_distillation_loss(self,):
        self.distill_loss = BinaryKLDivergence(name="distill_loss")

    def _get_distillation_loss(self, y_teacher, y_student):
        loss = self.distill_loss(
                        self.last_actfn(y_teacher, **self.last_actfn_kwargs),
                        self.last_actfn(y_student, **self.last_actfn_kwargs),
                                )
        return loss
