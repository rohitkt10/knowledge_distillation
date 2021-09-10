import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from ._distiller import Distiller
from ..augmentations import Augmentation, AugmentedModel
from ..utils.loss_functions import BinaryKLDivergence

__all__ = ['BasicDistiller']

class BasicDistiller(Distiller):
    """
    Implementation of the basic distillation process
    as detailed in Hinton et al (2015) [1].

    The loss function is a combination of the standard
    classification losss from the data plus the classification
    loss on the temperature modulated soft targets generated
    from the teacher model.

    Currently this model is setup for distilling classifiers (binary or multiclass).

    REFERENCES:
    [1] Hinton, G., Vinyals, O., & Dean, J. (2015).
        Distilling the knowledge in a neural network.
        arXiv preprint arXiv:1503.02531.
    """
    def __init__(self,
                student,
                teacher=None,
                precompute_teacher_logits=None,
                *args, **kwargs):
        super().__init__(student, teacher, precompute_teacher_logits, *args, **kwargs)

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
        self.alpha = alpha
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

    def train_step(self, data):
        if self.precompute_teacher_logits:
            data = data[0] + (data[1],) ## rearrange data

        if isinstance(self.student, AugmentedModel):
            data = self.student.get_augmented_data(data)

        # unpack data tuple and get teacher model predictions
        if self.precompute_teacher_logits:
            x, y, y_pred_teacher_logits = data
        else:
            x, y = data
            y_pred_teacher_logits = self.teacher(x, training=False) ## logits of teacher model (batch, numtasks,)
        y_pred_teacher = self.last_actfn(y_pred_teacher_logits, **self.last_actfn_kwargs)
        #print(y_pred_teacher)

        # record differentiable operations
        with tf.GradientTape() as tape:
            # get the data loss
            y_pred_logits = self.student(x, training=True)  ## logits of the student model
            y_pred = self.last_actfn(y_pred_logits, **self.last_actfn_kwargs)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.student.losses)

            # get the distillation loss
            soft_teacher_logits = y_pred_teacher_logits / self.temperature
            soft_student_logits = y_pred_logits / self.temperature
            distill_loss = self._get_distillation_loss(soft_teacher_logits, soft_student_logits)

            # get the total loss
            total_loss = loss + distill_loss*self.alpha

        # compute gradients and take SGD step
        variables = self.student.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # finish remaining steps of train_step
        self.compiled_metrics.update_state(y, y_pred)
        res = {m.name:m.result() for m in self.metrics}
        res.update({"distillation_loss":distill_loss})
        return res

    def test_step(self, data):
        if self.precompute_teacher_logits:
            data = data[0] + (data[1],) ## rearrange data

        x, *y = data
        #y_true = y[0]
        y_pred_logits = self.student(x, training=False)
        y_pred = self.last_actfn(y_pred_logits, **self.last_actfn_kwargs)
        loss = self.compiled_loss(y[0], y_pred,regularization_losses=0.)
        self.compiled_metrics.update_state(y[0], y_pred)
        results = {m.name: m.result() for m in self.metrics}
        return results
