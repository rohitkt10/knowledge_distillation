import tensorflow as tf
from tensorflow import keras as tfk
from ._distiller import Distiller
from ..augmentations import Augmentation, AugmentedModel
from ..utils.loss_functions import BinaryKLDivergence
from tensorflow.keras.losses import KLDivergence

__all__ = ['OnlineDistiller']

class OnlineDistiller(Distiller):
    def __init__(self,
                student,
                teacher,
                *args, **kwargs):
        super().__init__(student, teacher, None, *args, **kwargs)
        self.teacher.trainable = True

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
        self.teacher.compile(optimizer, loss, metrics, *args, **kwargs)
        super().compile(optimizer, loss, metrics, *args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

        # determine the distillation loss function and final layer activation
        if 'binary' in loss.__class__.__name__.lower():
            self.last_actfn = tf.math.sigmoid
            self.last_actfn_kwargs = {}
            self._set_distillation_loss(losstype='binary')
        elif 'categorical' in loss.__class__.__name__.lower():
            self.last_actfn = tf.nn.softmax
            self.last_actfn_kwargs = {'axis':-1}
            self._set_distillation_loss(losstype='categorical')
        else:
            raise ValueError("Inappropriate loss function passed.")

    def _set_distillation_loss(self, losstype='binary'):
        if losstype == 'binary':
            self.distill_loss = BinaryKLDivergence(name="distill_loss")
        elif losstype == 'categorical':
            self.distill_loss = KLDivergence(name="distill_loss")

    def _get_distillation_loss(self, y_teacher, y_student):
        loss = self.distill_loss(
                        self.last_actfn(y_teacher, **self.last_actfn_kwargs),
                        self.last_actfn(y_student, **self.last_actfn_kwargs),
                                )
        return loss

    def train_step(self, data):
        # unpack data
        if isinstance(self.student, AugmentedModel):
            data = self.student.get_augmented_data(data)
        x, y = data

        # record differentiable operations
        with tf.GradientTape() as tape:
            # get the teacher and student predictions
            y_pred_logits = self.student(x, training=True)  ## logits of the student model
            y_pred_teacher_logits = self.teacher(x, training=True) ## logits of the teacher model
            y_pred = self.last_actfn(y_pred_logits, **self.last_actfn_kwargs)
            y_pred_teacher = self.last_actfn(y_pred_teacher_logits, **self.last_actfn_kwargs)

            # get the total student loss
            student_loss = self.compiled_loss(y, y_pred, regularization_losses=self.student.losses)  ## data loss + reg. loss
            soft_teacher_logits = y_pred_teacher_logits / self.temperature
            soft_student_logits = y_pred_logits / self.temperature
            distill_loss = self._get_distillation_loss(soft_teacher_logits, soft_student_logits)
            total_student_loss = student_loss + distill_loss*self.alpha

            # get the total teacher loss
            teacher_loss = self.loss(y, y_pred_teacher)
            total_teacher_loss = teacher_loss + tf.add_n(self.teacher.losses)

            total_loss = total_student_loss + total_teacher_loss

        # compute gradients and take SGD step
        variables = self.student.trainable_variables + self.teacher.variables
        gradients = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # finish remaining steps of train_step
        self.compiled_metrics.update_state(y, y_pred)
        res = {m.name:m.result() for m in self.metrics}
        res.update({"teacher_loss":teacher_loss, "distillation_loss":distill_loss})
        return res

    def test_step(self, data):
        x, y = data

        #y_true = y[0]
        y_pred_logits = self.student(x, training=False)
        y_pred = self.last_actfn(y_pred_logits, **self.last_actfn_kwargs)
        loss = self.compiled_loss(y, y_pred,regularization_losses=0.)
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        return results
