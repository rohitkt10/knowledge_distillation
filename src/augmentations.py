import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow import keras as tfk

__all__ = [
    'Augmentation',
    'MixupAugmentation',
    'RCAugmentation',
    'GaussianNoiseAugmentation',
    'ShiftAugmentation',
    'AugmentedModel',
        ]

class Augmentation():
    def __call__(self, data):
        raise NotImplementedError()

class MixupAugmentation(Augmentation):
    def __init__(self, alpha=0.2):
        self.lam_dist = tfd.Beta(alpha, alpha)

    def _einsum_exp(self, x):
        n = len(x.shape)
        if n == 2:
            return "ij, i -> ij"
        elif n == 3:
            return "ijk, i -> ijk"
        elif n == 4:
            return "ijkl, i -> ijkl"
        else:
            raise NotImplementedError()

    def __call__(self, data):
        x, *y = data
        idxs = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        xp, yp = tf.gather(x, idxs, axis=0), [tf.gather(_y, idxs, axis=0) for _y in y]
        lam = self.lam_dist.sample(tf.shape(x)[0])
        exp = self._einsum_exp(x)
        xm, ym = tf.einsum(exp, x, lam) + tf.einsum(exp, x, 1.-lam), \
        [lam[:, None]*_y + (1-lam[:, None])*_yp for (_y, _yp) in zip(y, yp)]
        return (xm,) + tuple(ym)

class GaussianNoiseAugmentation(Augmentation):
    def __init__(self, mean=0., stddev=1e-2,softmax=True):
        self.gdist = tfd.Normal(loc=mean, scale=stddev)
        self.softmax = softmax
    def __call__(self, data):
        x, *y = data
        delta = self.gdist.sample(tf.shape(x))
        if self.softmax:
            xd = tf.nn.softmax(x + delta, axis=-1)
        else:
            xd = x + delta
        return (xd,) + tuple(y)

class RCAugmentation(Augmentation):
    def __call__(self, data):
        x, *y = data
        x_rc = x[:, ::-1, ::-1]
        return (x_rc,) + tuple(y)

class ShiftAugmentation(Augmentation):
    def __init__(self, shift, direction='right'):
        assert direction.lower() in ['right', 'left']
        assert shift > 0
        self.shift = int(shift)
        self.direction = direction.lower()
    def __call__(self, data):
        x, *y = data
        if self.direction == 'right':
            xs = tf.roll(x, shift=self.shift, axis=1)
        else:
            xs = tf.roll(x, shift=-self.shift, axis=1)
        return (xs,) + tuple(y)

class AugmentedModel(tfk.Model):
    def __init__(self, model, augmentations, subsample=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(augmentations) > 0, 'Pass some augmentation strategies.'
        assert np.all([isinstance(aug, Augmentation) for aug in augmentations]), \
                            'Augmentations have to be instances of the Augmentation class'
        self.model = model
        self.augmentations = augmentations
        self.subsample = subsample

    def call(self, *args, **kwargs):
        return self.model.call(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.model.save(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        return self.model.save_weights(*args, **kwargs)

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)  ## necessary to save the compile options
        super().compile(*args, **kwargs)

    def get_augmented_data(self, data):
        augmented_data = []
        n = len(data) ## number of tensors in the data tuple
        N = tf.shape(data[0])[0] # original batch size
        for augmentation in self.augmentations:
            augmented_data.append(augmentation(data))
        tensors = []  ## this will contain the list of augmented inputs and label tensors
        for i in range(n):
            tensor = []
            for j in range(len(augmented_data)):
                tensor.append(augmented_data[j][i])
            tensor = tf.concat(tensor, axis=0)
            tensors.append(tensor)

        # subsample the augmented tensors if asked
        if not self.subsample:
            return tuple(tensors)
        else:
            M = tf.shape(tensors[0])[0]  ## batch size of the augmented dataset
            idx = tf.squeeze(tf.random.categorical(logits=tf.ones((1, N)), num_samples=M, dtype=tf.int32))
            for i in range(len(tensors)):
                tensors[i] = tf.gather(tensors[i], idx)
            return tuple(tensors)

    def train_step(self, data):
        data = self.get_augmented_data(data)
        return super().train_step(data)
