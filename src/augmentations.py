import tensorflow as tf
from tensorflow_probability import distributions as tfd

class Augmentation():
    def __call__(self, data):
        raise NotImplementedError()

class MixupAugmentation(Augmentation):
    def __init__(self, alpha=1.):
        self.lam_dist = tfd.Beta(alpha, alpha)
    def __call__(self, data):
        x, y = data
        idxs = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        xp, yp = tf.gather(x, idxs, axis=0), tf.gather(y, idxs, axis=0)
        lam = self.lam_dist.sample(tf.shape(x)[0])
        xm, ym = lam[:, None, None]*x + (1-lam[:, None, None])*xp, lam[:, None]*y + (1-lam[:, None])*yp
        return (xm, ym)

class GaussianNoiseAugmentation(Augmentation):
    def __init__(self, mean=0., stddev=1e-2,softmax=True):
        self.gdist = tfd.Normal(loc=mean, scale=stddev)
        self.softmax = softmax
    def __call__(self, data):
        x, y = data
        delta = self.gdist.sample(tf.shape(x))
        if self.softmax:
            xd = tf.nn.softmax(x + delta, axis=-1)
        else:
            xd = x + delta
        return (xd, y)

class RCAugmentation(Augmentation):
    def __call__(self, data):
    x, y = data
    x_rc = x[:, ::-1, ::-1]
    return (x_rc, y)


class ShiftAugmentation(Augmentation):
    def __init__(self, shift, direction='right'):
        assert direction.lower() in ['right', 'left']
        assert shift > 0
        self.shift = int(shift)
        self.direction = direction.lower()
    def __call__(self, data):
        x, y = data
        if self.direction == 'right':
            xs = tf.roll(x, shift=self.shift, axis=1)
        else:
            xs = tf.roll(x, shift=-self.shift, axis=1)
        return (xs, y)