import tensorflow as tf


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, tf.keras.layers.BatchNormalization):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(self,
                 model,
                 bn_lambda,
                 last_epoch=-1,
                 setter=set_bn_momentum_default):
        if not isinstance(model, tf.keras.layers.Layer):
            raise RuntimeError(
                "Class '{}' is not a Tensorflow Keras Layer".format(
                    type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))
