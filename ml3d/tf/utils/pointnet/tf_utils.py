import tensorflow as tf
from typing import List, Tuple


class SharedMLP(tf.keras.Sequential):

    def __init__(
        self,
        args: List[int],
        *,
        bn: bool = False,
        activation=tf.keras.layers.ReLU(),
        preact: bool = False,
        first: bool = False,
        instance_norm: bool = False,
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add(
                Conv2d(args[i],
                       args[i + 1],
                       bn=(not first or not preact or (i != 0)) and bn,
                       activation=activation if (not first or not preact or
                                                 (i != 0)) else None,
                       preact=preact,
                       instance_norm=instance_norm))


class _ConvBase(tf.keras.Sequential):

    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size,
                 stride,
                 padding,
                 activation,
                 bn,
                 init,
                 conv=None,
                 batch_norm=None,
                 bias=True,
                 preact=False,
                 instance_norm=False,
                 instance_norm_func=None):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(in_size,
                         out_size,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=bias,
                         bias_initializer=tf.keras.initializers.Constant(0.0))

        if bn:
            bn_unit = batch_norm()
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size,
                                             affine=False,
                                             track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size,
                                             affine=False,
                                             track_running_stats=False)

        if preact:
            if bn:
                self.add(bn_unit)

            if activation is not None:
                self.add(activation)

            if not bn and instance_norm:
                self.add(in_unit)

        self.add(conv_unit)

        if not preact:
            if bn:
                self.add(bn_unit)

            if activation is not None:
                self.add(activation)

            if not bn and instance_norm:
                self.add(in_unit)


class _BNBase(tf.keras.Sequential):

    def __init__(self, in_size, batch_norm=None):
        super().__init__()
        self.add(batch_norm(kernel_initializer=tf.keras.initializers.Constant(1.0), bias_initializer=tf.keras.initializers.Constant(0.0)))


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int):
        super().__init__(in_size, batch_norm=tf.keras.layers.BatchNormalization())


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int):
        super().__init__(in_size, batch_norm=tf.keras.layers.BatchNormalization(axis=1))


class Conv1d(_ConvBase):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 activation=tf.keras.layers.ReLU(),
                 bn: bool = False,
                 init=tf.keras.initializers.he_normal(),
                 bias: bool = True,
                 preact: bool = False,
                 instance_norm=False):
        super().__init__(in_size,
                         out_size,
                         kernel_size,
                         stride,
                         padding,
                         activation,
                         bn,
                         init,
                         conv=nn.Conv1d,
                         batch_norm=BatchNorm1d,
                         bias=bias,
                         preact=preact,
                         instance_norm=instance_norm,
                         instance_norm_func=nn.InstanceNorm1d)


class Conv2d(_ConvBase):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 kernel_size: Tuple[int, int] = (1, 1),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 activation=tf.keras.layers.ReLU(),
                 bn: bool = False,
                 init=tf.keras.initializers.he_normal(),
                 bias: bool = True,
                 preact: bool = False,
                 instance_norm=False):
        super().__init__(in_size,
                         out_size,
                         kernel_size,
                         stride,
                         padding,
                         activation,
                         bn,
                         init,
                         conv=nn.Conv2d,
                         batch_norm=BatchNorm2d,
                         bias=bias,
                         preact=preact,
                         instance_norm=instance_norm,
                         instance_norm_func=nn.InstanceNorm2d)


class FC(tf.keras.Sequential):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 activation=tf.keras.layers.ReLU(),
                 bn: bool = False,
                 init=None,
                 preact: bool = False):
        super().__init__()

        fc = tf.keras.layers.Linear(in_size, out_size, bias=not bn, kernel_initializer=init, bias_initializer=tf.keras.initializers.Constant(0.0))

        if preact:
            if bn:
                self.add(tf.keras.layers.BatchNormalization(in_size))

            if activation is not None:
                self.add(activation)

        self.add(fc)

        if not preact:
            if bn:
                self.add(BatchNorm1d(out_size))

            if activation is not None:
                self.add(activation)
