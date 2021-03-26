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
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add(
                Conv2d(args[i],
                       args[i + 1],
                       bn=(not first or not preact or (i != 0)) and bn,
                       activation=activation if (not first or not preact or
                                                 (i != 0)) else None,
                       preact=preact))


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
                 use_bias=True,
                 preact=False):
        super().__init__()

        use_bias = use_bias and (not bn)
        conv_unit = conv(in_size,
                         out_size,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         use_bias=use_bias,
                         bias_initializer=tf.keras.initializers.Constant(0.0),
                         data_format="channels_first")

        if bn:
            bn_unit = tf.keras.layers.BatchNormalization(axis=1,
                                                         momentum=0.9,
                                                         epsilon=1e-05)

        if preact:
            if bn:
                self.add(bn_unit)

            if activation is not None:
                self.add(activation)

        self.add(conv_unit)

        if not preact:
            if bn:
                self.add(bn_unit)

            if activation is not None:
                self.add(activation)


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
                 use_bias: bool = True,
                 preact: bool = False):
        super().__init__(in_size,
                         out_size,
                         kernel_size,
                         stride,
                         padding,
                         activation,
                         bn,
                         init,
                         conv=tf.keras.layers.Conv1D,
                         use_bias=use_bias,
                         preact=preact)


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
                 use_bias: bool = True,
                 preact: bool = False):
        super().__init__(in_size,
                         out_size,
                         kernel_size,
                         stride,
                         padding,
                         activation,
                         bn,
                         init,
                         conv=tf.keras.layers.Conv2D,
                         use_bias=use_bias,
                         preact=preact)


class FC(tf.keras.Sequential):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 activation=tf.keras.layers.ReLU(),
                 bn: bool = False,
                 init=None,
                 preact: bool = False,
                 training=True):
        super().__init__()

        fc = tf.keras.layers.Linear(
            in_size,
            out_size,
            use_bias=not bn,
            kernel_initializer=init,
            bias_initializer=tf.keras.initializers.Constant(0.0))

        if preact:
            if bn:
                self.add(
                    tf.keras.layers.BatchNormalization(axis=1,
                                                       momentum=0.9,
                                                       epsilon=1e-05))

            if activation is not None:
                self.add(activation)

        self.add(fc)

        if not preact:
            if bn:
                self.add(
                    tf.keras.layers.BatchNormalization(axis=1,
                                                       momentum=0.9,
                                                       epsilon=1e-05))

            if activation is not None:
                self.add(activation)
