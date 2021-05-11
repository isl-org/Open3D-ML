import tensorflow as tf
import numpy as np


class OneCycleScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Scheduler class for cyclic learning rate scheduling.

    Args:
        total_step: number of steps for one cycle.
        lr_max: maximum cyclic learning rate.
        div_factor: factor by which initial learning starts.
    """

    def __init__(self, total_step, lr_max=0.002, div_factor=10.0):

        self.lr_max = lr_max
        self.div_factor = div_factor
        self.total_step = total_step
        super(OneCycleScheduler, self).__init__()

    def __call__(self, step):
        lr_low = self.lr_max / self.div_factor

        angle = (np.pi / self.total_step * step)
        lr1 = tf.abs(lr_low + (self.lr_max - lr_low) * tf.math.sin(angle))

        angle = (np.pi / self.total_step) * (
            (step - self.total_step / 2) % self.total_step)

        lr2 = tf.abs(self.lr_max * tf.math.cos(angle))

        lr = tf.where(step < self.total_step / 2, lr1, lr2)

        return lr
