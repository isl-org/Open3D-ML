import tensorflow as tf
import math


class CosineWarmupLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]
