import tensorflow as tf


class FocalLoss(tf.Module):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

    Args:
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def __call__(self, pred, target, weight=None, avg_factor=None):

        pred_sigmoid = tf.math.sigmoid(pred)

        if len(pred.shape) > 1:
            target = tf.one_hot(target, int(pred.shape[-1]))
        target = tf.cast(target, pred.dtype)

        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)

        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * tf.pow(pt, self.gamma)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(target,
                                                       pred) * focal_weight

        if weight is not None:
            loss = loss * weight

        loss = loss * self.loss_weight

        if avg_factor:
            return tf.reduce_sum(loss) / avg_factor
        else:
            return tf.reduce_mean(loss)
