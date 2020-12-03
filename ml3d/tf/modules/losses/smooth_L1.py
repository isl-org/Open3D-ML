import tensorflow as tf


class SmoothL1Loss(tf.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def __call__(self, pred, target, weight=None, avg_factor=None, **kwargs):
        """Forward function.

        Args:
            pred (tf.Tensor): The prediction.
            target (tf.Tensor): The learning target of the prediction.
            weight (tf.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert pred.shape == target.shape and tf.size(target) > 0

        diff = tf.abs(pred - target)

        loss = tf.where(diff < self.beta, 0.5 * diff * diff / self.beta,
                        diff - 0.5 * self.beta)

        if weight is not None:
            loss = loss * weight

        loss = loss * self.loss_weight

        if avg_factor:
            return tf.reduce_sum(loss) / avg_factor
        else:
            return tf.reduce_mean(loss)
