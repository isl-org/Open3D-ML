import tensorflow as tf


class CrossEntropyLoss(tf.Module):
    """CrossEntropyLoss."""

    def __init__(self, loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def __call__(self,
                 cls_score,
                 label,
                 weight=None,
                 avg_factor=None,
                 **kwargs):
        """Forward function.

        Args:
            cls_score (tf.Tensor): The prediction.
            label (tf.Tensor): The learning label of the prediction.
            weight (tf.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            tf.Tensor: The calculated loss
        """
        if weight is not None:
            loss = self.loss_fn(label, cls_score, sample_weight=weight)
        else:
            loss = self.loss_fn(label, cls_score)

        loss = loss * self.loss_weight

        if avg_factor:
            return tf.reduce_sum(loss) / avg_factor
        else:
            return tf.reduce_mean(loss)
