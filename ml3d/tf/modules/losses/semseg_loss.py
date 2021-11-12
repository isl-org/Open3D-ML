import tensorflow as tf
from ....datasets.utils import DataProcessing as DP


class SemSegLoss(object):
    """Loss functions for semantic segmentation."""

    def __init__(self, pipeline, model, dataset):
        super(SemSegLoss, self).__init__()
        # weighted_CrossEntropyLoss
        self.num_classes = model.cfg.num_classes
        self.ignored_label_inds = model.cfg.ignored_label_inds
        self.class_weights = None

        if 'class_weights' in dataset.cfg.keys() and len(
                dataset.cfg.class_weights) != 0:
            weights = DP.get_class_weights(dataset.cfg.class_weights)
            self.class_weights = tf.convert_to_tensor(weights, dtype=tf.float32)

    def weighted_CrossEntropyLoss(self, logits, labels):
        if self.class_weights is None:
            return tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
        else:
            # calculate the weighted cross entropy according to the inverse frequency
            one_hot_labels = tf.one_hot(labels, depth=self.num_classes)
            weights = tf.reduce_sum(self.class_weights * one_hot_labels, axis=1)
            unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=one_hot_labels)
            weighted_losses = unweighted_losses * weights
            output_loss = tf.reduce_mean(weighted_losses)

            return output_loss

    def filter_valid_label(self, scores, labels):
        """Filter out invalid points."""
        logits = tf.reshape(scores, [-1, self.num_classes])
        labels = tf.reshape(labels, [-1])

        # Boolean mask of points that should be ignored
        ignored_bool = tf.zeros_like(labels, dtype=tf.bool)
        for ign_label in self.ignored_label_inds:
            ignored_bool = tf.logical_or(ignored_bool,
                                         tf.equal(labels, ign_label))

        # Collect logits and labels that are not ignored
        valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
        valid_logits = tf.gather(logits, valid_idx, axis=0)
        valid_labels_init = tf.gather(labels, valid_idx, axis=0)

        # Reduce label values in the range of logit shape
        reducing_list = tf.range(self.num_classes, dtype=tf.int32)
        inserted_value = tf.zeros((1,), dtype=tf.int32)
        for ign_label in self.ignored_label_inds:
            if ign_label >= 0:
                reducing_list = tf.concat([
                    reducing_list[:ign_label], inserted_value,
                    reducing_list[ign_label:]
                ], 0)

        valid_labels = tf.gather(reducing_list, valid_labels_init)

        return valid_logits, valid_labels
