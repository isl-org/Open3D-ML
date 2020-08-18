import torch
import torch.nn as nn
import numpy as np
from ....datasets.semantickitti import DataProcessing

def filter_valid_label(scores, labels, num_classes, ignored_label_inds,
                       device):
    """filter out invalid points"""
        logits = tf.reshape(results, [-1, self.cfg.num_classes])
        labels = tf.reshape(labels, [-1])

        # Boolean mask of points that should be ignored
        ignored_bool = tf.zeros_like(labels, dtype=tf.bool)
        for ign_label in self.cfg.ignored_label_inds:
            ignored_bool = tf.logical_or(ignored_bool, tf.equal(labels, ign_label))

        # Collect logits and labels that are not ignored
        valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
        valid_logits = tf.gather(logits, valid_idx, axis=0)
        valid_labels_init = tf.gather(labels, valid_idx, axis=0)

        # Reduce label values in the range of logit shape
        reducing_list = tf.range(self.cfg.num_classes, dtype=tf.int32)
        inserted_value = tf.zeros((1,), dtype=tf.int32)
        for ign_label in self.cfg.ignored_label_inds:
            reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        valid_labels = tf.gather(reducing_list, valid_labels_init)

    return valid_scores, valid_labels


class SemSegLoss(object):
    """Loss functions for semantic segmentation"""
    def __init__(self, pipeline, model, dataset):
        super(SemSegLoss, self).__init__()
        # weighted_CrossEntropyLoss


        if 'class_weights' in dataset.cfg.keys():
            weights = DataProcessing.get_class_weights(dataset.cfg.class_weights)
            self.class_weights = tf.convert_to_tensor(weights, dtype=tf.float32)