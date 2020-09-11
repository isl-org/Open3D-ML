import tensorflow as tf
import numpy as np


class SemSegMetric(object):
    """docstring for SemSegLoss"""

    def __init__(self, pipeline, model, dataset):
        super(SemSegMetric, self).__init__()
        # weighted_CrossEntropyLoss
        self.pipeline = pipeline
        self.model = model
        self.dataset = dataset

    def acc(self, scores, labels):
        r"""
            Compute the per-class accuracies and the overall accuracy 

            Parameters
            ----------
            scores: torch.FloatTensor, shape (B?, C, N)
                raw scores for each class
            labels: torch.LongTensor, shape (B?, N)
                ground truth labels

            Returns
            -------
            list of floats of length num_classes+1 
            (last item is overall accuracy)
        """
        correct_prediction = tf.nn.in_top_k(labels, scores, 1)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        num_classes = scores.shape[-1]
        predictions = tf.argmax(scores, axis=-1)

        accuracies = []
        labels = tf.cast(labels, tf.int64)
        accuracy_mask = predictions == labels

        total_corret = 0
        total_label = 0

        for label in range(num_classes):
            label_mask = labels == label
            num_correct = (accuracy_mask & label_mask).numpy().sum()
            num_label = label_mask.numpy().sum()
            if num_label == 0:
                num_label = 1
            per_class_accuracy = num_correct / num_label
            total_corret += num_correct
            total_label += num_label
            accuracies.append(per_class_accuracy)
        # overall accuracy
        accuracies.append(total_corret / total_label)
        #accuracies = np.array(accuracies)

        return accuracies

    def iou(self, scores, labels):
        r"""
            Compute the per-class IoU and the mean IoU 

            Parameters
            ----------
            scores: torch.FloatTensor, shape (B?, C, N)
                raw scores for each class
            labels: torch.LongTensor, shape (B?, N)
                ground truth labels

            Returns
            -------
            list of floats of length num_classes+1 (last item is mIoU)
        """
        num_classes = scores.shape[-1]
        predictions = tf.argmax(scores, axis=-1).numpy()
        labels = tf.cast(labels, tf.int64).numpy()

        ious = []
        total_i = 0
        total_u = 0

        for label in range(num_classes):
            pred_mask = predictions == label
            labels_mask = labels == label
            num_i = (pred_mask & labels_mask).sum()
            num_u = (pred_mask | labels_mask).sum()
            if num_u == 0:
                num_u = 1
            iou = num_i / num_u
            total_i += num_i
            total_u += num_u
            ious.append(iou)
        ious.append(total_i / total_u)
        return ious
