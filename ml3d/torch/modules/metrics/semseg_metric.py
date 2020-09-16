import torch
import numpy as np


class SemSegMetric(object):
    """docstring for SemSegLoss"""

    def __init__(self, pipeline, model, dataset, device):
        super(SemSegMetric, self).__init__()
        # weighted_CrossEntropyLoss
        self.pipeline = pipeline
        self.model = model
        self.dataset = dataset
        self.device = device

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
        num_classes = scores.size(-2)
        predictions = torch.max(scores, dim=-2).indices

        accuracies = []
        accuracy_mask = predictions == labels

        n_total = 0
        n_correct = 0

        for label in range(num_classes):
            label_mask = labels == label
            per_class_accuracy = (accuracy_mask & label_mask).float().sum()
            n_correct += per_class_accuracy
            per_class_accuracy /= label_mask.float().sum()
            n_total += label_mask.float().sum()
            accuracies.append(per_class_accuracy.cpu().item())
        # overall accuracy
        accuracies.append((n_correct / n_total).cpu().item())
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
        num_classes = scores.size(-2)
        predictions = torch.max(scores, dim=-2).indices

        ious = []

        n_total = 0
        n_correct = 0

        for label in range(num_classes):
            pred_mask = predictions == label
            labels_mask = labels == label
            iou = (pred_mask & labels_mask).float().sum()
            n_correct += iou
            iou = iou / (pred_mask | labels_mask).float().sum()
            n_total += (pred_mask | labels_mask).float().sum()
            ious.append(iou.cpu().item())
        ious.append((n_correct / n_total).cpu().item())
        return ious
