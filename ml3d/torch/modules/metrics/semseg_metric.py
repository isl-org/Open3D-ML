import torch
import numpy as np


class SemSegMetric(object):
    """Metrics for semantic segmentation"""

    def __init__(self, pipeline, model, dataset, device):
        super(SemSegMetric, self).__init__()
        # weighted_CrossEntropyLoss
        self.pipeline = pipeline
        self.model = model
        self.dataset = dataset
        self.device = device

    def confusion_matrix(self, scores, labels):
        r"""
            Compute the confusion matrix of one batch

            Parameters
            ----------
            scores: torch.FloatTensor, shape (B?, C, N)
                raw scores for each class
            labels: torch.LongTensor, shape (B?, N)
                ground truth labels

            Returns
            -------
            confusion matrix of this batch
        """
        num_classes = scores.size(-2)
        predictions = torch.max(scores, dim=-2).indices.cpu().data.numpy()
        labels = labels.cpu().data.numpy()

        conf_m = np.zeros((num_classes, num_classes), dtype=np.int32)

        for label in range(num_classes):
            for pred in range(num_classes):
                conf_m[label][pred] = np.sum(
                    np.logical_and(labels == label, predictions == pred))
        return conf_m

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
        accuracies.append(np.nanmean(accuracies))
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

        ious.append(np.nanmean(ious))
        return ious

    def filter_valid_label_np(self, pred, gt):
        """filter out invalid points"""

        ignored_label_inds = self.dataset.cfg.ignored_label_inds

        ignored_bool = np.zeros_like(gt, dtype=np.bool)
        for ign_label in ignored_label_inds:
            ignored_bool = np.logical_or(ignored_bool, np.equal(gt, ign_label))

        valid_idx = np.where(np.logical_not(ignored_bool))[0]

        valid_pred = pred[valid_idx]
        valid_gt = gt[valid_idx]

        # Reduce label values in the range of logit shape
        reducing_list = np.arange(0, self.dataset.num_classes, dtype=np.int64)
        inserted_value = np.zeros([1], dtype=np.int64)

        for ign_label in ignored_label_inds:
            reducing_list = np.concatenate([
                reducing_list[:ign_label], inserted_value,
                reducing_list[ign_label:]
            ], 0)

        valid_gt = reducing_list[valid_gt]

        return valid_pred, valid_gt

    def iou_np_label(self, pred, gt):
        valid_pred, valid_gt = self.filter_valid_label_np(pred, gt)
        num_classes = self.dataset.num_classes

        ious = []

        n_total = 0
        n_correct = 0

        for label in range(num_classes):
            pred_mask = valid_pred == label
            labels_mask = valid_gt == label
            iou = (pred_mask & labels_mask).sum()
            n_correct += iou
            iou = iou / (pred_mask | labels_mask).sum()
            n_total += (pred_mask | labels_mask).sum()
            ious.append(iou)

        ious.append(np.nanmean(ious))
        return ious

    def acc_np_label(self, pred, gt):
        valid_pred, valid_gt = self.filter_valid_label_np(pred, gt)
        num_classes = self.dataset.num_classes

        accuracies = []
        accuracy_mask = valid_pred == valid_gt

        n_total = 0
        n_correct = 0

        for label in range(num_classes):
            label_mask = valid_gt == label
            per_class_accuracy = (accuracy_mask & label_mask).sum()
            n_correct += per_class_accuracy
            per_class_accuracy /= label_mask.sum()
            n_total += label_mask.sum()
            accuracies.append(per_class_accuracy)

        accuracies.append(np.nanmean(accuracies))
        return accuracies
