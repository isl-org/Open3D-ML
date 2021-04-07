import torch
import numpy as np


class SemSegMetric(object):
    """Metrics for semantic segmentation"""

    def __init__(self):
        super(SemSegMetric, self).__init__()
        self.conf_matrix = None
        self.num_classes = None

    def update(self, scores, labels):
        conf = self.confusion_matrix(scores, labels)
        if self.conf_matrix is None:
            self.conf_matrix = conf.copy()
            self.num_classes = conf.shape[0]
        else:
            assert self.conf_matrix.shape == conf.shape
            self.conf_matrix += conf

    def acc(self):
        if self.conf_matrix is None:
            return None

        accs = []
        for label in range(self.num_classes):
            tp = np.longlong(self.conf_matrix[label, label])
            fn = np.longlong(self.conf_matrix[label, :].sum()) - tp
            fp = np.longlong(self.conf_matrix[:, label].sum()) - tp
            tn = np.longlong(self.conf_matrix.sum()) - (tp + fp + fn)

            if tp + tn + fp + fn == 0:
                acc = float('nan')
            else:
                acc = (tp + tn) / (tp + tn + fp + fn)

            accs.append(acc)

        accs.append(np.nanmean(accs))

        return accs

    def iou(self):
        if self.conf_matrix is None:
            return None

        ious = []
        for label in range(self.num_classes):
            tp = np.longlong(self.conf_matrix[label, label])
            fn = np.longlong(self.conf_matrix[label, :].sum()) - tp
            fp = np.longlong(self.conf_matrix[:, label].sum()) - tp

            if tp + fp + fn == 0:
                iou = float('nan')
            else:
                iou = (tp) / (tp + fp + fn)

            ious.append(iou)

        ious.append(np.nanmean(ious))

        return ious

    def reset(self):
        self.conf_matrix = None

    @staticmethod
    def confusion_matrix(scores, labels):
        r"""
            Compute the confusion matrix of one batch

            Parameters
            ----------
            scores: torch.FloatTensor, shape (B?, N, C)
                raw scores for each class
            labels: torch.LongTensor, shape (B?, N)
                ground truth labels

            Returns
            -------
            confusion matrix of this batch
        """
        N = scores.size(-2)
        C = scores.size(-1)
        y_pred = scores.detach().cpu().numpy().reshape(-1, C)  # (N, C)
        y_pred = np.argmax(y_pred, axis=1)  # (N,)

        y_true = labels.detach().cpu().numpy().reshape(-1,)

        y = np.bincount(C * y_true + y_pred, minlength=C * C)

        if len(y) < C * C:
            y = np.concatenate([y, np.zeros((C * C - len(y)), dtype=np.long)])

        y = y.reshape(C, C)

        return y
