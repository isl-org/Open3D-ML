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
        if self.conf_matrix is None:
            return None

        accs = []
        for label in range(self.num_classes):
            tp = np.longlong(self.conf_matrix[label, label])
            fn = np.longlong(self.conf_matrix[label, :].sum()) - tp

            if tp + fn == 0:
                acc = float('nan')
            else:
                acc = tp / (tp + fn)

            accs.append(acc)

        accs.append(np.nanmean(accs))

        return accs

    def iou(self):
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
        C = scores.size(-1)
        y_pred = scores.detach().cpu().numpy().reshape(-1, C)  # (N, C)
        y_pred = np.argmax(y_pred, axis=1)  # (N,)

        y_true = labels.detach().cpu().numpy().reshape(-1,)

        y = np.bincount(C * y_true + y_pred, minlength=C * C)

        if len(y) < C * C:
            y = np.concatenate([y, np.zeros((C * C - len(y)), dtype=np.long)])

        y = y.reshape(C, C)

        return y
