import numpy as np
import warnings


class SemSegMetric(object):
    """Metrics for semantic segmentation.

    Accumulate confusion matrix over training loop and
    computes accuracy and mean IoU.
    """

    def __init__(self):
        super(SemSegMetric, self).__init__()
        self.confusion_matrix = None
        self.num_classes = None

    def update(self, scores, labels):
        conf = self.get_confusion_matrix(scores, labels)
        if self.confusion_matrix is None:
            self.confusion_matrix = conf.copy()
            self.num_classes = conf.shape[0]
        else:
            assert self.confusion_matrix.shape == conf.shape
            self.confusion_matrix += conf

    def acc(self):
        """Compute the per-class accuracies and the overall accuracy.

        Args:
            scores (torch.FloatTensor, shape (B?, C, N):
                raw scores for each class.
            labels (torch.LongTensor, shape (B?, N)):
                ground truth labels.

        Returns:
            A list of floats of length num_classes+1.
            Consists of per class accuracy. Last item is Overall Accuracy.
        """
        if self.confusion_matrix is None:
            return None

        accs = []
        for label in range(self.num_classes):
            tp = np.longlong(self.confusion_matrix[label, label])
            fn = np.longlong(self.confusion_matrix[label, :].sum()) - tp

            if tp + fn == 0:
                acc = float('nan')
            else:
                acc = tp / (tp + fn)

            accs.append(acc)

        accs.append(np.nanmean(accs))

        return accs

    def iou(self):
        """Compute the per-class IoU and the mean IoU.

        Args:
            scores (torch.FloatTensor, shape (B?, C, N):
                raw scores for each class.
            labels (torch.LongTensor, shape (B?, N)):
                ground truth labels.

        Returns:
            A list of floats of length num_classes+1.
            Consists of per class IoU. Last item is mIoU.
        """
        if self.confusion_matrix is None:
            return None

        ious = []
        for label in range(self.num_classes):
            tp = np.longlong(self.confusion_matrix[label, label])
            fn = np.longlong(self.confusion_matrix[label, :].sum()) - tp
            fp = np.longlong(self.confusion_matrix[:, label].sum()) - tp

            if tp + fp + fn == 0:
                iou = float('nan')
            else:
                iou = (tp) / (tp + fp + fn)

            ious.append(iou)

        ious.append(np.nanmean(ious))

        return ious

    def reset(self):
        self.confusion_matrix = None

    @staticmethod
    def get_confusion_matrix(scores, labels):
        """Computes the confusion matrix of one batch

        Args:
            scores (torch.FloatTensor, shape (B?, N, C):
                raw scores for each class.
            labels (torch.LongTensor, shape (B?, N)):
                ground truth labels.

        Returns:
            Confusion matrix for current batch.
        """
        C = scores.size(-1)
        y_pred = scores.detach().cpu().numpy().reshape(-1, C)  # (N, C)
        y_pred = np.argmax(y_pred, axis=1)  # (N,)

        y_true = labels.detach().cpu().numpy().reshape(-1,)

        y = np.bincount(C * y_true + y_pred, minlength=C * C)

        if len(y) < C * C:
            y = np.concatenate([y, np.zeros((C * C - len(y)), dtype=np.long)])
        else:
            if len(y) > C * C:
                warnings.warn(
                    "Prediction has fewer classes than ground truth. This may affect accuracy."
                )
            y = y[-(C * C):]  # last c*c elements.

        y = y.reshape(C, C)

        return y
