import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(index, classes):
    out_idx = torch.arange(classes, device=index.device)
    out_idx = torch.unsqueeze(out_idx, 0)
    index = torch.unsqueeze(index, -1)
    return (index == out_idx).float()


class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss."""

    def __init__(self, loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, weight=None, avg_factor=None, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = F.cross_entropy(cls_score, label, reduction='none')

        if weight is not None:
            loss = loss * weight

        loss = loss * self.loss_weight

        if avg_factor:
            return loss.sum() / avg_factor
        else:
            return loss.mean()
