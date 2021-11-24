import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(index, classes):
    out_idx = torch.arange(classes, device=index.device)
    out_idx = torch.unsqueeze(out_idx, 0)
    index = torch.unsqueeze(index, -1)
    return (index == out_idx).float()


class FocalLoss(nn.Module):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

    Args:
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        pred_sigmoid = pred.sigmoid()
        if len(pred.shape) > 1:
            target = one_hot(target, int(pred.shape[-1]))

        target = target.type_as(pred)

        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight

        if weight is not None:
            loss = loss * weight

        loss = loss * self.loss_weight

        if avg_factor is None:
            return loss.mean()
        elif avg_factor > 0:
            return loss.sum() / avg_factor
        else:
            return loss
