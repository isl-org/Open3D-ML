
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(index, classes):
    out_idx = torch.arange(classes, device=index.device)
    out_idx = torch.unsqueeze(out_idx, 0)
    index = torch.unsqueeze(index, -1)
    return (index==out_idx).float()


def focal_loss(pred,
               target,
               weight=None,
               gamma=2.0,
               alpha=0.25,
               avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()

    target = one_hot(target, int(pred.shape[-1])).type_as(pred)

    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight

    if weight is not None:
        loss = loss * weight
            
    if avg_factor:
        return loss.sum() / avg_factor
    else:
        return loss.mean()



if __name__ == '__main__':
    device = torch.device('cuda:0')
    input = torch.tensor([[0.8,0.4,0.5],[0.1,0.2,0.7],[0.1,0.2,0.7],[0.1,0.2,0.7]], dtype=torch.float32, device=device)
    gt = torch.tensor([0,1,2,0], dtype=torch.int64, device=device)
    print(focal_loss(input, gt))
