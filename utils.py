#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.metric import Metric, MeanSquaredError
from torch import Tensor, tensor


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        mask = torch.isnan(target)
        return F.l1_loss(output[~mask], target[~mask])


class RelativeError(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_relative_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_relative_error = torch.sum(torch.abs(preds - target) / target)
        n_obs = target.numel()

        self.sum_relative_error += sum_relative_error
        self.total += n_obs

    def compute(self):
        """
        Computes mean squared error over state.
        """
        return self.sum_relative_error / self.total


class RootMeanSquaredError(MeanSquaredError):
    def compute(self):
        return (self.sum_relative_error / self.total) ** 0.5


def iou(pred, target, gpu=False):
    """

    Args:
        pred ([type]): [description]
        target ([type]): [description]

    Returns:
        [type]: [description]
    """
    # pred and target has the format of
    # N, H, W
    device = "cuda" if gpu else "cpu"
    n_class = 41

    ious = torch.zeros((2, n_class)).to(device)
    for i in range(1, n_class):
        predMask = pred == i
        targetMask = target == i
        intersection = (predMask & targetMask).sum()
        union = (predMask | targetMask).sum()
        # intersection of each class
        ious[0, i] += intersection.float()
        # union of each class
        ious[1, i] += union.float()
    return ious


def pixel_acc(pred, target):
    # res = (pred == target).to(dtype=torch.float).mean()
    # do not count class 0

    res = (pred[target != 0] == target[target != 0]).to(dtype=torch.float).mean()
    return res


def rel_err(preds, target, gpu=False):
    device = "cuda" if gpu else "cpu"
    mask = torch.isnan(target)
    target, preds = target[~mask], preds[~mask]
    return (torch.abs(target - preds) / target).sum() / torch.numel(target)


def rms_err(preds, target, gpu=False):
    device = "cuda" if gpu else "cpu"
    mask = torch.isnan(target)
    target, preds = target[~mask], preds[~mask]
    return torch.sqrt(((target - preds) ** 2).sum() / torch.numel(target))


def log10_err(preds, target, gpu=False):
    device = "cuda" if gpu else "cpu"
    mask = torch.isnan(target)
    target, preds = target[~mask], preds[~mask]
    return torch.abs(torch.log10(target) - torch.log10(preds)).sum() / torch.numel(
        target
    )
