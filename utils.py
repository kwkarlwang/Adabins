# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.metric import Metric
from torchmetrics import MeanSquaredError
from torch import Tensor, tensor
from typing import Optional, Callable, Any

from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss import chamfer_distance



class AdabinsLoss(nn.Module):
    def __init__(self, lamb, chamfer):
        super().__init__()
        self.depth_loss = EigenLoss(lamb)
        self.bins_loss = BinsChamferLoss()
        self.chamfer = chamfer

    def forward(self,output,target):
        bin_edges, pred = output
        loss = self.depth_loss(pred, target)
        if self.chamfer > 0:
            loss += self.chamfer * self.bins_loss(bin_edges,target)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        mask = torch.isnan(target)
        return F.l1_loss(output[~mask], target[~mask])


class BinsChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)

        target_points = target_depth_maps.flatten(1)
        mask = target_points.ge(1e-3)
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = (
            torch.Tensor([len(t) for t in target_points])
            .long()
            .to(target_depth_maps.device)
        )

        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)

        loss, _ = chamfer_distance(
            x=input_points, y=target_points, y_lengths=target_lengths
        )
        return loss


# refer to https://proceedings.neurips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf
class EigenLoss(nn.Module):
    def __init__(self, lamb=0.5):
        super().__init__()
        self.lamb = lamb

    def forward(self, output, target, mask=None):
        if mask is not None:
            output, target = output[mask], target[mask]
        diff = torch.log(output) - torch.log(target)
        return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * 10.0


class RelativeError(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        device="cuda",
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_relative_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.sum_relative_error = self.sum_relative_error.to(device)
        self.total = self.total.to(device)

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
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
        device="cuda",
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.sum_squared_error = self.sum_squared_error.to(device)
        self.total = self.total.to(device)

    def compute(self):
        return (self.sum_squared_error / self.total) ** 0.5


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
    mask = torch.isnan(target)
    target, preds = target[~mask], preds[~mask]
    return (torch.abs(target - preds) / target).sum() / torch.numel(target)


def rms_err(preds, target, gpu=False):
    mask = torch.isnan(target)
    target, preds = target[~mask], preds[~mask]
    return torch.sqrt(((target - preds) ** 2).sum() / torch.numel(target))


def log10_err(preds, target, gpu=False):
    mask = torch.isnan(target)
    target, preds = target[~mask], preds[~mask]
    return torch.abs(torch.log10(target) - torch.log10(preds)).sum() / torch.numel(
        target
    )
