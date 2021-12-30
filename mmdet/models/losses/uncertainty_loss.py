import torch.nn as nn
import torch
from ..builder import LOSSES
import numpy as np
from .utils import weighted_loss
import torch.nn.functional as F
from mmdet.core import bbox_overlaps


def _gaussian_dist_pdf(val, mean, var):
    return torch.exp(- (val - mean) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * var)
# @weighted_loss
# def uncertainty_loss(pos_uncertainty_pred, pos_bbox_pred, pos_bbox_target, decoded_pred_bbox, decoded_target_bbox, eps=1e-6):
def uncertainty_loss(pos_uncertainty_pred, pos_bbox_pred, pos_bbox_target, centerness_targets, ious, eps=1e-9):
    """Forward function of loss.

    Args:
        pred (torch.Tensor): The prediction.(standard deviation)
        pred_bbox (torch.Tensor) : Predicted bbox on reg branch
        target_bbox (torch.Tensor): The learning target of the prediction.(target bbox)
    """

    # torch.pi = torch.acos(torch.zeros(1)).float() * 2
    device = torch.device("cuda")
    # torch.pi = torch.pi.to(device)
    #
    # ious = bbox_overlaps(decoded_pred_bbox.clone().detach(), decoded_target_bbox, is_aligned=True).clamp(min=eps)
    # loss_a = (((pos_bbox_target - pos_bbox_pred.clone().detach()) ** 2) / (2 * (pos_uncertainty_pred ** 2 + 1e-6))) +  (pos_uncertainty_pred ** 2).log()/2
    # # loss_a = (((pos_bbox_target-pos_bbox_pred) ** 2) / (2 * (pos_uncertainty_pred ** 2 + 1e-6))) + (pos_uncertainty_pred ** 2).log() / 2
    # # print(pos_uncertainty_pred.mean(dim=0))
    # # print('loss_a', loss_a)
    # loss_b = torch.sum(loss_a, dim = 1) # pos point num
    # # print('loss_b', loss_a)
    # loss_c = loss_b + (2 * (2 * torch.pi).log())
    # # print('loss_c', loss_a)
    # ret = ious * loss_c
    # # print('ret', ret)
    # loss = ret.mean()
    # # print('mean', loss)

    # loss_a = (pos_uncertainty_pred * -1).exp()/2
    # loss_b = (pos_bbox_target - pos_bbox_pred) ** 2
    # loss_c = (loss_a * loss_b) + pos_uncertainty_pred/2
    #loss_d = ious * torch.sum(loss_c, dim = 1)

    # smooth_loss = torch.nn.SmoothL1Loss(reduction='none')
    #
    # loss = torch.exp(-pos_uncertainty_pred) * smooth_loss(pos_bbox_pred, pos_bbox_target) + 0.5 * pos_uncertainty_pred
    # loss = torch.sum(loss, dim = 1)
    # # loss = ious * loss
    # loss = loss.mean()


    # # alpha = (pos_uncertainty_pred ** 2).log()
    # # alpha = pos_uncertainty_pred
    #
    # ## normalize bbox target
    # nor = np.ndarray(4,)
    # nor[0] = 1333 # left
    # nor[1] = 800 # top
    # nor[2] = 1333 #right
    # nor[3] = 800 #bottom
    # nor = torch.from_numpy(nor)
    # nor = nor.type(torch.float)
    # nor = nor.to(device)
    # pos_bbox_target= pos_bbox_target / nor

    # smooth_loss = torch.nn.SmoothL1Loss(reduction='none')
    # loss = torch.exp(-pos_uncertainty_pred) * smooth_loss(pos_bbox_pred, pos_bbox_target) + 0.5 * pos_uncertainty_pred
    # loss = torch.sum(loss, dim = 1)
    # loss = loss * centerness_targets
    # # loss = ious * loss
    # loss = loss.mean()

    loss = - torch.log(_gaussian_dist_pdf(pos_bbox_pred, pos_bbox_target, pos_uncertainty_pred) + 1e-9)/4
    loss = torch.sum(loss, dim = 1)
    # loss = loss * centerness_targets
    loss = loss * ious
    return loss.mean()
    # return loss


@LOSSES.register_module()
class UncertaintyLoss(nn.Module):
    """UncertaintyLoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 0.05
    """

    def __init__(self, reduction='mean', loss_weight=0.25):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    # def forward(self, pos_pred, pos_pred_bbox, pos_target_bbox, decoded_pred_bbox, decoded_target_bbox, **kwargs):
    def forward(self, pos_uncertainty_pred, pos_pred_bbox, pos_target_bbox, centerness_targets, ious, **kwargs):
        """Forward function of loss.

        Args:
            pos_pred (torch.Tensor): The prediction(standard deviation).
            pos_pred_bbox (torch.Tensor): predicted bbox (l, r, t, b)
            pos_bbox_target (torch.Tensor): positive bbox target (l, r, t, b)
            decoded_pred_bbox(torch.Tensor): decoded predicted bbox (l, r, t, b) to (xlt, ylt, xrb, yrb)
            decoded_target_bbox(torch.Tensor): decoded bbox target (l, r, t, b) to (xlt, ylt, xrb, yrb)
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """

        loss = self.loss_weight * uncertainty_loss(
            pos_uncertainty_pred,
            pos_pred_bbox,
            pos_target_bbox,
            centerness_targets,ious)
            #decoded_pred_bbox,
            #decoded_target_bbox,
            #**kwargs)

        # print('0.05 x loss', loss)
        return loss