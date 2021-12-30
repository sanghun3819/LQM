import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from mmdet.models.losses.gfocal_loss import QualityFocalLoss
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead

import numpy as np
from typing import Iterator, List, Tuple, Union

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)     ##### probability of discrete regression
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)   ##### linear combination of (discrete reg & probability)
        return x

class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor(torch.Tensor): float matrix of Nx4.
    """

    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 4, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes" :
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: str) -> "Boxes":
        return Boxes(self.tensor.to(device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: BoxSizeType) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        self.tensor[:, 0].clamp_(min=0, max=w)
        self.tensor[:, 1].clamp_(min=0, max=h)
        self.tensor[:, 2].clamp_(min=0, max=w)
        self.tensor[:, 3].clamp_(min=0, max=h)

    def nonempty(self, threshold: int = 0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """
        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
        with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: BoxSizeType, boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @staticmethod
    def cat(boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        assert len(boxes_list) > 0
        assert all(isinstance(box, Boxes) for box in boxes_list)

        cat_boxes = type(boxes_list[0])(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor

@HEADS.register_module()
class GFocalAdvanceHeadv3(AnchorHead):
    """Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.

    GFocal head structure is similar with GFL head, however GFocal uses
    the statistics of learned distribution to guide the 
    localization quality estimation (LQE)

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
        reg_topk (int): top-k statistics of distribution to guide LQE
        reg_channels (int): hidden layer unit to generate LQE
    Example:
        >>> self = GFocalHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 reg_max=16,
                 reg_topk=4,
                 reg_channels=64,
                 add_mean=True,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.reg_max = reg_max             ##### normalized discrete regression (0~16) : x axis (distribution of regression graph)
        self.reg_topk = reg_topk           ##### topk : 4 (default)
        self.reg_channels = reg_channels   ##### fully connected layer channel : H x W x (64)
        self.add_mean = add_mean           ##### topk + mean
        self.total_dim = reg_topk          ##### n of fully connected layer input : H x W x 4(n), default = 4

        ###########################################################################
        self.repeated_conv = 3
        self.repeated_conv_channels = 256
        self.dcn_conv = False
        ###########################################################################

        if add_mean:
            self.total_dim += 1
        print('total dim = ', self.total_dim * 4)

        super(GFocalAdvanceHeadv3, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral = Integral(self.reg_max) 
        self.loss_dfl = build_loss(loss_dfl)

        self.loss_cls_offset = build_loss(dict(
            type='QualityFocalLoss_sub',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0))

        self.loss_bbox_offset = build_loss(dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):                             ##### x4 conv layer
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)) 
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        
        ###########################################################################
        self.conv_cls_refine = nn.ModuleList()
        self.conv_reg_refine = nn.ModuleList()
        for i in range(self.repeated_conv):
            chn = self.feat_channels if i == 0 else self.repeated_conv_channels
            if self.dcn_conv and i == self.repeated_conv - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.conv_cls_refine.append(
                ConvModule(
                    chn,
                    self.repeated_conv_channels,
                    1,
                    stride=1,
                    # padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias='auto'))
        for i in range(self.repeated_conv):
            chn = self.feat_channels if i == 0 else self.repeated_conv_channels
            if self.dcn_conv and i == self.repeated_conv - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.conv_reg_refine.append(
                ConvModule(
                    chn,
                    self.repeated_conv_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias='auto'))
        self.conv_cls_refine_pred = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg_refine_pred = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.add_module("uncertainty_cls_branch", UncertaintyClsBranch(self.feat_channels, self.feat_channels, uncertain=True))        
        self.add_module("uncertainty_reg_branch", UncertaintyRegBranch(self.feat_channels, 64, uncertain=True))
        ###########################################################################

        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(                                      ##### cls prediction conv layer
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(                                      ##### reg prediction conv layer
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList(                                   ##### stride scale for fpn
            [Scale(1.0) for _ in self.anchor_generator.strides])

        # original
        # conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]  ### 4 direction x topk(4)+mean : input
        # conf_vector += [self.relu]
        # conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()] ##### Distribution-Guided Quality Predictor (DGQP) : 1x1 --> ReLu --> 1x1 --> Sigmoid

        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]  
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 4, 1), nn.Sigmoid()]
        self.conv_uncertainty = nn.Sequential(*conf_vector)

        reg_conf_vector = [nn.Conv2d(4, 1, 1), nn.Sigmoid()]  
        self.reg_conf = nn.Sequential(*reg_conf_vector) 

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_conf:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

        ###########################################################################
        bias_cls_pred = bias_init_with_prob(0.01)
        normal_init(self.conv_cls_refine_pred, std=0.01, bias=bias_cls_pred)
        normal_init(self.conv_reg_refine_pred, std=0.01)

        for m in self.conv_uncertainty:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        for modules in [
            self.uncertainty_cls_branch, # self.conv_cls_refine,
            self.uncertainty_reg_branch # self.conv_reg_refine
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)
        
        for m in self.conv_cls_refine:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        for m in self.conv_reg_refine:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        ###########################################################################

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(self.gfl_reg(reg_feat)).float()                         ##### why 17 x 4 not 16 x 4 ?????? --> 0 ~ 16 (discrete regression range) --> 17
        N, C, H, W = bbox_pred.size()                                             ##### N 68 H W (reg_max = 16)
        prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max+1, H, W), dim=2)    ##### N 4 17 H W --> Distribution of bbox regression --> softmax along discrete regression
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)                            ##### return topk value, indices(_)

        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                             dim=2)
        else:
            stat = prob_topk

        ###########################################################################
        uncertainty = self.conv_uncertainty(stat.reshape(N, -1, H, W))
        # uncertainty = torch.sigmoid(uncertainty)
        quality_score = self.reg_conf(uncertainty)
        ###########################################################################

        # quality_score = self.reg_conf(stat.reshape(N, -1, H, W))                  ##### I score
        cls_score = self.gfl_cls(cls_feat).sigmoid() * quality_score              ##### J = C x I

        ###########################################################################
        # cls refinement
        cls_refine = self.uncertainty_cls_branch(cls_feat, uncertainty)
        for cls_layer in self.conv_cls_refine:
            cls_refine = cls_layer(cls_refine)
        # cls_offset_score = self.conv_cls_refine_pred(cls_refine).sigmoid() * quality_score
        cls_offset_score = self.conv_cls_refine_pred(cls_refine).sigmoid()

        # bbox refinement 
        reg_refine = self.uncertainty_reg_branch(reg_feat, uncertainty)
        for reg_layer in self.conv_reg_refine:
            reg_refine = reg_layer(reg_refine)
        bbox_offset = self.conv_reg_refine_pred(reg_refine)
        ###########################################################################

        return cls_score, bbox_pred, cls_offset_score, bbox_offset                                           

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, 
                    cls_score, 
                    cls_offset, #####
                    bbox_pred, 
                    bbox_offset, #####
                    labels, label_weights, bbox_targets, 
                    refine_labels, #####
                    refine_labels_weights, #####
                    bbox_targets_for_refine, #####
                    bbox_offset_targets, #####
                    stride, 
                    num_total_refine_samples, #####
                    num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4). N is number of img.
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))

        ######################################################################
        cls_offset = cls_offset.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_offset = bbox_offset.permute(0, 2, 3, 1).reshape(-1, 4)
        ######################################################################

        bbox_targets = bbox_targets.reshape(-1, 4)                              ##### bbox target
        labels = labels.reshape(-1)                                             ##### cls target
        label_weights = label_weights.reshape(-1)                               

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        # pos_inds = ((labels >= 0)
        #             & (labels < bg_class_ind)).nonzero().squeeze(1)             #### positive indicies
        pos_inds = torch.nonzero(((labels >= 0) & (labels < bg_class_ind)), as_tuple=True)[0]
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)                ##### regression value through integral distribution
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]              ##### normalized bbox target 
            score[pos_inds] = bbox_overlaps(                                    ##### cls-quality joint target(IoU score)
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)

            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(pos_anchor_centers,
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,           ##### decoded bbox (corners --> bbox)
                pos_decode_bbox_targets,        ##### normalized decoded bbox target
                weight=weight_targets,
                # weight=None,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

             
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).cuda()
            loss_cls = cls_score.sum() * 0

        # cls (qfl) loss --> estimate IoU score of GT class (soft one-hot label)
        loss_cls = self.loss_cls(cls_score, (labels, score),
                                weight=label_weights,
                                avg_factor=num_total_samples)

        #######################################################################################
        # compute loss for refinement
        bbox_target_for_refine = bbox_targets_for_refine.reshape(-1, 4)
        bbox_off_target = bbox_offset_targets.reshape(-1, 4)
        refine_labels = refine_labels.reshape(-1)
        refine_labels_weights = refine_labels_weights.reshape(-1)
        bg_class_refine_ind = self.num_classes
        # pos_refine_inds = ((refine_labels >= 0)
        #             & (refine_labels < bg_class_refine_ind)).nonzero().squeeze(1)             #### positive indicies
        pos_refine_inds = torch.nonzero(((refine_labels >= 0) & (refine_labels < bg_class_refine_ind)), as_tuple=True)[0]
        refine_score = refine_labels_weights.new_zeros(refine_labels.shape)

        if len(pos_refine_inds) > 0:
            pos_refine_bbox_targets = bbox_target_for_refine[pos_refine_inds]
            # pos_bbox_offset_targets = bbox_off_target[pos_refine_inds] / stride[0]
            pos_bbox_offset_targets = bbox_off_target[pos_refine_inds] 
            pos_refine_bbox_pred = bbox_pred[pos_refine_inds]
            pos_bbox_offset = bbox_offset[pos_refine_inds]
            pos_refine_anchors = anchors[pos_refine_inds]
            pos_refine_anchor_centers = self.anchor_center(pos_refine_anchors) / stride[0]

            refine_weight_targets = cls_offset.detach()  ##### ??
            refine_weight_targets = refine_weight_targets.max(dim=1)[0][pos_refine_inds]      ##### ??

            pos_refine_bbox_pred_corners = self.integral(pos_refine_bbox_pred)                ##### regression value through integral distribution
            pos_refine_decode_bbox_pred = distance2bbox(pos_refine_anchor_centers,
                                                pos_refine_bbox_pred_corners)

            pos_refine_decode_bbox_targets = pos_refine_bbox_targets / stride[0]              ##### normalized bbox target 
            refine_score[pos_refine_inds] = bbox_overlaps(                                    ##### cls-quality joint target(IoU score)
                pos_refine_decode_bbox_pred.detach(),
                pos_refine_decode_bbox_targets,
                is_aligned=True)

            # regression offset loss
            # loss_bbox_offset = torch.abs(pos_bbox_offset - pos_bbox_offset_targets) * refine_score.unsqueeze(1).sum()
            # inverse_refine_score = torch.ones(refine_score[pos_refine_inds].size()).cuda()
            # inverse_refine_score = inverse_refine_score - refine_score[pos_refine_inds]
            loss_bbox_offset = self.loss_bbox_offset(pos_bbox_offset,
                                                     pos_bbox_offset_targets)
                                                    #  weight=inverse_refine_score.unsqueeze(1),
                                                    #  avg_factor=num_total_refine_samples)
                                                    # weight=refine_score[pos_refine_inds].unsqueeze(1),
                                                    # avg_factor=num_total_refine_samples)
            
        else :
            # loss_bbox_offset = bbox_offset.sum() * 0 
            loss_bbox_offset = bbox_offset.sum() * 0
            refine_weight_targets = torch.tensor(0).cuda()
            loss_cls_offset = cls_offset.sum() * 0

        if len(pos_refine_inds) > 1:
            # cls offset (qfl) loss --> estimate IoU score of GT class (soft one-hot label)
            loss_cls_offset = self.loss_cls_offset(cls_offset, (refine_labels, refine_score),
                                                    # weight=None,
                                                    # weight=refine_labels_weights,
                                                    # avg_factor=refine_valid_num + 1)
                                                    weight=refine_labels_weights,
                                                    avg_factor=num_total_refine_samples)
        else:
            loss_cls_offset = cls_offset.sum() * 0
        #######################################################################################

        return loss_cls, loss_bbox, loss_dfl, loss_cls_offset, loss_bbox_offset, weight_targets.sum(), refine_weight_targets.sum() 

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'cls_offsets', 'bbox_offsets'))
    def loss(self,
             cls_scores,
             bbox_preds,
             cls_offsets,    #####
             bbox_offsets,   #####
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            bbox_offsets (list[Tensor]) : Box offset prediction between Integral of bbox_preds 
                                         (distance from bbox edge to anchor center) and GT box    
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            bbox_preds, #####
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        # fpn scale level list
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg, 
         refine_labels_list, refine_label_weights_list, bbox_targets_for_refine_list, bbox_refine_targets_list, bbox_refine_weights_list, 
         num_total_refine_pos, num_total_refine_neg) = cls_reg_targets #####

        # print('pos :', num_total_pos, 'refine_pos :', num_total_refine_pos, 'neg :', num_total_neg, 'refine_neg :', num_total_refine_neg)

        #total positive number by classification label (if cls label == bg(80) : negative)
        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos).cuda()).item()
        num_total_samples = max(num_total_samples, 1.0)

        #total refine positive number by classification label (if refine cls label == bg(80) : negative)
        num_total_refine_samples = reduce_mean(
            torch.tensor(num_total_refine_pos).cuda()).item()
        num_total_refine_samples = max(num_total_refine_samples, 1.0)

        # calc loss_single for each fpn scale level
        losses_cls, losses_bbox, losses_dfl, losses_cls_offset, losses_bbox_offset,\
            avg_factor, refine_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                cls_offsets, ######
                bbox_preds,
                bbox_offsets, #######
                labels_list,
                label_weights_list,
                bbox_targets_list,
                refine_labels_list, ######
                refine_label_weights_list, ######
                bbox_targets_for_refine_list,
                bbox_refine_targets_list,    #######
                self.anchor_generator.strides,
                num_total_refine_samples= num_total_refine_samples, #####
                num_total_samples=num_total_samples)

        # avg_factor : sum(cls_score.max(dim=1))
        avg_factor = sum(avg_factor) # sum for list
        avg_factor = reduce_mean(avg_factor).item() # obtain avg_factor from multi-GPUs

        ###########################################################
        refine_avg_factor = sum(refine_avg_factor)
        refine_avg_factor = reduce_mean(refine_avg_factor).item()
        ###########################################################

        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

        ##################################################################################
        # losses_cls_offset = list(map(lambda x: x / refine_avg_factor, losses_cls_offset))
        # losses_bbox_offset = list(map(lambda x: x / refine_avg_factor, losses_bbox_offset))
        ##################################################################################

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_cls_offset=losses_cls_offset ,loss_bbox_offset=losses_bbox_offset)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'cls_offsets', 'bbox_offsets'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   cls_offsets,
                   bbox_offsets,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_offsets)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            cls_offset_list = [ ####
                cls_offsets[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_offset_list = [ ####
                bbox_offsets[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    cls_offset_list,
                                                    bbox_offset_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    cls_offset_list,
                                                    bbox_offset_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           cls_offsets,
                           bbox_offsets, #####
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                has shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for a single
                scale level with shape (4*(n+1), H, W), n is max value of
                integral set.
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): Bbox predictions in shape (N, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (N,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []

        # compute bboxes in a single image using multi scale level features
        for cls_score, bbox_pred, cls_offset, bbox_offset, stride, anchors in zip( #####
                cls_scores, bbox_preds, cls_offsets, bbox_offsets, self.anchor_generator.strides, #####
                mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0] #####
            scores_offset = cls_offset.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            bbox_offset = bbox_offset.permute(1, 2, 0).reshape(-1, 4)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # max_scores, _ = scores.max(dim=1)
                # max_scores, _ = scores_offset.max(dim=1)
                max_scores, _ = torch.sqrt(scores  * scores_offset).max(dim=1)

                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                bbox_offset = bbox_offset[topk_inds, :] #####
                scores = scores[topk_inds, :]
                scores_offset = scores_offset[topk_inds, :]

            bboxes = distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)

            #bbox inference with box offset
            #########################################################################
            bbox_std = bboxes.detach().new_tensor([0.5, 0.5, 0.5, 0.5])
            det_boxes_wh = bboxes[:, 2:4] - bboxes[:, 0:2]
            det_boxes_wh = torch.cat([det_boxes_wh, det_boxes_wh], dim=1)
            bboxes = bboxes + (bbox_offset * bbox_std * det_boxes_wh)
            #########################################################################

            # final_scores = scores
            # final_scores = scores_offset
            final_scores = torch.sqrt(scores * scores_offset)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(final_scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    @force_fp32(apply_to=('bbox_pred_list'))
    def get_targets(self,
                    anchor_list,              ### [5, 16400, 4] * img
                    valid_flag_list,          ### [5 ,16400]    * img 
                    bbox_pred_list, #####     ### [4, ]  * lvl   ### need to change to per image frame
                    gt_bboxes_list,           ### [6, 4] * img
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        
        #######################################################################
        # multi scale level of each img --> concat scale levels of each img
        mlvl_bboxes = []
        mimg_bboxes = []
        for img_id in range(len(img_metas)): ## per img
            bbox_preds = [
                bbox_pred_list[i][img_id].detach() for i in range(len(bbox_pred_list))
            ]
            for bbox_pred, stride in zip( ##### per level
                    bbox_preds, self.anchor_generator.strides):
                assert stride[0] == stride[1]
                bbox_pred = bbox_pred.permute(1, 2, 0)
                bbox_pred = self.integral(bbox_pred) * stride[0] ##### distribution --> corner prediction
                mimg_bboxes.append(bbox_pred)   ####  [ (15200, 68), (3800, 68),...(x5)]
            mlvl_bboxes.append(mimg_bboxes)     ####  [ [ (15200, 68), (3800, 68),...(x5)] * img  ]

        bbox_img_0 = torch.cat((mlvl_bboxes[0][0], mlvl_bboxes[0][1], mlvl_bboxes[0][2], mlvl_bboxes[0][3], mlvl_bboxes[0][4]), dim=0)
        bbox_img_1 = torch.cat((mlvl_bboxes[1][0], mlvl_bboxes[1][1], mlvl_bboxes[1][2], mlvl_bboxes[1][3], mlvl_bboxes[1][4]), dim=0)
        bbox_img_2 = torch.cat((mlvl_bboxes[2][0], mlvl_bboxes[2][1], mlvl_bboxes[2][2], mlvl_bboxes[2][3], mlvl_bboxes[2][4]), dim=0)
        bbox_img_3 = torch.cat((mlvl_bboxes[3][0], mlvl_bboxes[3][1], mlvl_bboxes[3][2], mlvl_bboxes[3][3], mlvl_bboxes[3][4]), dim=0)

        mlvl_bboxes = [bbox_img_0, bbox_img_1, bbox_img_2, bbox_img_3]  #### [(20236, 4) * img]
        assert num_level_anchors[0]+num_level_anchors[1]+num_level_anchors[2]+num_level_anchors[3]+num_level_anchors[4] == mlvl_bboxes[0].size(0)
        #######################################################################


        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        # compute targets for each image
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,         #### anchor_list - [ (20267, 4) * img ]
         all_bbox_weights, pos_inds_list, neg_inds_list, 
         all_refine_labels, all_refine_labels_weights, all_bbox_targets_for_refine, all_bbox_refine_targets, all_bbox_refine_weights, 
         refine_pos_inds, refine_neg_inds) = multi_apply(   #####
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             mlvl_bboxes, #####
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        #######################################################################
        num_total_refine_pos = sum([max(inds.numel(), 1) for inds in refine_pos_inds])
        num_total_refine_neg = sum([max(inds.numel(), 1) for inds in refine_neg_inds])
        refine_labels_list = images_to_levels(all_refine_labels,
                                             num_level_anchors)
        refine_labels_weights_list = images_to_levels(all_refine_labels_weights,
                                             num_level_anchors)                                             
        bbox_refine_targets_list = images_to_levels(all_bbox_refine_targets,
                                             num_level_anchors)
        bbox_refine_weights_list = images_to_levels(all_bbox_refine_weights,
                                             num_level_anchors)
        bbox_targets_for_refine_list = images_to_levels(all_bbox_targets_for_refine,
                                             num_level_anchors)
        #######################################################################                                             
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg,
                refine_labels_list, refine_labels_weights_list, bbox_targets_for_refine_list, 
                bbox_refine_targets_list, bbox_refine_weights_list, num_total_refine_pos, num_total_refine_neg) ######

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           bbox_preds, #####
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,     ##### flat_anchors : (22400, 4)
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]   #### (20805, 4)

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        
        else:
            # print('no assigned positive label')
            pass

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        else:
            # print('no assigned negative label')
            pass

        # Get cls-reg refine target
        ###########################################################################
        assert flat_anchors.size(0) == bbox_preds.size(0)
        assert flat_anchors.size(1) == bbox_preds.size(1)
        decode_bbox = bbox_preds.new_zeros(bbox_preds.size(), dtype=torch.float)
        decode_bbox = distance2bbox(self.anchor_center(flat_anchors), bbox_preds, max_shape=img_meta['img_shape']) 
        bbox_valid_flags = (decode_bbox[:, 0] < decode_bbox[:, 2]) & (decode_bbox[:, 1] < decode_bbox[:, 3]) & (decode_bbox[:, 0]>=0)\
                         & (decode_bbox[:, 1]>=0) & (decode_bbox[:, 2]>=0) & (decode_bbox[:, 3]>=0)
        
        refine_inside_flags = anchor_inside_flags(decode_bbox, bbox_valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not refine_inside_flags.any():
            return (None, ) * 7
        
        # assign gt and sample anchors
        bboxes = refine_inside_flags.new_zeros((refine_inside_flags.size(0), 4), dtype=torch.float)
        bboxes = decode_bbox[refine_inside_flags, :]   #### (20805, 4)

        num_level_bboxes_inside = self.get_num_level_anchors_inside(
            num_level_anchors, refine_inside_flags)
        bbox_assign_result = self.assigner.assign(bboxes, num_level_bboxes_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        bbox_sampling_result = self.sampler.sample(bbox_assign_result, bboxes,
                                              gt_bboxes)

        num_valid_bboxes = bboxes.shape[0]
        refine_bbox_targets = torch.zeros_like(bboxes)
        refine_bbox_weights = torch.zeros_like(bboxes)
        refine_labels = bboxes.new_full((num_valid_bboxes, ),
                                  self.num_classes,
                                  dtype=torch.long)
        refine_label_weights = bboxes.new_zeros(num_valid_bboxes, dtype=torch.float)
        bbox_offset_targets = torch.zeros_like(bboxes)
        bbox_offset_weights = torch.zeros_like(bboxes)

        refine_pos_inds = bbox_sampling_result.pos_inds
        refine_neg_inds = bbox_sampling_result.neg_inds
        # print('num_refine_pos :', len(refine_pos_inds), 'num_refine_neg :', len(refine_neg_inds))
        if len(refine_pos_inds) > 0:
            refine_pos_bbox_targets = bbox_sampling_result.pos_gt_bboxes
            refine_bbox_targets[refine_pos_inds, :] = refine_pos_bbox_targets
            refine_bbox_weights[refine_pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                refine_labels[refine_pos_inds] = 0
            else:
                refine_labels[refine_pos_inds] = gt_labels[bbox_sampling_result.pos_assigned_gt_inds]
                bbox_std = bboxes.new_tensor([0.5, 0.5, 0.5, 0.5])
                pre_boxes_wh = bboxes[refine_pos_inds, 2:4] - bboxes[refine_pos_inds, 0:2] ### [xmax - xmin, ymax - ymin] = [width, height]
                pre_boxes_wh = torch.cat([pre_boxes_wh, pre_boxes_wh], dim=1) ### [width, height, width, height]
                refine_bbox_offset_target = (refine_bbox_targets[refine_pos_inds] - bboxes[refine_pos_inds]) / (pre_boxes_wh * bbox_std)
                # refine_bbox_offset_target = (refine_bbox_targets[refine_pos_inds] - decode_inside_bbox_preds[refine_pos_inds])
                bbox_offset_targets[refine_pos_inds, :] = refine_bbox_offset_target
                bbox_offset_weights[refine_pos_inds, :] = 1.0
            if self.train_cfg.pos_weight <= 0:
                refine_label_weights[refine_pos_inds] = 1.0
            else:
                refine_label_weights[refine_pos_inds] = self.train_cfg.pos_weight
        
        else:
            # print('no assigned positive label')
            pass

        if len(refine_neg_inds) > 0:
            refine_label_weights[refine_neg_inds] = 1.0
        else:
            # print('no assigned negative label')
            pass

        refine_label_check = torch.nonzero(((refine_labels>=0)&(refine_labels<81)),as_tuple=True)[0]
        assert len(refine_label_check) > 0

        # map up to original set of anchors    
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0) # 20267
            ret_anchors = unmap(anchors, num_total_anchors, inside_flags)
            ret_labels  = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            ret_label_weights= unmap(label_weights, num_total_anchors, inside_flags, fill=0)
            ret_bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags, fill=0)
            ret_bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags, fill=0)

            ret_refine_labels         = unmap(refine_labels, num_total_anchors, refine_inside_flags, fill=self.num_classes)
            ret_refine_labels_weights = unmap(refine_label_weights, num_total_anchors, refine_inside_flags, fill=0)
            ret_bbox_refine_targets   = unmap(refine_bbox_targets, num_total_anchors, refine_inside_flags, fill=0)
            ret_bbox_offset_targets   = unmap(bbox_offset_targets, num_total_anchors, refine_inside_flags, fill=0)
            ret_bbox_offset_weights   = unmap(bbox_offset_weights, num_total_anchors, refine_inside_flags, fill=0)

        return (ret_anchors, ret_labels, ret_label_weights, ret_bbox_targets, ret_bbox_weights,
                pos_inds, neg_inds, 
                ret_refine_labels, ret_refine_labels_weights, ret_bbox_refine_targets, ret_bbox_offset_targets, ret_bbox_offset_weights, 
                refine_pos_inds, refine_neg_inds) #####

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()
    area2 = boxes2.area()

    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

class UncertaintyClsBranch(nn.Module):
    def __init__(self, in_channels, refeat_channels, uncertain=True):
        """
        :param in_channels:
        """
        super(UncertaintyClsBranch, self).__init__()
        self.uncertain = uncertain
        self.cur_point_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                refeat_channels,
                kernel_size=1),
            nn.InstanceNorm2d(refeat_channels),
            nn.ReLU())

        self.ltrb_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                refeat_channels * 4,
                kernel_size=1),
            nn.InstanceNorm2d(refeat_channels * 4),
            nn.ReLU())

        self.uncertainty_attentive = UncertaintyAttentive(Uncertain=self.uncertain, refeat_channel=refeat_channels)

        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(
                5 * refeat_channels,
                in_channels,
                kernel_size=1),
            nn.ReLU())

    def forward(self, feature, uncertainties):
        N, C, H, W = feature.shape

        fm_short = self.cur_point_conv(feature)
        feature = self.ltrb_conv(feature)
        ltrb_conv = self.uncertainty_attentive(feature, uncertainties)
        #ltrb_conv = ltrb_conv.permute(0, 3, 1, 2).reshape(N, -1, H, W)
        attentive_conv = torch.cat([ltrb_conv, fm_short], dim=1)
        attentive_conv = self.uncertainty_conv(attentive_conv)
        return attentive_conv

class UncertaintyAttentive(nn.Module):
    def __init__(self, Uncertain, refeat_channel):
        super(UncertaintyAttentive, self).__init__()
        self.uncertain = Uncertain
        self.refeat_channel = refeat_channel

    def forward(self, feature, uncertainties):
        N, C, H, W = feature.shape
        feature = feature.contiguous()
        uncertainties = uncertainties.contiguous()

        uncertainty_clone   = uncertainties.clone().detach()
        uncertainty_lefts   = uncertainty_clone[:, 0, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 0, :, :]
        uncertainty_tops    = uncertainty_clone[:, 1, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 1, :, :]
        uncertainty_rights  = uncertainty_clone[:, 2, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 2, :, :]
        uncertainty_bottoms = uncertainty_clone[:, 3, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 3, :, :]

        output = torch.zeros_like(feature)

        for index, feature in enumerate(feature) :
            output[index][0:self.refeat_channel, :, :] = feature[0:self.refeat_channel, :, :] * uncertainty_lefts[index]
            output[index][self.refeat_channel : self.refeat_channel * 2, :, :]    = feature[self.refeat_channel : self.refeat_channel * 2, :, :] * uncertainty_tops[index]
            output[index][self.refeat_channel * 2: self.refeat_channel * 3, :, :] = feature[self.refeat_channel * 2: self.refeat_channel * 3, :, :] * uncertainty_rights[index]
            output[index][self.refeat_channel * 3: self.refeat_channel * 4, :, :] = feature[self.refeat_channel * 3: self.refeat_channel * 4, :, :] * uncertainty_bottoms[index]

        # output = torch.cat([ltrb_l, ltrb_t, ltrb_r, ltrb_b], dim=1)
        
        return output

class UncertaintyRegBranch(nn.Module):
    def __init__(self, in_channels, refeat_channels, uncertain=True):
        """
        :param in_channels:
        """
        super(UncertaintyRegBranch, self).__init__()
        self.uncertain = uncertain
        self.cur_point_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                refeat_channels,
                kernel_size=1),
            nn.InstanceNorm2d(refeat_channels),
            nn.ReLU())

        self.ltrb_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                refeat_channels * 4,
                kernel_size=1),
            nn.InstanceNorm2d(refeat_channels * 4),
            nn.ReLU())

        self.uncertainty_attentive = UncertaintyAttentive_reg(Uncertain=self.uncertain, refeat_channel=refeat_channels)

        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(
                5 * refeat_channels,
                in_channels,
                kernel_size=1),
            nn.ReLU())

    def forward(self, feature, uncertainties):
        N, C, H, W = feature.shape

        fm_short = self.cur_point_conv(feature)
        feature = self.ltrb_conv(feature)
        ltrb_conv = self.uncertainty_attentive(feature, uncertainties)
        #ltrb_conv = ltrb_conv.permute(0, 3, 1, 2).reshape(N, -1, H, W)
        attentive_conv = torch.cat([ltrb_conv, fm_short], dim=1)
        attentive_conv = self.uncertainty_conv(attentive_conv)
        return attentive_conv

class UncertaintyAttentive_reg(nn.Module):
    def __init__(self, Uncertain, refeat_channel):
        super(UncertaintyAttentive_reg, self).__init__()
        self.uncertain = Uncertain
        self.refeat_channel = refeat_channel

    def forward(self, feature, uncertainties):
        N, C, H, W = feature.shape
        feature = feature.contiguous()
        uncertainties = uncertainties.contiguous()

        uncertainty_clone   = uncertainties.clone().detach()
        uncertainty_lefts   = uncertainty_clone[:, 0, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 0, :, :]
        uncertainty_tops    = uncertainty_clone[:, 1, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 1, :, :]
        uncertainty_rights  = uncertainty_clone[:, 2, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 2, :, :]
        uncertainty_bottoms = uncertainty_clone[:, 3, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 3, :, :]

        output = torch.zeros_like(feature)

        for index, feature in enumerate(feature) :
            output[index][0:self.refeat_channel, :, :] = feature[0:self.refeat_channel, :, :] * uncertainty_lefts[index]
            output[index][self.refeat_channel : self.refeat_channel * 2, :, :]    = feature[self.refeat_channel : self.refeat_channel * 2, :, :] * uncertainty_tops[index]
            output[index][self.refeat_channel * 2: self.refeat_channel * 3, :, :] = feature[self.refeat_channel * 2: self.refeat_channel * 3, :, :] * uncertainty_rights[index]
            output[index][self.refeat_channel * 3: self.refeat_channel * 4, :, :] = feature[self.refeat_channel * 3: self.refeat_channel * 4, :, :] * uncertainty_bottoms[index]

        # output = torch.cat([ltrb_l, ltrb_t, ltrb_r, ltrb_b], dim=1)
        
        return output

