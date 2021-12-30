import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms, multiclass_nms_with_uncertainty,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead


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


@HEADS.register_module()
class UncertaintyGuideHead(AnchorHead):
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
        # self.repeated_conv = 3
        # self.repeated_conv_channels = 256
        # self.dcn_conv = False
        ###########################################################################

        if add_mean:
            self.total_dim += 1
        print('total dim = ', self.total_dim * 4)

        super(UncertaintyGuideHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral = Integral(self.reg_max) 
        self.loss_dfl = build_loss(loss_dfl)
        # self.loss_bbox_offset = build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.loss_uncertainty = build_loss(dict(type='UncertaintyLoss', loss_weight=0.05))

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
        # self.conv_cls_refine = nn.ModuleList()
        # self.conv_reg_refine = nn.ModuleList()
        # for i in range(self.repeated_conv):
        #     chn = self.feat_channels if i == 0 else self.repeated_conv_channels
        #     if self.dcn_conv and i == self.repeated_conv - 1:
        #         conv_cfg = dict(type='DCNv2')
        #     else:
        #         conv_cfg = self.conv_cfg
        #     self.conv_cls_refine.append(
        #         ConvModule(
        #             chn,
        #             self.repeated_conv_channels,
        #             1,
        #             stride=1,
        #             # padding=1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=self.norm_cfg,
        #             bias='auto'))
        # for i in range(self.repeated_conv):
        #     chn = self.feat_channels if i == 0 else self.repeated_conv_channels
        #     if self.dcn_conv and i == self.repeated_conv - 1:
        #         conv_cfg = dict(type='DCNv2')
        #     else:
        #         conv_cfg = self.conv_cfg
        #     self.conv_reg_refine.append(
        #         ConvModule(
        #             chn,
        #             self.repeated_conv_channels,
        #             3,
        #             stride=1,
        #             padding=1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=self.norm_cfg,
        #             bias='auto'))
        # self.conv_cls_refine_pred = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        # self.conv_reg_refine_pred = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_uncertainty = nn.Conv2d(self.feat_channels, 4, 3 , padding=1)
        # self.add_module("uncertainty_cls_branch", UncertaintyClsBranch(self.feat_channels, self.feat_channels, uncertain=False))        
        # self.add_module("uncertainty_reg_branch", UncertaintyRegBranch(self.feat_channels, 64, uncertain=False))
        # self.add_module("uncertainty_reg_branch", UncertaintyBranch(self.feat_channels, 64, uncertain=False))
        ###########################################################################

        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(                                      ##### cls prediction conv layer
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(                                      ##### reg prediction conv layer
            # self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
            self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList(                                   ##### stride scale for fpn
            [Scale(1.0) for _ in self.anchor_generator.strides])

        # original
        # conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]  ### 4 direction x topk(4)+mean : input
        conf_vector = [nn.Conv2d(4, self.reg_channels, 1)]  ### 4 direction x topk(4)+mean : input
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()] ##### Distribution-Guided Quality Predictor (DGQP) : 1x1 --> ReLu --> 1x1 --> Sigmoid
        self.reg_conf = nn.Sequential(*conf_vector) 

        
        
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
        # bias_cls_pred = bias_init_with_prob(0.01)
        # normal_init(self.conv_cls_refine_pred, std=0.01, bias=bias_cls_pred)
        # normal_init(self.conv_reg_refine_pred, std=0.01)
        normal_init(self.conv_uncertainty, std=0.01)

        # for m in self.conv_uncertainty:
        #     if isinstance(m, nn.Conv2d):
        #         normal_init(m, std=0.01)

        # for modules in [
        #     # self.uncertainty_cls_branch, # self.conv_cls_refine,
        #     self.uncertainty_reg_branch # self.conv_reg_refine
        # ]:
        #     for layer in modules.modules():
        #         if isinstance(layer, nn.Conv2d):
        #             torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
        #             torch.nn.init.constant_(layer.bias, 0)
        #         if isinstance(layer, nn.GroupNorm):
        #             torch.nn.init.constant_(layer.weight, 1)
        #             torch.nn.init.constant_(layer.bias, 0)
        
        # for m in self.conv_cls_refine:
        #     if isinstance(m.conv, nn.Conv2d):
        #         normal_init(m.conv, std=0.01)
        # for m in self.conv_reg_refine:
        #     if isinstance(m.conv, nn.Conv2d):
        #         normal_init(m.conv, std=0.01)
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
        bbox_pred = F.relu(bbox_pred)
        # N, C, H, W = bbox_pred.size()                                             ##### N 68 H W (reg_max = 16)
        # prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max+1, H, W), dim=2)    ##### N 4 17 H W --> Distribution of bbox regression --> softmax along discrete regression
        # prob_topk, _ = prob.topk(self.reg_topk, dim=2)                            ##### return topk value, indices(_)

        # if self.add_mean:
        #     stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
        #                      dim=2)
        # else:
        #     stat = prob_topk

        ###########################################################################
        uncertainty = self.conv_uncertainty(reg_feat)
        # uncertainty = torch.sigmoid(uncertainty)

        certainty = uncertainty.new_ones(uncertainty.size(), dtype=torch.float).cuda()
        certainty = certainty - uncertainty

        # uncertainty = self.conv_uncertainty(stat.reshape(N, -1, H, W))
        # uncertainty = torch.sigmoid(uncertainty)
        # quality_score = self.reg_conf(stat.reshape(N, -1, H, W))

        quality_score = self.reg_conf(certainty)
        ###########################################################################

        # quality_score = self.reg_conf(stat.reshape(N, -1, H, W))                  ##### I score
        cls_score = self.gfl_cls(cls_feat).sigmoid() * quality_score              ##### J = C x I

        ###########################################################################
        # cls refinement
        # cls_refine = self.uncertainty_cls_branch(cls_feat, uncertainty)
        # for cls_layer in self.conv_cls_refine:
        #     cls_refine = cls_layer(cls_refine)
        # cls_offset = self.conv_cls_refine_pred(cls_refine)

        # bbox refinement 
        # reg_refine = self.uncertainty_reg_branch(reg_feat, quality_score)
        # for reg_layer in self.conv_reg_refine:
        #     reg_refine = reg_layer(reg_refine)
        # bbox_offset = self.conv_reg_refine_pred(reg_refine)
        ###########################################################################

        return cls_score, bbox_pred, uncertainty, quality_score                                           

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

    def loss_single(self, anchors, cls_score, bbox_pred, 
                    # bbox_offset, ##### 
                    uncertainty,
                    labels, label_weights, bbox_targets, 
                    stride, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
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
        # bbox_pred = bbox_pred.permute(0, 2, 3,
        #                               1).reshape(-1, 4 * (self.reg_max + 1))

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        ######################################################################
        # bbox_offset = bbox_offset.permute(0, 2, 3, 1).reshape(-1, 4)
        uncertainty = uncertainty.permute(0, 2, 3, 1).reshape(-1, 4).sigmoid()
        ######################################################################

        bbox_targets = bbox_targets.reshape(-1, 4)                              ##### bbox target
        labels = labels.reshape(-1)                                             ##### cls target
        label_weights = label_weights.reshape(-1)                               ##### ATSS function ?? 

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)             #### positive indicies
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            # pos_bbox_offset = bbox_offset[pos_inds] #####
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            # pos_bbox_pred_corners = self.integral(pos_bbox_pred)                ##### regression value through integral distribution
            # pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
            #                                      pos_bbox_pred_corners)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers, pos_bbox_pred)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]              ##### normalized bbox target 
            score[pos_inds] = bbox_overlaps(                                    ##### cls-quality joint target(IoU score)
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)

            # pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            # target_corners = bbox2distance(pos_anchor_centers,
            #                                pos_decode_bbox_targets,
            #                                self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,           ##### decoded bbox (corners --> bbox)
                pos_decode_bbox_targets,        ##### normalized decoded bbox target
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            # loss_dfl = self.loss_dfl(
            #     pred_corners,
            #     target_corners,
            #     weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
            #     avg_factor=4.0)

            # pos_scaled_achors = anchors[pos_inds]
            # pos_scaled_achor_centers = self.anchor_center(pos_scaled_achors)
            # pos_scaled_bbox_pred_corners = pos_bbox_pred_corners * stride[0]
            # pos_scaled_bbox_pred =  distance2bbox(pos_scaled_achor_centers, pos_scaled_bbox_pred_corners)
           
            # # get bbox_offset targets
            # ##############################################################
            # bbox_std = pos_scaled_bbox_pred.new_tensor([0.5 , 0.5, 0.5, 0.5])
            # pre_boxes_wh = pos_scaled_bbox_pred[:, 2:4] - pos_scaled_bbox_pred[:, 0:2]
            # pre_boxes_wh = torch.cat([pre_boxes_wh, pre_boxes_wh], dim=1)
            # bbox_off_target = (pos_bbox_targets - pos_scaled_bbox_pred.detach()) / (pre_boxes_wh.detach() * bbox_std.detach())
            # ##############################################################

            # bbox offset loss
            ##############################################################
            # loss_bbox_offset = torch.abs(pos_bbox_offset - bbox_off_target) * score.unsqueeze(1).sum()
            # loss_bbox_offset = self.loss_bbox_offset(pos_bbox_offset,
            #                                          bbox_off_target,
            #                                          weight=score[pos_inds].unsqueeze(1),
            #                                         #  avg_factor=len(pos_inds))
            #                                         avg_factor=num_total_samples)
            #                                         # avg_factor=1.0)
            # pos_ious = (score > 0).nonzero().reshape(-1)
            # num_pos_ious = num_pos_ious = len(pos_ious)
            # loss_bbox_offset /= num_pos_ious + 4
            ##############################################################

            # uncertainty loss
            pos_uncertainty = uncertainty[pos_inds]
            target_corner = bbox2distance(self.anchor_center(pos_anchors) / stride[0], pos_bbox_targets / stride[0])

            # flatten_uncertainty = pos_uncertainty.reshape(-1)
            # flatten_pos_bbox_pred = pos_bbox_pred.reshape(-1)
            # flatten_target_corner = target_corner.reshape(-1)

            loss_uncertainty = self.loss_uncertainty(pos_uncertainty,
                                                     pos_bbox_pred,
                                                     target_corner,
                                                     score[pos_inds],
                                                     score[pos_inds])

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).cuda()
            # loss_bbox_offset = bbox_offset.sum() * 0 ####
            loss_uncertainty = uncertainty.sum() * 0

        # cls (qfl) loss --> estimate IoU score of GT class (soft one-hot label)
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        # return loss_cls, loss_bbox, loss_dfl, loss_uncertainty, weight_targets.sum()
        return loss_cls, loss_bbox, loss_uncertainty, weight_targets.sum(), score[pos_inds].sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_offsets', 'uncertainty', 'quality'))
    def loss(self,
             cls_scores,
             bbox_preds,
            #  bbox_offsets,   #####
             uncertainty, #####
             quality,
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
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos).cuda()).item()
        num_total_samples = max(num_total_samples, 1.0)

        # losses_cls, losses_bbox, losses_dfl, loss_uncertainty,\
        losses_cls, losses_bbox, loss_uncertainty,\
            avg_factor, uncertainty_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                # bbox_offsets, #######
                uncertainty, #####
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        uncertainty_avg_factor = sum(uncertainty_avg_factor)
        uncertainty_avg_factor = reduce_mean(uncertainty_avg_factor).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        # losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        # losses_bbox_offset = list(map(lambda x: x / avg_factor, losses_bbox_offset))
        # losses_uncertainty = list(map(lambda x: x / uncertainty_avg_factor, loss_uncertainty))
        return dict(
            # loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_uncertainty=loss_uncertainty)
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_uncertainty=loss_uncertainty)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'uncertainty', 'quality'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                #    bbox_offsets,
                   uncertainty,
                   qualty,
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
        assert len(cls_scores) == len(bbox_preds) == len(uncertainty)
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
            # bbox_offset_list = [ ####
            #     bbox_offsets[i][img_id].detach() for i in range(num_levels)
            # ]
            uncertainty_list = [ ####
                uncertainty[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    uncertainty_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    uncertainty_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           uncertainties,
                        #    bbox_offsets, #####
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
        mlvl_uncertainties = []

        for cls_score, bbox_pred, uncertainty, stride, anchors in zip( #####
                cls_scores, bbox_preds, uncertainties, self.anchor_generator.strides, #####
                mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            # bbox_pred = bbox_pred.permute(1, 2, 0)
            # bbox_pred = self.integral(bbox_pred) * stride[0] #####
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride[0]
            # bbox_offset = bbox_offset.permute(1, 2, 0).reshape(-1,4)
            uncertainty = uncertainty.permute(1, 2, 0).reshape(-1, 4).sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                # bbox_offset = bbox_offset[topk_inds, :] #####
                scores = scores[topk_inds, :]
                uncertainty = uncertainty[topk_inds, :]

            bboxes = distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)

            #bbox inference with box offset
            #########################################################################
            # bbox_std = bboxes.detach().new_tensor([0.5 , 0.5, 0.5, 0.5])
            # det_boxes_wh = bboxes[:, 2:4] - bboxes[:, 0:2]
            # det_boxes_wh = torch.cat([det_boxes_wh, det_boxes_wh], dim=1)
            # bboxes = bboxes + (bbox_offset * bbox_std * det_boxes_wh)
            #########################################################################
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_uncertainties.append(uncertainty)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_uncertainties = torch.cat(mlvl_uncertainties)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_with_uncertainty(mlvl_bboxes, mlvl_uncertainties, mlvl_scores,
            # det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
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
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
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
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
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
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

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
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside


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

class UncertaintyBranch(nn.Module):
    def __init__(self, in_channels, refeat_channels, uncertain=True):
        """
        :param in_channels:
        """
        super(UncertaintyBranch, self).__init__()
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
                # refeat_channels * 4,
                refeat_channels,
                kernel_size=1),
            # nn.InstanceNorm2d(refeat_channels * 4),
            nn.InstanceNorm2d(refeat_channels),
            nn.ReLU())

        self.uncertainty_attentive = UncertaintyAttention(Uncertain=self.uncertain, refeat_channel=refeat_channels)

        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(
                # 5 * refeat_channels,
                2 * refeat_channels,
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

class UncertaintyAttention(nn.Module):
    def __init__(self, Uncertain, refeat_channel):
        super(UncertaintyAttention, self).__init__()
        self.uncertain = Uncertain
        self.refeat_channel = refeat_channel

    def forward(self, feature, uncertainties):
        N, C, H, W = feature.shape
        feature = feature.contiguous()
        uncertainties = uncertainties.contiguous()

        uncertainty_clone   = uncertainties.clone().detach()
        # uncertainty_lefts   = uncertainty_clone[:, 0, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 0, :, :]
        # uncertainty_tops    = uncertainty_clone[:, 1, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 1, :, :]
        # uncertainty_rights  = uncertainty_clone[:, 2, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 2, :, :]
        # uncertainty_bottoms = uncertainty_clone[:, 3, :, :] if self.uncertain else 1.0 - uncertainty_clone[:, 3, :, :]

        output = torch.zeros_like(feature)

        for index, feature in enumerate(feature) :
            # output[index][0:self.refeat_channel, :, :] = feature[0:self.refeat_channel, :, :] * uncertainty_lefts[index]
            # output[index][self.refeat_channel : self.refeat_channel * 2, :, :]    = feature[self.refeat_channel : self.refeat_channel * 2, :, :] * uncertainty_tops[index]
            # output[index][self.refeat_channel * 2: self.refeat_channel * 3, :, :] = feature[self.refeat_channel * 2: self.refeat_channel * 3, :, :] * uncertainty_rights[index]
            # output[index][self.refeat_channel * 3: self.refeat_channel * 4, :, :] = feature[self.refeat_channel * 3: self.refeat_channel * 4, :, :] * uncertainty_bottoms[index]
            output[index][0:self.refeat_channel, :, :] = feature[0:self.refeat_channel, :, :] * uncertainty_clone[index]

        # output = torch.cat([ltrb_l, ltrb_t, ltrb_r, ltrb_b], dim=1)
        
        return output