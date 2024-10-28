import copy
import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import deform_conv2d
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmengine.model import bias_init_with_prob, normal_init
from mmdet.registry import MODELS
from mmdet.structures.bbox import distance2bbox
from mmdet.models.dense_heads import ATSSHead
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import multi_apply, images_to_levels, unmap, \
    filter_scores_and_topk
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean


@MODELS.register_module()
class FAM3DHead(ATSSHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_dcn: int = 0,
                 anchor_type: str = 'anchor_free',
                 use_atan: bool = False,
                 offset_channel_shrink: int = 4,
                 **kwargs) -> None:
        assert anchor_type in ['anchor_based', 'anchor_free']
        self.num_dcn = num_dcn
        self.anchor_type = anchor_type
        self.use_atan = use_atan
        self.offset_channel_shrink = offset_channel_shrink
        super(FAM3DHead, self).__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = dict(type='DCNv2', deform_groups=4) if i < self.num_dcn else self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.fam3d_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.fam3d_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            3,
            padding=1)
        self.pyramid_mapping_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs * 2,
                      self.feat_channels // self.offset_channel_shrink,
                      1),
            nn.ReLU(inplace=True))
        self.spacial_offset_module = nn.Conv2d(
            self.feat_channels // self.offset_channel_shrink * 3,
            4 * 2,
            3,
            padding=1)
        self.scale_offset_module = nn.Conv2d(
            self.feat_channels // self.offset_channel_shrink * 3,
            4,
            3,
            padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.pyramid_mapping_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.fam3d_cls, std=0.01, bias=bias_cls)
        normal_init(self.fam3d_reg, std=0.01)
        normal_init(self.spacial_offset_module, std=0.001)
        normal_init(self.scale_offset_module, std=0.001)

    @staticmethod
    def anchor_center(anchors: Tensor) -> Tensor:
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    batch_img_metas: List[dict],
                    device: Union[torch.device, str] = 'cuda') -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (torch.device or str): Device for returned tensors.
                Defaults to cuda.

        Returns:
            tuple:

            - anchor_list (list[list[Tensor]]): Anchors of each image.
            - valid_flag_list (list[list[Tensor]]): Valid flags of each
              image.
        """
        num_imgs = len(batch_img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        # concat stride to anchors
        multi_level_strides = [
            torch.full_like(anchors[:, 0:1], stride[0])
            for anchors, stride in zip(multi_level_anchors, self.prior_generator.strides)
        ]
        multi_level_anchors = [
            torch.cat([anchors, strides], dim=-1)
            for anchors, strides in zip(multi_level_anchors, multi_level_strides)
        ]
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Decoded box for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_anchors * 4. In [tl_x, tl_y, br_x, br_y] format.
        """
        offset_feats = []
        cls_scores = []
        bbox_preds = []
        for idx, (feat, scale, stride) in enumerate(zip(feats, self.scales, self.prior_generator.strides)):
            b, c, h, w = feat.shape
            anchor = self.prior_generator.single_level_grid_priors(
                (h, w), idx, device=feat.device)
            anchor = torch.cat([anchor for _ in range(b)])

            cls_feat = feat
            reg_feat = feat
            inter_feat = []
            for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
                cls_feat = cls_conv(cls_feat)
                inter_feat.append(cls_feat)
                reg_feat = reg_conv(reg_feat)
                inter_feat.append(reg_feat)
            inter_feat = torch.cat(inter_feat, dim=1)

            # cls prediction
            cls_score = self.fam3d_cls(cls_feat).sigmoid()

            # reg prediction before alignment
            if self.anchor_type == 'anchor_free':
                reg_dist = scale(self.fam3d_reg(reg_feat).exp()).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = distance2bbox(self.anchor_center(anchor) / stride[0], reg_dist) \
                    .reshape(b, h, w, 4).permute(0, 3, 1, 2)
            elif self.anchor_type == 'anchor_based':
                reg_dist = scale(self.fam3d_reg(reg_feat)).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = self.bbox_coder.decode(anchor, reg_dist) \
                               .reshape(b, h, w, 4).permute(0, 3, 1, 2) / stride[0]
            else:
                raise NotImplementedError(
                    f'Unknown anchor type: {self.anchor_type}.'
                    f'Please use `anchor_free` or `anchor_based`.')

            cls_scores.append(cls_score)
            bbox_preds.append(reg_bbox)
            offset_feats.append(self.pyramid_mapping_module(inter_feat))
        bbox_preds_new = []
        for idx in range(len(feats)):
            b, c, h, w = offset_feats[idx].shape
            if idx > 0:
                lower = F.interpolate(
                    offset_feats[idx - 1],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False)
            else:
                lower = offset_feats[idx]

            if idx < len(feats) - 1:
                upper = F.interpolate(
                    offset_feats[idx + 1],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False)
            else:
                upper = offset_feats[idx]
            offset_feat = torch.cat([lower, offset_feats[idx], upper], dim=1)

            scale_offset = self.scale_offset_module(offset_feat)
            if self.use_atan:
                scale_offset = scale_offset.atan() / (math.pi / 2)
                if idx == 0:
                    scale_offset = scale_offset.clamp(min=0)
                if idx == len(feats) - 1:
                    scale_offset = scale_offset.clamp(max=0)
            else:
                if idx == 0:
                    scale_offset = scale_offset.clamp(min=0)
                else:
                    scale_offset = scale_offset.clamp(min=-1)
                if idx == len(feats) - 1:
                    scale_offset = scale_offset.clamp(max=0)
                else:
                    scale_offset = scale_offset.clamp(max=1)
            scale_offset_current = 1 - scale_offset.abs()
            scale_offset_top = scale_offset.clamp(min=0)
            scale_offset_bottom = (-scale_offset).clamp(min=0)
            bbox_pred = bbox_preds[idx] * scale_offset_current
            if idx > 0:
                bbox_pred = bbox_pred + scale_offset_bottom * \
                            F.interpolate(
                                bbox_preds[idx - 1],
                                size=(h, w),
                                mode='bilinear',
                                align_corners=False) * \
                            self.prior_generator.strides[idx - 1][0] / self.prior_generator.strides[idx][0]
            if idx < len(feats) - 1:
                bbox_pred = bbox_pred + scale_offset_top * \
                            F.interpolate(
                                bbox_preds[idx + 1],
                                size=(h, w),
                                mode='bilinear',
                                align_corners=False) * \
                            self.prior_generator.strides[idx + 1][0] / self.prior_generator.strides[idx][0]

            spacial_offset = self.spacial_offset_module(offset_feat)
            bbox_pred = self.deform_sampling(
                bbox_pred.contiguous(), spacial_offset.contiguous())
            bbox_preds_new.append(bbox_pred)
        return tuple(cls_scores), tuple(bbox_preds_new)

    @staticmethod
    def deform_sampling(feat, offset):
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for for feature sampliing
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        y = deform_conv2d(feat, offset, weight, 1, 0, 1, c, c)
        return y

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, assign_metrics: Tensor,
                            stride: Tuple[int, int]) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (Tuple[int, int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        assign_metrics = assign_metrics.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        # regression loss
        if len(pos_inds) > 0:
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            pos_bbox_weight = assign_metrics[pos_inds]

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, assign_metrics.sum(), pos_bbox_weight.sum()

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ], dim=1)
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) * stride[0]
            for bbox_pred, stride in zip(bbox_preds, self.prior_generator.strides)
        ], dim=1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        labels_list, label_weights_list, bbox_targets_list, \
            alignment_metrics_list = cls_reg_targets

        losses_cls, losses_bbox, \
            cls_avg_factors, bbox_avg_factors = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            alignment_metrics_list,
            self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_targets(self,
                    cls_scores: List[List[Tensor]],
                    bbox_preds: List[List[Tensor]],
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (list[list[Tensor]]): Classification predictions of
                images, a 3D-Tensor with shape [num_imgs, num_priors,
                num_classes].
            bbox_preds (list[list[Tensor]]): Decoded bboxes predictions of one
                image, a 3D-Tensor with shape [num_imgs, num_priors, 4] in
                [tl_x, tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: a tuple containing learning targets.

                - anchors_list (list[list[Tensor]]): Anchors of each level.
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - norm_alignment_metrics_list (list[Tensor]): Normalized
                  alignment metrics of each level.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        # anchor_list: list(b * [-1, 4])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        all_labels, all_label_weights, all_bbox_targets, \
            all_assign_metrics = multi_apply(
            self._get_targets_single,
            cls_scores,
            bbox_preds,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)

        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        assign_metrics_list = images_to_levels(all_assign_metrics, num_level_anchors)

        return labels_list, label_weights_list, \
            bbox_targets_list, assign_metrics_list

    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (Tensor): Box scores for each image.
            bbox_preds (Tensor): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
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
                norm_alignment_metrics (Tensor): Normalized alignment metrics
                    of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors[:, :4], valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])

        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        pred_instances = InstanceData(
            priors=anchors,
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :])
        assign_result = self.assigner.assign(
            pred_instances, gt_instances, gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        bbox_targets = torch.zeros_like(anchors[:, :4])
        assign_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']

            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_ind in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_ind]
            assign_metrics[gt_class_inds] = assign_result.max_overlaps[gt_class_inds]

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(
                label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(
                bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(
                assign_metrics, num_total_anchors, inside_flags)

        return labels, label_weights, bbox_targets, \
            assign_metrics

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: Optional[ConfigDict] = None,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (:obj:`ConfigDict`, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, priors, stride in zip(
                cls_score_list, bbox_pred_list, mlvl_priors,
                self.prior_generator.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride[0]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)

            score_thr = cfg.get('score_thr', 0)
            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bboxes = filtered_results['bbox_pred']

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
