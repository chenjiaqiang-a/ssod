from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes
from mmdet.models.task_modules import BaseAssigner, AssignResult
from mmdet.utils import ConfigType


@TASK_UTILS.register_module()
class AdaptiveSampleAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth with adaptive sample
    label assignment, much like the dynamic soft label assignment.

    Args:
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        center_distance_weight (float): weight of the center distance.
            Defaults to 0.001.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
    """

    def __init__(
            self,
            topk: int = 13,
            iou_weight: float = 3.0,
            center_distance_weight: float = 0.001,
            iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
    ) -> None:
        self.topk = topk
        self.iou_weight = iou_weight
        self.center_distance_weight = center_distance_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs):
        """Assign gt to priors.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors are
                anchors, shape (n, 4). The ``strides`` of anchors will
                be used to rescale distances. The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.
        Returns:
            obj:`AssignResult`: The assigned result.
        """
        INF = 100000000
        EPS = 1.0e-7

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors

        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)

        prior_center = (priors[:, [2, 3]] + priors[:, [0, 1]]) / 2.0
        if isinstance(gt_bboxes, BaseBoxes):
            is_in_gts = gt_bboxes.find_inside_points(prior_center)
        else:
            lt_ = prior_center[:, None] - gt_bboxes[:, :2]
            rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

            deltas = torch.cat([lt_, rb_], dim=-1)
            is_in_gts = deltas.min(dim=-1).values > 0
        valid_mask = is_in_gts.sum(dim=1) > 0
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,),
                                                      -1,
                                                      dtype=torch.long)
            assign_result = AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            return assign_result

        if isinstance(gt_bboxes, BaseBoxes):
            gt_center = gt_bboxes.centers
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
        prior_center = prior_center[valid_mask]
        strides = priors[valid_mask][:, 4]
        distance = (prior_center[:, None, :] - gt_center[None, :, :]).pow(2).sum(-1).sqrt() / strides[:, None]
        dist_cost = torch.pow(10, distance) * self.center_distance_weight

        pairwise_ious = self.iou_calculator(valid_decoded_bbox,
                                            gt_bboxes).detach()
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        gt_onehot_label = F.one_hot(
            gt_labels.to(torch.int64),
            pred_scores.shape[-1]
        ).float().unsqueeze(0).repeat(num_valid, 1, 1)
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        pt = (1 - valid_pred_scores) * gt_onehot_label + valid_pred_scores * (1 - gt_onehot_label)
        alpha, gamma = 0.25, 2
        focal_weight = (alpha * gt_onehot_label + (1 - alpha) * (1 - gt_onehot_label)) * pt.pow(gamma)
        cls_cost = F.binary_cross_entropy(
            valid_pred_scores, gt_onehot_label, reduction='none') * focal_weight
        cls_cost = cls_cost.sum(dim=-1)

        cost_matrix = cls_cost + iou_cost + dist_cost

        matched_gt_inds, matched_pred_ious = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask)
        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious

        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
                           num_gt: int,
                           valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets. Same as SimOTA.

        Args:
            cost (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.

        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_gt_inds, matched_pred_ious
