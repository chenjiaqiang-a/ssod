# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from mmengine import MessageHub
from mmdet.models import BaseDetector
from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean


@MODELS.register_module()
class DenseTeacher(BaseDetector):
    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.student = MODELS.build(detector)
        self.teacher = MODELS.build(detector)
        self.semi_train_cfg = semi_train_cfg
        self.semi_test_cfg = semi_test_cfg
        if self.semi_train_cfg.get('freeze_teacher', True) is True:
            self.freeze(self.teacher)

        cls_weight = self.semi_train_cfg.get('cls_weight', 4.0)
        reg_weight = self.semi_train_cfg.get('reg_weight', 1.0)
        centerness_weight = self.semi_train_cfg.get('centerness_weight', 1.0)
        self.loss_cls = MODELS.build(dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=cls_weight))
        self.loss_bbox = MODELS.build(dict(
            type='GIoULoss',
            loss_weight=reg_weight))
        self.loss_centerness = MODELS.build(dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=centerness_weight))


    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()
        losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))

        message_hub = MessageHub.get_current_instance()
        cur_iter = message_hub.get_info('iter') + 1
        burn_in_steps = self.semi_train_cfg.get('burn_in_steps', 0)
        if cur_iter <= burn_in_steps:
            return losses

        unsup_losses = self.loss_by_pseudo_instances(
            multi_batch_inputs['unsup_teacher'],
            multi_batch_data_samples['unsup_teacher'],
            multi_batch_inputs['unsup_student'],
            multi_batch_data_samples['unsup_student'])

        target = burn_in_steps + self.semi_train_cfg.get('warmup_steps', 3000)
        suppress = self.semi_train_cfg.get('suppress', 'linear')
        unsup_factor = 1.0
        if suppress == 'linear':
            if cur_iter <= target:
                unsup_factor *= (cur_iter - burn_in_steps) / (target - burn_in_steps)
        elif suppress == 'step':
            if cur_iter <= target:
                unsup_factor *= 0.25
        elif suppress == 'exp':
            if cur_iter <= target:
                unsup_factor *= np.exp((cur_iter - target) / 1000)
        else:
            raise ValueError(
                f"The `suppress` can only be one of ['linear', 'exp', 'step'], but get {suppress}")
        losses.update(**reweight_loss_dict(unsup_losses, unsup_factor))

        return losses

    def loss_by_gt_instances(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and ground-truth data
        samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """

        losses = self.student.loss(batch_inputs, batch_data_samples)
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.)
        return rename_loss_dict('sup_', reweight_loss_dict(losses, sup_weight))

    def loss_by_pseudo_instances(self,
                                 batch_inputs_teacher: Tensor,
                                 batch_data_samples_teacher: SampleList,
                                 batch_inputs_student: Tensor,
                                 batch_data_samples_student: SampleList) -> dict:
        x = self.student.extract_feat(batch_inputs_student)

        bbox_head = self.student.bbox_head
        num_images = len(batch_data_samples_student)
        num_classes = bbox_head.cls_out_channels
        cls_scores, bbox_preds, centernesses = bbox_head(x)

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_images, -1, num_classes)
            for cls_score in cls_scores
        ], dim=1).view(-1, num_classes)
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for bbox_pred in bbox_preds
        ], dim=1).view(-1, 4)
        flatten_centernesses = torch.cat([
            centerness.permute(0, 2, 3, 1).reshape(num_images, -1, 1)
            for centerness in centernesses
        ], dim=1).view(-1, 1)
        flatten_points = torch.cat(
            [torch.cat(all_level_points) for _ in range(num_images)])

        cls_targets, bbox_targets, centernesses_targets, mask, avg_factor = \
            self.get_teacher_output_and_ragion_selection(
                batch_inputs_teacher, batch_data_samples_teacher)
        avg_factor = max(reduce_mean(avg_factor), 1.0)

        losses = dict()
        losses['loss_cls'] = self.loss_cls(
            flatten_cls_scores, cls_targets, avg_factor=avg_factor)

        if torch.any(mask):
            pos_decoded_bbox_preds = bbox_head.bbox_coder.decode(
                flatten_points[mask], flatten_bbox_preds[mask])
            pos_decoded_target_preds = bbox_head.bbox_coder.decode(
                flatten_points[mask], bbox_targets[mask])
            losses['loss_bbox'] = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds)
            losses['loss_centerness'] = self.loss_centerness(
                flatten_centernesses[mask],
                centernesses_targets[mask])
        else:
            losses['loss_bbox'] = flatten_bbox_preds[mask].sum()
            losses['loss_centerness'] = flatten_centernesses[mask].sum()

        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1.) if torch.any(mask) else 0.
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))

    @torch.no_grad()
    def get_teacher_output_and_ragion_selection(
            self, batch_inputs: Tensor, batch_data_samples: SampleList):
        self.teacher.eval()
        x = self.teacher.extract_feat(batch_inputs)

        bbox_head = self.teacher.bbox_head
        num_images = len(batch_data_samples)
        num_classes = bbox_head.cls_out_channels
        cls_scores, bbox_preds, centernesses = bbox_head(x)
        cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_images, -1, num_classes)
            for cls_score in cls_scores
        ], dim=1).view(-1, num_classes)
        strides = bbox_head.strides if bbox_head.norm_on_bbox else [1.] * len(bbox_preds)
        bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_images, -1, 4) / stride
            for bbox_pred, stride in zip(bbox_preds, strides)
        ], dim=1).view(-1, 4)
        centernesses = torch.cat([
            centerness.permute(0, 2, 3, 1).reshape(num_images, -1, 1).sigmoid()
            for centerness in centernesses
        ], dim=1).view(-1, 1)

        # Region Selection
        ratio = self.semi_train_cfg.get('distill_ratio', 0.01)
        count_num = int(cls_scores.size(0) * ratio)
        scores = cls_scores.sigmoid()
        max_scores = torch.max(scores, dim=1)[0]
        sorted_scores, sorted_inds = torch.topk(max_scores, scores.size(0))
        mask = torch.zeros_like(max_scores)
        mask[sorted_inds[:count_num]] = 1.0
        avg_factor = sorted_scores[:count_num].sum()
        b_mask = mask > 0

        scores[mask <= 0] = 0.0
        return scores, bbox_preds, centernesses, b_mask, avg_factor

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if self.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.student(
                batch_inputs, batch_data_samples, mode='predict')

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> SampleList:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        if self.semi_test_cfg.get('forward_on', 'teacher') == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='tensor')
        else:
            return self.student(
                batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'teacher':
            return self.teacher.extract_feat(batch_inputs)
        else:
            return self.student.extract_feat(batch_inputs)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Add teacher and student prefixes to model parameter names."""
        if not any([
                'student' in key or 'teacher' in key
                for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())
            state_dict.update({'teacher.' + k: state_dict[k] for k in keys})
            state_dict.update({'student.' + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
