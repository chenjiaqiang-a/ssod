from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmengine import MessageHub
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models import BaseDetector
from mmdet.models.utils import reweight_loss_dict, rename_loss_dict, images_to_levels
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import get_box_tensor, bbox_project
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean

from ..utils.feature_map import mlvl_feature_map_project


@MODELS.register_module()
class PriorsMatch(BaseDetector):
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

        self.num_classes = self.student.bbox_head.num_classes
        semi_weight_cfg = semi_train_cfg.get(
            'semi_weight_cfg', dict(type='PseudoLabelWeight'))
        semi_weight_cfg['num_classes'] = self.num_classes
        self.reweighter = TASK_UTILS.build(semi_weight_cfg)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        losses = dict()
        losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))

        message_hub = MessageHub.get_current_instance()
        cur_iter = message_hub.get_info('iter') + 1

        if cur_iter > self.semi_train_cfg.get('burn_in_steps', -1):
            pseudo_instances = self.get_pseudo_instances(
                multi_batch_inputs['unsup_teacher'],
                multi_batch_data_samples['unsup_teacher'],
                multi_batch_data_samples['unsup_student'])
            losses.update(**self.loss_by_pseudo_instances(
                multi_batch_inputs['unsup_student'], pseudo_instances))
        return losses

    def loss_by_gt_instances(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList) -> dict:
        losses = self.student.loss(batch_inputs, batch_data_samples)
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.0)
        return rename_loss_dict('sup_', reweight_loss_dict(losses, sup_weight))

    @torch.no_grad()
    def get_pseudo_instances(self, batch_inputs: Tensor,
                             batch_data_samples_teacher: SampleList,
                             batch_data_samples_student: SampleList):
        self.teacher.eval()
        x = self.teacher.extract_feat(batch_inputs)
        bbox_head = self.teacher.bbox_head
        num_imgs = len(batch_data_samples_teacher)
        bg_class_ind = self.num_classes

        cls_scores, bbox_preds, centernesses = bbox_head(x)
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)

        mlvl_scores = []
        mlvl_labels = []
        for cls_score in cls_scores:
            if bbox_head.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(1)[:, :-1]
            score_thr = self.semi_train_cfg.get('bg_score_thr', 0.05)
            max_scores, labels = torch.max(scores, dim=1, keepdim=True)
            labels[max_scores < score_thr] = bg_class_ind
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        homography_matrix_list = []
        ori_shapes = []
        img_shapes = []
        for data_samples_teacher, data_samples_student in zip(
                batch_data_samples_teacher, batch_data_samples_student):
            teacher_matrix = torch.from_numpy(data_samples_teacher.homography_matrix).to(
                self.data_preprocessor.device)
            student_matrix = torch.from_numpy(data_samples_student.homography_matrix).to(
                self.data_preprocessor.device)
            homography_matrix = student_matrix @ teacher_matrix.inverse()
            homography_matrix_list.append(homography_matrix)
            ori_shapes.append(data_samples_teacher.img_shape)
            img_shapes.append(data_samples_student.img_shape)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        num_level_points = [points.size(0) for points in all_level_points]
        points = torch.cat(all_level_points)
        strides = bbox_head.strides if bbox_head.norm_on_bbox else [1.] * len(bbox_preds)
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) / stride
            for bbox_pred, stride in zip(bbox_preds, strides)
        ], dim=1)

        all_bboxes = []
        for img_id, (homography_matrix, ori_shape, img_shape) in enumerate(
                zip(homography_matrix_list, ori_shapes, img_shapes)):
            bboxes = bbox_head.bbox_coder.decode(
                points, flatten_bbox_preds[img_id], max_shape=ori_shape)
            bboxes = bbox_project(get_box_tensor(bboxes), homography_matrix, img_shape)
            all_bboxes.append(bboxes)
        mlvl_bboxes = images_to_levels(all_bboxes, num_level_points)
        mlvl_bboxes = [bboxes.view(num_imgs, h, w, 4).permute(0, 3, 1, 2)
                       for bboxes, (h, w) in zip(mlvl_bboxes, featmap_sizes)]

        mlvl_labels = mlvl_feature_map_project([labels + 1 for labels in mlvl_labels],
                                               homography_matrix_list, ori_shapes, img_shapes)
        mlvl_scores = mlvl_feature_map_project(mlvl_scores, homography_matrix_list, ori_shapes, img_shapes)
        mlvl_bboxes = mlvl_feature_map_project(mlvl_bboxes, homography_matrix_list, ori_shapes, img_shapes)

        flatten_labels = torch.cat([labels.permute(0, 2, 3, 1).reshape(-1) - 1
                                    for labels in mlvl_labels])
        flatten_labels[flatten_labels < 0] = bg_class_ind
        flatten_scores = torch.cat([scores.permute(0, 2, 3, 1).reshape(-1, bbox_head.cls_out_channels)
                                    for scores in mlvl_scores])
        flatten_bboxes = torch.cat([bboxes.permute(0, 2, 3, 1).reshape(-1, 4)
                                    for bboxes in mlvl_bboxes])
        flatten_points = torch.cat([p.repeat(num_imgs, 1)
                                    for p in all_level_points])

        ws = torch.clamp(flatten_bboxes[:, 2] - flatten_bboxes[:, 0], min=0)
        hs = torch.clamp(flatten_bboxes[:, 3] - flatten_bboxes[:, 1], min=0)
        areas = ws * hs

        valid_flags = torch.full_like(flatten_labels, True, dtype=torch.bool)
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        valid_flags[pos_inds] = valid_flags[pos_inds] & (areas[pos_inds] > 0)
        if self.semi_train_cfg.get('use_valid_sample_selection', True):
            pos_points = flatten_points[pos_inds]
            pos_bboxes = flatten_bboxes[pos_inds]
            point_in_bbox = \
                (pos_points[:, 0] >= pos_bboxes[:, 0]) & (pos_points[:, 0] <= pos_bboxes[:, 2]) & \
                (pos_points[:, 1] >= pos_bboxes[:, 1]) & (pos_points[:, 1] <= pos_bboxes[:, 3])
            valid_flags[pos_inds] = valid_flags[pos_inds] & point_in_bbox

        return {
            'scores': flatten_scores[valid_flags],
            'labels': flatten_labels[valid_flags],
            'bboxes': flatten_bboxes[valid_flags],
            'valid_flags': valid_flags,
        }

    def loss_by_pseudo_instances(self, batch_inputs: Tensor, pseudo_instances: dict):
        x = self.student.extract_feat(batch_inputs)
        bbox_head = self.student.bbox_head

        cls_scores, bbox_preds, _ = bbox_head(x)
        assert len(cls_scores) == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        bg_class_ind = self.num_classes

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        valid_labels = pseudo_instances['labels']
        valid_scores = pseudo_instances['scores']
        valid_bboxes = pseudo_instances['bboxes']
        valid_flags = pseudo_instances['valid_flags']
        valid_points = torch.cat([p.repeat(num_imgs, 1)
                                  for p in all_level_points])[valid_flags]

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = torch.cat([cls_score.permute(0, 2, 3, 1).reshape(-1, bbox_head.cls_out_channels)
                                        for cls_score in cls_scores])[valid_flags]
        flatten_bbox_preds = torch.cat([bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                                        for bbox_pred in bbox_preds])[valid_flags]
        weights = flatten_cls_scores.new_ones((flatten_cls_scores.size(0),))
        pos_inds = ((valid_labels >= 0) & (valid_labels < bg_class_ind)).nonzero().reshape(-1)

        if len(pos_inds) > 1:
            weights[pos_inds] *= self.reweighter.compute_weights(valid_scores[pos_inds])

        avg_factor = weights[pos_inds].sum()
        avg_factor = max(reduce_mean(avg_factor), 1.0)
        if valid_flags.any():
            loss_cls = bbox_head.loss_cls(
                flatten_cls_scores, valid_labels, weight=weights, avg_factor=avg_factor)
        else:
            loss_cls = flatten_cls_scores.sum()

        if len(pos_inds) > 0:
            pos_points = valid_points[pos_inds]
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_decoded_bbox_preds = bbox_head.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            loss_bbox = bbox_head.loss_bbox(
                pos_decoded_bbox_preds,
                valid_bboxes[pos_inds],
                avg_factor=avg_factor)
        else:
            loss_bbox = flatten_bbox_preds[pos_inds].sum()

        losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.0)
        return rename_loss_dict('unsup_', reweight_loss_dict(losses, unsup_weight))

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with
        post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

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
                 batch_data_samples: OptSampleList = None) -> tuple:
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
