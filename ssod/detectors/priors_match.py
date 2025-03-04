from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models import BaseDetector
from mmdet.models.utils import select_single_mlvl, images_to_levels, rename_loss_dict, reweight_loss_dict, multi_apply
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import bbox2roi, bbox_project, get_box_tensor
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, InstanceList

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

        self.num_classes = self.student.roi_head.bbox_head.num_classes
        self.with_background = self.semi_train_cfg.get('with_background', True)
        if self.with_background:
            self.prob_dim = self.num_classes + 1
        else:
            self.prob_dim = self.num_classes

        semi_weight_cfg = semi_train_cfg.get(
            'semi_weight_cfg', dict(type='PseudoLabelWeight'))
        semi_weight_cfg['num_classes'] = self.prob_dim
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

        rpn_pseudo_instances, roi_pseudo_instances = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'],
            multi_batch_data_samples['unsup_teacher'],
            multi_batch_data_samples['unsup_student'])
        losses.update(**self.loss_by_pseudo_instances(
            multi_batch_inputs['unsup_student'],
            rpn_pseudo_instances,
            roi_pseudo_instances))
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

    @torch.no_grad()
    def get_pseudo_instances(self,
                             batch_inputs: Tensor,
                             batch_data_samples_teacher: SampleList,
                             batch_data_samples_student: SampleList):
        self.teacher.eval()
        assert self.teacher.with_bbox, 'Bbox head must be implemented.'

        x = self.teacher.extract_feat(batch_inputs)
        batch_input_shape_teacher = batch_data_samples_teacher[0].batch_input_shape
        batch_input_shape_student = batch_data_samples_student[0].batch_input_shape
        feat_scales = [batch_input_shape_teacher[0] // feat.size(2) for feat in x]

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

        feature_map_project_args = dict(
            feat_scales=feat_scales,
            homography_matrix_list=homography_matrix_list,
            ori_shapes=ori_shapes,
            img_shapes=img_shapes,
            batch_input_shape=batch_input_shape_student)
        rpn_results_list, rpn_pseudo_instances = self.get_rpn_pseudo_instances(
            x, batch_data_samples_teacher, feature_map_project_args)

        roi_pseudo_instances = self.get_roi_pseudo_instances(
            x, rpn_results_list, feature_map_project_args)

        return rpn_pseudo_instances, roi_pseudo_instances

    @torch.no_grad()
    def get_rpn_pseudo_instances(self,
                                 x: Tuple[Tensor],
                                 batch_data_samples: SampleList,
                                 feature_map_project_args):
        rpn_head = self.teacher.rpn_head
        cls_scores, bbox_preds = rpn_head(x)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples]
            rpn_results_list = rpn_head.predict_by_feat(
                cls_scores, bbox_preds, batch_img_metas=batch_img_metas, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples]

        if rpn_head.use_sigmoid_cls:
            mlvl_scores = [cls_score.sigmoid() for cls_score in cls_scores]
        else:
            mlvl_scores = [cls_score.softmax(1)[:] for cls_score in cls_scores]

        mlvl_scores = mlvl_feature_map_project(
            mlvl_scores, **feature_map_project_args)

        return rpn_results_list, {'mlvl_scores': mlvl_scores}

    @torch.no_grad()
    def get_roi_pseudo_instances(self,
                                 x: Tuple[Tensor],
                                 rpn_results_list: InstanceList,
                                 project_args: dict):
        roi_head = self.teacher.roi_head

        proposals_list = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals_list)
        bbox_results = roi_head._bbox_forward(x, rois)

        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals_list)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)
        bbox_preds = bbox_preds.split(num_proposals_per_img, 0)

        pseudo_instances = {
            'proposals': [],
            'scores': [],
            'labels': [],
            'bboxes': [],
        }

        for proposals, cls_score, bbox_pred, homography_matrix, ori_shape, img_shape in zip(
                proposals_list, cls_scores, bbox_preds,
                project_args['homography_matrix_list'],
                project_args['ori_shapes'],
                project_args['img_shapes']):
            num_proposals = proposals.size(0)
            if num_proposals == 0:
                pseudo_instances['proposals'].append(proposals)
                pseudo_instances['scores'].append(cls_score)
                pseudo_instances['labels'].append(cls_score.max(dim=-1)[1])
                pseudo_instances['bboxes'].append(bbox_pred)
                continue

            scores = torch.softmax(cls_score, dim=-1)
            _, labels = torch.max(scores, dim=-1)

            pos_inds = torch.nonzero((labels >= 0) & (labels < self.num_classes), as_tuple=True)[0]
            bboxes = roi_head.bbox_head.bbox_coder.decode(proposals, bbox_pred, max_shape=ori_shape)
            bboxes = bbox_project(get_box_tensor(bboxes).view(-1, 4), homography_matrix, img_shape)
            if not roi_head.bbox_head.reg_class_agnostic:
                pos_bboxes = bboxes.view(num_proposals, -1, 4)[pos_inds, labels[pos_inds]]
                bboxes = pos_bboxes.new_zeros((num_proposals, 4))
                bboxes[pos_inds, :] = pos_bboxes
            ws = torch.clamp(bboxes[..., 2] - bboxes[..., 0], min=0)
            hs = torch.clamp(bboxes[..., 3] - bboxes[..., 1], min=0)
            areas_bbox = ws * hs

            projected_proposals = bbox_project(proposals, homography_matrix, img_shape)
            ws = torch.clamp(projected_proposals[..., 2] - projected_proposals[..., 0], min=0)
            hs = torch.clamp(projected_proposals[..., 3] - projected_proposals[..., 1], min=0)
            areas_proposal = ws * hs

            valid_flags = areas_proposal > 0
            valid_flags[pos_inds] = valid_flags[pos_inds] & (areas_bbox[pos_inds] > 0)

            pseudo_instances['proposals'].append(projected_proposals[valid_flags])
            pseudo_instances['scores'].append(scores[valid_flags])
            pseudo_instances['labels'].append(labels[valid_flags])
            pseudo_instances['bboxes'].append(bboxes[valid_flags])

        return pseudo_instances

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 rpn_pseudo_instances,
                                 roi_pseudo_instances):
        x = self.student.extract_feat(batch_inputs)

        losses = dict()
        losses.update(**self.rpn_loss_by_pseudo_instances(
            x, rpn_pseudo_instances))
        losses.update(**self.roi_loss_by_pseudo_instances(
            x, roi_pseudo_instances))
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.0)
        return rename_loss_dict('unsup_', reweight_loss_dict(losses, unsup_weight))

    def rpn_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                     rpn_pseudo_instances) -> dict:
        rpn_head = self.student.rpn_head
        cls_scores, bbox_preds = rpn_head(x)

        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        priors = rpn_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        num_lvl_priors = [anchors.size(0) for anchors in priors]
        priors = torch.cat(priors)

        num_imgs = x[0].size(0)
        num_expected_pos = int(rpn_head.sampler.num * rpn_head.sampler.pos_fraction)
        num_expected_neg = rpn_head.sampler.num - num_expected_pos
        mlvl_scores = rpn_pseudo_instances['mlvl_scores']
        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        avg_factor = 0.0
        for img_id in range(num_imgs):
            scores_list = select_single_mlvl(mlvl_scores, img_id)
            scores = torch.cat([s.permute(1, 2, 0).reshape(-1, rpn_head.cls_out_channels)
                                for s in scores_list])
            if not rpn_head.use_sigmoid_cls:
                scores = scores[:, :-1]
            scores = scores.squeeze(1)

            pos_inds = torch.nonzero(scores > self.semi_train_cfg.get('rpn_pos_thr', 0.9), as_tuple=True)[0]
            neg_inds = torch.nonzero(scores < self.semi_train_cfg.get('rpn_neg_thr', 0.1), as_tuple=True)[0]
            if pos_inds.numel() > num_expected_pos:
                perm = torch.randperm(pos_inds.numel())[:num_expected_pos].to(pos_inds.device)
                pos_inds = pos_inds[perm]
            if neg_inds.numel() > num_expected_neg:
                perm = torch.randperm(neg_inds.numel())[:num_expected_neg].to(neg_inds.device)
                neg_inds = neg_inds[perm]

            labels = priors.new_full((scores.size(0),),
                                     rpn_head.num_classes,
                                     dtype=torch.long)
            label_weights = priors.new_zeros((scores.size(0),), dtype=torch.float)
            bbox_targets = torch.zeros_like(priors)
            bbox_weights = torch.zeros_like(priors)

            if len(pos_inds) > 0:
                labels[pos_inds] = 0
                label_weights[pos_inds] = 1.0 if rpn_head.train_cfg['pos_weight'] <= 0 \
                    else rpn_head.train_cfg['pos_weight']
            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0
            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_bbox_targets.append(bbox_targets)
            all_bbox_weights.append(bbox_weights)
            avg_factor += len(pos_inds) + len(neg_inds)
        all_anchors = [priors for _ in range(num_imgs)]
        anchor_list = images_to_levels(all_anchors, num_lvl_priors)
        labels_list = images_to_levels(all_labels, num_lvl_priors)
        label_weights_list = images_to_levels(all_label_weights, num_lvl_priors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_lvl_priors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_lvl_priors)

        losses_cls, losses_bbox = multi_apply(
            rpn_head.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor)

        return dict(loss_rpn_cls=losses_cls)

    def roi_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                     roi_pseudo_instances) -> dict:
        roi_head = self.student.roi_head
        rois = bbox2roi(roi_pseudo_instances['proposals'])
        bbox_results = roi_head._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        labels_list = []
        label_weights_list = []
        bbox_targets_list = []
        bbox_weights_list = []

        num_classes = roi_head.bbox_head.num_classes
        num_expected_pos = int(roi_head.bbox_sampler.num * roi_head.bbox_sampler.pos_fraction)
        num_expected_neg = roi_head.bbox_sampler.num - num_expected_pos
        for proposals, scores, labels, bboxes in zip(
                roi_pseudo_instances['proposals'],
                roi_pseudo_instances['scores'],
                roi_pseudo_instances['labels'],
                roi_pseudo_instances['bboxes']):
            num_samples = proposals.size(0)
            if num_samples == 0:
                continue
            reg_dim = bboxes.size(-1) if roi_head.bbox_head.reg_decoded_bbox \
                else roi_head.bbox_head.bbox_coder.encode_size

            pos_inds = torch.nonzero((labels >= 0) & (labels < num_classes), as_tuple=True)[0]
            neg_inds = torch.nonzero(labels == num_classes, as_tuple=True)[0]
            if pos_inds.numel() > num_expected_pos:
                perm = torch.randperm(pos_inds.numel())[:num_expected_pos].to(pos_inds.device)
                pos_inds = pos_inds[perm]
            if neg_inds.numel() > num_expected_neg:
                perm = torch.randperm(neg_inds.numel())[:num_expected_neg].to(neg_inds.device)
                neg_inds = neg_inds[perm]

            label_weights = cls_score.new_zeros((num_samples,))
            bbox_targets = cls_score.new_zeros((num_samples, reg_dim))
            bbox_weights = cls_score.new_zeros((num_samples, reg_dim))

            if len(pos_inds) > 0:
                label_weights[pos_inds] = 1.0 if roi_head.train_cfg['pos_weight'] <= 0 \
                    else roi_head.train_cfg['pos_weight']

                pos_bboxes = bboxes[pos_inds]
                if not roi_head.bbox_head.reg_decoded_bbox:
                    pos_bbox_targets = roi_head.bbox_head.bbox_coder.encode(
                        proposals[pos_inds], pos_bboxes)
                else:
                    pos_bbox_targets = pos_bboxes
                bbox_targets[pos_inds, :] = pos_bbox_targets
                bbox_weights[pos_inds, :] = 1.0
            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0

            labels_list.append(labels)
            label_weights_list.append(label_weights)
            bbox_targets_list.append(bbox_targets)
            bbox_weights_list.append(bbox_weights)
        scores = torch.cat(roi_pseudo_instances['scores'], 0)
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        train_inds = torch.nonzero(label_weights > 0, as_tuple=True)[0]
        scores = scores[train_inds]
        labels = labels[train_inds]
        label_weights = label_weights[train_inds]
        bbox_targets = bbox_targets[train_inds]
        bbox_weights = bbox_weights[train_inds]

        if not self.with_background:
            pos_inds = torch.nonzero((labels >= 0) & (labels < num_classes), as_tuple=True)[0]
            if len(pos_inds) > 1:
                label_weights[pos_inds] *= self.reweighter.compute_weights(
                    scores[pos_inds, :self.prob_dim])
        else:
            label_weights *= self.reweighter.compute_weights(
                scores[:, :self.prob_dim])

        return roi_head.bbox_head.loss(
            cls_score[train_inds],
            bbox_pred[train_inds],
            rois[train_inds],
            labels,
            label_weights,
            bbox_targets,
            bbox_weights)

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
