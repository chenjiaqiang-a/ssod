import copy
from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch import Tensor
from mmengine import MessageHub
from mmengine.fileio import load
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models import BaseDetector
from mmdet.models.utils import rename_loss_dict, reweight_loss_dict, unpack_gt_instances
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import bbox2roi, bbox_project, get_box_tensor, bbox_overlaps
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, InstanceList

from ..utils.dist_utils import concat_all_gather


@MODELS.register_module()
class LabelMatch(BaseDetector):
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
        self.boxes_per_labeled_image, self.labeled_cls_dist = self._get_labeled_data_info(
            self.semi_train_cfg['labeled_ann_file'], self.num_classes)

        self.max_cls_per_img = 20
        self.queue_num = self.semi_train_cfg.get('queue_num', 128 * 100)
        self.queue_len = self.queue_num * self.max_cls_per_img
        self.register_buffer("score_queue", torch.zeros(self.queue_len, self.num_classes))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.reliable_percent = self.semi_train_cfg.get('reliable_percent', 0.2)
        self.queue_is_ready = False

        cls_pseudo_thr = self.semi_train_cfg.get('cls_pseudo_thr', 0.9)
        self.cls_thr_reliable = [cls_pseudo_thr] * self.num_classes
        self.cls_thr_uncertain = [cls_pseudo_thr] * self.num_classes

        self.use_rplm = self.semi_train_cfg.get('use_rplm', True)
        self.rplm_score = self.semi_train_cfg.get('rplm_score', 0.8)
        self.rplm_iou = self.semi_train_cfg.get('rplm_iou', 0.8)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def _get_labeled_data_info(ann_file: str, num_classes: int):
        info = load(ann_file)
        image_num = len(info['images'])
        box_num = len(info['annotations'])
        assert image_num > 0 and box_num > 0

        cls_num = [0] * num_classes
        for ann in info['annotations']:
            class_id = int(ann['category_id']) - 1
            cls_num[class_id] += 1
        labeled_cls_dist = np.array([c / box_num for c in cls_num])
        boxes_per_labeled_image = box_num / image_num
        print_log(f'labeled class distribution: {labeled_cls_dist}', logger='current')
        print_log(f'there are {boxes_per_labeled_image} boxes per labeled image', logger='current')

        return boxes_per_labeled_image, labeled_cls_dist

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        losses = dict()
        losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))

        batch_pseudo_data_samples, batch_info = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'],
            multi_batch_data_samples['unsup_teacher'])
        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(
            batch_pseudo_data_samples,
            multi_batch_data_samples['unsup_student'])

        losses.update(**self.loss_by_pseudo_instances(
            multi_batch_inputs['unsup_student'],
            multi_batch_data_samples['unsup_student'],
            batch_info))

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
    def get_pseudo_instances(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList):
        self.teacher.eval()
        assert self.teacher.with_bbox, 'Bbox head must be implemented.'
        x = self.teacher.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.teacher.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.teacher.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=False)

        batch_info = {
            'feat': x,
            'img_shape': [],
            'homography_matrix': [],
            'metainfo': []
        }

        self._update_act()
        score_list = [[] for _ in range(self.num_classes)]
        for data_samples, results in zip(batch_data_samples, results_list):
            res_r, res_u = [], []
            for cls in range(self.num_classes):
                cls_res = results[results.labels == cls]
                if len(cls_res) >= self.max_cls_per_img:
                    score = cls_res.scores[:self.max_cls_per_img]
                else:
                    score = cls_res.scores.new_zeros((self.max_cls_per_img,))
                    score[:len(cls_res)] = cls_res.scores
                score_list[cls].append(score)
                flag_r = cls_res.scores >= self.cls_thr_reliable[cls]
                flag_u = (cls_res.scores >= self.cls_thr_uncertain[cls]) & (~flag_r)
                res_r.append(cls_res[flag_r])
                res_u.append(cls_res[flag_u])
            res_r = InstanceData.cat(res_r)
            res_u = InstanceData.cat(res_u)

            data_samples.gt_instances = res_r
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
            data_samples.uncertain_instances = res_u
            data_samples.uncertain_instances.bboxes = bbox_project(
                data_samples.uncertain_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)

            batch_info['img_shape'].append(data_samples.img_shape)
            batch_info['homography_matrix'].append(
                torch.from_numpy(data_samples.homography_matrix).to(
                    self.data_preprocessor.device))
            batch_info['metainfo'].append(data_samples.metainfo)

        score_list = [torch.cat(s) for s in score_list]
        scores = torch.stack(score_list, dim=1)
        self._dequeue_and_enqueue(scores)

        return batch_data_samples, batch_info

    def project_pseudo_instances(self, batch_pseudo_data_samples: SampleList,
                                 batch_data_samples: SampleList) -> SampleList:
        wh_thr = self.semi_train_cfg.get('min_pseudo_bbox_wh', (1e-2, 1e-2))
        for pseudo_data_samples, data_samples in zip(batch_pseudo_data_samples,
                                                     batch_data_samples):
            data_samples.gt_instances = copy.deepcopy(
                pseudo_data_samples.gt_instances)
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device), data_samples.img_shape)
            bboxes = data_samples.gt_instances.bboxes
            if bboxes.shape[0] > 0:
                w = bboxes[:, 2] - bboxes[:, 0]
                h = bboxes[:, 3] - bboxes[:, 1]
                data_samples.gt_instances = data_samples.gt_instances[
                    (w > wh_thr[0]) & (h > wh_thr[1])]

            data_samples.uncertain_instances = copy.deepcopy(
                pseudo_data_samples.uncertain_instances)
            data_samples.uncertain_instances.bboxes = bbox_project(
                data_samples.uncertain_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device), data_samples.img_shape)
            bboxes = data_samples.uncertain_instances.bboxes
            if bboxes.shape[0] > 0:
                w = bboxes[:, 2] - bboxes[:, 0]
                h = bboxes[:, 3] - bboxes[:, 1]
                data_samples.uncertain_instances = data_samples.uncertain_instances[
                    (w > wh_thr[0]) & (h > wh_thr[1])]
        return batch_data_samples

    def _dequeue_and_enqueue(self, scores: Tensor):
        """update score queue"""
        if dist.is_initialized():
            scores = concat_all_gather(scores)
        batch_size = scores.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0
        self.score_queue[ptr:ptr + batch_size, :] = scores
        if (not self.queue_is_ready) and ptr + batch_size >= self.queue_len:
            self.queue_is_ready = True
        ptr = (ptr + batch_size) % self.queue_len
        self.queue_ptr[0] = ptr

    def _update_act(self):
        if not self.queue_is_ready:
            return

        message_hub = MessageHub.get_current_instance()
        cur_iter = message_hub.get_info('iter') + 1

        if cur_iter % self.semi_train_cfg.get('act_update_interval', 1) == 0:
            reliable_boxes_per_img = self.boxes_per_labeled_image * self.reliable_percent
            reliable_location = self.queue_num * reliable_boxes_per_img * self.labeled_cls_dist
            uncertain_location = self.queue_num * self.boxes_per_labeled_image * self.labeled_cls_dist
            score, _ = self.score_queue.sort(dim=0, descending=True)
            for cls in range(self.num_classes):
                self.cls_thr_reliable[cls] = max(0.05, score[int(reliable_location[cls]), cls].item())
                self.cls_thr_uncertain[cls] = max(0.05, score[int(uncertain_location[cls]), cls].item())
            if cur_iter % 1000 == 0:
                print_log(f'Update reliable score thr: {self.cls_thr_reliable}')
                print_log(f'Update uncertain score thr: {self.cls_thr_uncertain}')

    def loss_by_pseudo_instances(self, batch_inputs: Tensor,
                                 batch_data_samples: SampleList, batch_info: dict) -> dict:
        losses = dict()
        x = self.student.extract_feat(batch_inputs)

        # rpn loss
        rpn_losses, rpn_results_list = self.rpn_loss_by_pseudo_instances(
            x, batch_data_samples)
        losses.update(**rpn_losses)

        # roi loss
        roi_losses = self.roi_loss_by_pseudo_instances(
            x, rpn_results_list, batch_data_samples, batch_info)
        losses.update(**roi_losses)

        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.0)
        return rename_loss_dict('unsup_', reweight_loss_dict(losses, unsup_weight))

    def rpn_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                     batch_data_samples: SampleList) -> Tuple[dict, InstanceList]:
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for data_samples in rpn_data_samples:
            data_samples.gt_instances = InstanceData.cat(
                [data_samples.gt_instances, data_samples.uncertain_instances])
            # set cat_id of gt_labels to 0 in RPN
            data_samples.gt_instances.labels = \
                torch.zeros_like(data_samples.gt_instances.labels)

        proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                  self.student.test_cfg.rpn)

        rpn_losses, rpn_results_list = self.student.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        for key in rpn_losses.keys():
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        return rpn_losses, rpn_results_list

    def roi_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                     rpn_results_list: InstanceList,
                                     batch_data_samples: SampleList,
                                     batch_info: dict) -> dict:
        losses = dict()

        proposals, labels, label_weights, bbox_targets, bbox_weights \
            = self.proposal_self_assignment(
                rpn_results_list, batch_data_samples, batch_info)
        rois = bbox2roi(proposals)
        bbox_results = self.student.roi_head._bbox_forward(x, rois)
        losses.update(**self.student.roi_head.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights))
        return losses

    def proposal_self_assignment(self, rpn_results_list: InstanceList,
                                 batch_data_samples: SampleList, batch_info):
        UNCERTAIN_LABEL = 1e6
        roi_cfg = self.student.roi_head.train_cfg
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        batch_uncertain_instances = [
            data_samples.uncertain_instances for data_samples in batch_data_samples]

        priors_list = []
        labels_list = []
        label_weights_list = []
        bbox_targets_list = []
        bbox_weights_list = []
        num_imgs = len(batch_data_samples)
        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            gt_instances = batch_gt_instances[i]
            uncertain_instances = batch_uncertain_instances[i]
            uncertain_labels = uncertain_instances.labels
            uncertain_instances.labels = torch.full_like(uncertain_labels, UNCERTAIN_LABEL)

            # put uncertain_instances ahead to ease the index of uncertain_labels
            concat_instances = InstanceData.cat([uncertain_instances, gt_instances])
            assign_result = self.student.roi_head.bbox_assigner.assign(
                rpn_results, concat_instances,
                batch_gt_instances_ignore[i])
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result, rpn_results, concat_instances)

            reliable_pos_flag = sampling_result.pos_gt_labels != UNCERTAIN_LABEL
            uncertain_pos_flag = sampling_result.pos_gt_labels == UNCERTAIN_LABEL
            uncertain_inds = sampling_result.pos_inds[uncertain_pos_flag]
            uncertain_assigned_gt_inds = assign_result.gt_inds[uncertain_inds] - 1

            reliable_priors = sampling_result.pos_priors[reliable_pos_flag]
            uncertain_priors = sampling_result.pos_priors[uncertain_pos_flag]
            neg_priors = sampling_result.neg_priors

            reliable_gt_labels = sampling_result.pos_gt_labels[reliable_pos_flag]
            reliable_gt_bboxes = sampling_result.pos_gt_bboxes[reliable_pos_flag]

            uncertain_gt_labels = uncertain_labels[uncertain_assigned_gt_inds]
            uncertain_gt_bboxes = sampling_result.pos_gt_bboxes[uncertain_pos_flag]

            num_reliable = reliable_priors.size(0)
            num_uncertain = uncertain_priors.size(0)
            num_neg = neg_priors.size(0)
            num_samples = num_reliable + num_uncertain + num_neg

            labels = reliable_priors.new_full((num_samples,),
                                              self.num_classes,
                                              dtype=torch.long)
            reg_dim = reliable_gt_bboxes.size(-1) if self.student.roi_head.bbox_head.reg_decoded_bbox \
                else self.student.roi_head.bbox_head.bbox_coder.encode_size
            label_weights = reliable_priors.new_zeros(num_samples)
            bbox_targets = reliable_priors.new_zeros(num_samples, reg_dim)
            bbox_weights = reliable_priors.new_zeros(num_samples, reg_dim)

            if num_reliable > 0:
                labels[:num_reliable] = reliable_gt_labels
                reliable_weight = 1.0 if roi_cfg.pos_weight <= 0 else roi_cfg.pos_weight
                label_weights[:num_reliable] = reliable_weight
                if not self.student.roi_head.bbox_head.reg_decoded_bbox:
                    reliable_bbox_targets = self.student.roi_head.bbox_head.bbox_coder.encode(
                        reliable_priors, reliable_gt_bboxes)
                else:
                    # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                    # is applied directly on the decoded bounding boxes, both
                    # the predicted boxes and regression targets should be with
                    # absolute coordinate format.
                    reliable_bbox_targets = get_box_tensor(reliable_gt_bboxes)
                bbox_targets[:num_reliable, :] = reliable_bbox_targets
                bbox_weights[:num_reliable, :] = 1
            if num_uncertain > 0:
                labels[num_reliable:num_reliable + num_uncertain] = uncertain_gt_labels
                if not self.student.roi_head.bbox_head.reg_decoded_bbox:
                    uncertain_bbox_targets = self.student.roi_head.bbox_head.bbox_coder.encode(
                        uncertain_priors, uncertain_gt_bboxes)
                else:
                    uncertain_bbox_targets = get_box_tensor(uncertain_gt_bboxes)
                bbox_targets[num_reliable:num_reliable + num_uncertain, :] = uncertain_bbox_targets

                uncertain_label_weights = label_weights.new_zeros(num_uncertain)

                teacher_matrix = batch_info['homography_matrix'][i]
                img_shape = batch_info['img_shape'][i]
                student_matrix = torch.tensor(
                    batch_data_samples[i].homography_matrix, device=teacher_matrix.device)
                homography_matrix = teacher_matrix @ student_matrix.inverse()
                projected_priors = bbox_project(uncertain_priors, homography_matrix, img_shape)
                scores, bboxes = self.teacher_forward_single(
                    batch_info['feat'], projected_priors, img_id=i, img_shape=img_shape)

                uncertain_label_weights[:] = scores[
                    torch.arange(num_uncertain, dtype=torch.long, device=scores.device), uncertain_gt_labels]

                if self.use_rplm:
                    uncertain_bbox_weights = bbox_weights.new_zeros((num_uncertain, bbox_weights.size(-1)))

                    gt_bboxes = uncertain_instances.bboxes
                    gt_bboxes = bbox_project(gt_bboxes, homography_matrix, img_shape)
                    bboxes = bboxes.view(num_uncertain, -1, uncertain_gt_bboxes.size(-1))
                    for gt_ind in torch.unique(uncertain_assigned_gt_inds):
                        flag = uncertain_assigned_gt_inds == gt_ind
                        cur_label = uncertain_labels[gt_ind]
                        mean_score = scores[flag, cur_label].mean()
                        mean_iou = bbox_overlaps(gt_bboxes[gt_ind:gt_ind + 1], bboxes[flag, cur_label]).mean()
                        if mean_score >= self.rplm_score and mean_iou >= self.rplm_iou:
                            uncertain_label_weights[flag] = 1.0
                            uncertain_bbox_weights[flag, :] = 1.0
                    bbox_weights[num_reliable:num_reliable + num_uncertain, :] = uncertain_bbox_weights
                label_weights[num_reliable:num_reliable + num_uncertain] = uncertain_label_weights
            if num_neg > 0:
                label_weights[-num_neg:] = 1.0

            priors_list.append(torch.cat([reliable_priors, uncertain_priors, neg_priors], dim=0))
            labels_list.append(labels)
            label_weights_list.append(label_weights)
            bbox_targets_list.append(bbox_targets)
            bbox_weights_list.append(bbox_weights)

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        return priors_list, labels, label_weights, bbox_targets, bbox_weights

    @torch.no_grad()
    def teacher_forward_single(self, x: Tuple[Tensor], priors: Tensor, img_id: int, img_shape: Tuple[int, int]):
        img_inds = priors.new_full((priors.size(0), 1), img_id)
        rois = torch.cat([img_inds, priors], dim=-1)

        bbox_results = self.teacher.roi_head._bbox_forward(x, rois)
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']

        scores = torch.softmax(cls_scores, dim=-1)
        bboxes = self.teacher.roi_head.bbox_head.bbox_coder.decode(
            rois[:, 1:], bbox_preds, max_shape=img_shape)

        return scores, bboxes

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
