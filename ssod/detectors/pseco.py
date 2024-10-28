import copy
from typing import Optional, Tuple

import torch
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models import SemiBaseDetector, TwoStageDetector
from mmdet.models.utils import filter_gt_instances, unpack_gt_instances, rename_loss_dict, reweight_loss_dict
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_project, bbox2roi, bbox_overlaps
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, InstanceList


@MODELS.register_module()
class PseCo(SemiBaseDetector):
    def __init__(self,
                 detector: ConfigType,
                 data_preprocessor: OptConfigType = None,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        if semi_train_cfg is None:
            semi_train_cfg = dict()

        super(PseCo, self).__init__(
            detector=detector,
            data_preprocessor=data_preprocessor,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            init_cfg=init_cfg)

        if not isinstance(self.teacher, TwoStageDetector):
            raise UserWarning('Pseco currently only supports two-stage detectors,'
                              ' using other types of detectors may cause errors.')

        pla_iou_thr = self.semi_train_cfg.get('PLA_iou_thr', 0.4)
        self.initial_assigner = TASK_UTILS.build(dict(
            type='MaxIoUAssigner',
            pos_iou_thr=pla_iou_thr,
            neg_iou_thr=pla_iou_thr,
            min_pos_iou=pla_iou_thr,
            match_low_quality=False,
            ignore_iof_thr=-1,
        ))

        self.num_classes = self.student.roi_head.bbox_head.num_classes

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        self.teacher.eval()
        feats = self.teacher.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.teacher.rpn_head.predict(
                feats, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.teacher.roi_head.predict(
            feats, rpn_results_list, batch_data_samples, rescale=False)

        batch_info = {
            'feats': feats,
            'rpn_results_list': rpn_results_list,
            'matrix_list': [],
            'img_shape_list': [],
        }
        for data_samples, results in zip(batch_data_samples, results_list):
            homography_matrix = torch.from_numpy(data_samples.homography_matrix).to(
                self.data_preprocessor.device)
            batch_info['matrix_list'].append(homography_matrix)
            batch_info['img_shape_list'].append(data_samples.img_shape)

            data_samples.gt_instances = results
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                homography_matrix.inverse(),
                data_samples.ori_shape)

        score_thr = self.semi_train_cfg.get('pseudo_label_initial_score_thr', 0.3)
        return filter_gt_instances(batch_data_samples, score_thr=score_thr), batch_info

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_teacher_info: Optional[dict] = None) -> dict:
        losses = dict()
        feats = self.student.extract_feat(batch_inputs)

        # rpn
        rpn_losses, stu_rpn_results_list = self.rpn_loss_by_pseudo_instances(
            feats, batch_data_samples)
        losses.update(rpn_losses)

        # teacher prediction
        tea_matrix_list = batch_teacher_info['matrix_list']
        stu_matrix_list = [
            torch.from_numpy(data_samples.homography_matrix).to(self.data_preprocessor.device)
            for data_samples in batch_data_samples
        ]
        tea_img_shape_list = batch_teacher_info['img_shape_list']
        stu_img_shape_list = [data_samples.img_shape for data_samples in batch_data_samples]
        if self.semi_train_cfg.get('use_teacher_proposal', True) is True:
            rpn_results_list = copy.deepcopy(batch_teacher_info['rpn_results_list'])
            bboxes_preds, scores_preds = self.teacher_bbox_forward(
                batch_teacher_info['feats'], batch_teacher_info['rpn_results_list'])

            tea_bboxes_list = []
            tea_scores_list = scores_preds
            for bboxes_pred, rpn_results, tea_matrix, stu_matrix, tea_img_shape, stu_img_shape in zip(
                    bboxes_preds, rpn_results_list, tea_matrix_list, stu_matrix_list,
                    tea_img_shape_list, stu_img_shape_list):
                homography_matrix = stu_matrix @ tea_matrix.inverse()
                tea_bboxes = bbox_project(bboxes_pred, homography_matrix, stu_img_shape)

                rpn_results.bboxes = bbox_project(rpn_results.bboxes, homography_matrix, stu_img_shape)
                tea_bboxes_list.append(tea_bboxes)
        else:
            rpn_results_list = copy.deepcopy(stu_rpn_results_list)

            for res, tea_matrix, stu_matrix, tea_img_shape, stu_img_shape in zip(
                    stu_rpn_results_list, tea_matrix_list, stu_matrix_list,
                    tea_img_shape_list, stu_img_shape_list):
                homography_matrix = tea_matrix @ stu_matrix.inverse()
                res.bboxes = bbox_project(res.bboxes, homography_matrix, tea_img_shape)

            bboxes_preds, scores_preds = self.teacher_bbox_forward(
                batch_teacher_info['feats'], stu_rpn_results_list)

            tea_bboxes_list = []
            tea_scores_list = scores_preds
            for bboxes_pred, tea_matrix, stu_matrix, tea_img_shape, stu_img_shape in zip(
                    bboxes_preds, tea_matrix_list, stu_matrix_list,
                    tea_img_shape_list, stu_img_shape_list):
                homography_matrix = stu_matrix @ tea_matrix.inverse()
                tea_bboxes = bbox_project(bboxes_pred, homography_matrix, stu_img_shape)
                tea_bboxes_list.append(tea_bboxes)
        batch_teacher_info['bboxes_list'] = tea_bboxes_list
        batch_teacher_info['scores_list'] = tea_scores_list

        # roi
        losses.update(**self.roi_loss_by_pseudo_instances(
            feats, rpn_results_list, batch_data_samples, batch_teacher_info))

        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))

    def rpn_loss_by_pseudo_instances(self, feats: Tuple[Tensor],
                                     batch_data_samples: SampleList):
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        rpn_data_samples = filter_gt_instances(
            rpn_data_samples, score_thr=self.semi_train_cfg.rpn_pseudo_thr)
        proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                  self.student.test_cfg.rpn)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.student.rpn_head.loss_and_predict(
            feats, rpn_data_samples, proposal_cfg=proposal_cfg)

        for key in rpn_losses.keys():
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        return rpn_losses, rpn_results_list

    def roi_loss_by_pseudo_instances(self, feats: Tuple[Tensor],
                                     rpn_results_list: InstanceList,
                                     batch_data_samples: SampleList,
                                     batch_teacher_info: Optional[dict] = None):
        losses = dict()

        batch_data_samples = filter_gt_instances(
            batch_data_samples,
            score_thr=self.semi_train_cfg.get('roi_pseudo_thr', '0.5'))

        tea_bboxes_list = batch_teacher_info['bboxes_list']
        tea_scores_list = batch_teacher_info['scores_list']
        sampling_results = self.prediction_guided_label_assign(
            rpn_results_list, batch_data_samples, tea_bboxes_list, tea_scores_list)

        selected_bboxes = [res.priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]

        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feats, rois)

        cls_reg_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, self.student.train_cfg.rcnn)

        reg_weights = self.positive_proposal_consistency_voting(
            bbox_results['bbox_pred'],
            cls_reg_targets[0],
            selected_bboxes,
            pos_gt_bboxes_list,
            pos_assigned_gt_inds_list)

        # Focal loss
        roi_losses = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *(cls_reg_targets[:3]),
            reg_weights,
        )

        losses.update(roi_losses)
        return losses

    @torch.no_grad()
    def teacher_bbox_forward(self, feats, rpn_results_list):
        proposals_list = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals_list)
        bbox_results = self.teacher.roi_head._bbox_forward(feats, rois)
        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']

        scores_preds = cls_scores.softmax(dim=-1)[:, :-1]
        labels_preds = scores_preds.max(dim=-1)[1]
        bbox_preds = bbox_preds.view(
            bbox_preds.size(0), -1, self.teacher.roi_head.bbox_head.bbox_coder.encode_size)
        bbox_preds = bbox_preds[torch.arange(bbox_preds.size(0)), labels_preds]
        bboxes_preds = self.teacher.roi_head.bbox_head.bbox_coder.decode(
            rois[:, 1:], bbox_preds)

        num_proposals_per_img = tuple(len(p) for p in proposals_list)
        scores_preds = scores_preds.split(num_proposals_per_img, 0)
        bboxes_preds = bboxes_preds.split(num_proposals_per_img, 0)
        return bboxes_preds, scores_preds

    @torch.no_grad()
    def prediction_guided_label_assign(self,
                                       rpn_results_list,
                                       batch_data_samples,
                                       tea_bboxes_list,
                                       tea_scores_list):
        num_imgs = len(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = unpack_gt_instances(batch_data_samples)

        sampling_results = []
        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.initial_assigner.assign(
                InstanceData(priors=tea_bboxes_list[i]),
                # rpn_results,
                batch_gt_instances[i],
                batch_gt_instances_ignore[i])

            gt_inds = assign_result.gt_inds
            pos_inds = torch.nonzero(gt_inds > 0, as_tuple=False).reshape(-1)
            pos_assigned_gt_inds = gt_inds[pos_inds] - 1
            pos_labels = batch_gt_instances[i].labels[pos_assigned_gt_inds]

            pos_tea_scores = tea_scores_list[i][pos_inds]
            pos_tea_bboxes = tea_bboxes_list[i][pos_inds]
            ious = bbox_overlaps(pos_tea_bboxes, batch_gt_instances[i].bboxes)

            wh = rpn_results.priors[:, 2:4] - rpn_results.priors[:, :2]
            areas = wh.prod(dim=-1)
            pos_areas = areas[pos_inds]

            refined_gt_inds = self.assignment_refinement(gt_inds,
                                                         pos_inds,
                                                         pos_assigned_gt_inds,
                                                         pos_labels,
                                                         pos_tea_scores,
                                                         ious,
                                                         pos_areas)

            assign_result.gt_inds = refined_gt_inds + 1
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i])
            sampling_results.append(sampling_result)
        return sampling_results

    @torch.no_grad()
    def assignment_refinement(self, gt_inds, pos_inds, pos_assigned_gt_inds, pos_labels,
                              pos_tea_scores, ious, pos_areas):
        refined_gt_inds = gt_inds.new_full((gt_inds.shape[0],), -1)
        refined_pos_gt_inds = gt_inds.new_full((pos_inds.shape[0],), -1)

        gt_inds_set = torch.unique(pos_assigned_gt_inds)
        for gt_ind in gt_inds_set:
            pos_idx_per_gt = torch.nonzero(pos_assigned_gt_inds == gt_ind).reshape(-1)
            target_labels = pos_labels[pos_idx_per_gt]
            target_scores = pos_tea_scores[pos_idx_per_gt, target_labels]
            target_ious = ious[pos_idx_per_gt, gt_ind]
            target_areas = pos_areas[pos_idx_per_gt]

            cost = (target_ious * target_scores).sqrt()
            _, sort_idx = torch.sort(cost, descending=True)

            candidate_topk = min(pos_idx_per_gt.shape[0],
                                 self.semi_train_cfg.get('PLA_candidate_topk', 12))
            topk_ious, _ = torch.topk(target_ious, candidate_topk, dim=0)
            # calculate dynamic k for each gt
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
            sort_idx = sort_idx[:dynamic_ks]
            # filter some invalid (area == 0) proposals
            sort_idx = sort_idx[target_areas[sort_idx] > 0]
            pos_idx_per_gt = pos_idx_per_gt[sort_idx]

            refined_pos_gt_inds[pos_idx_per_gt] = gt_ind
        refined_gt_inds[pos_inds] = refined_pos_gt_inds
        return refined_gt_inds

    @torch.no_grad()
    def positive_proposal_consistency_voting(self,
                                             bbox_preds,
                                             labels,
                                             proposals_list,
                                             pos_gt_bboxes_list,
                                             pos_assigned_gt_inds_list):
        """ Compute regression weights for each proposal according
            to Positive-proposal Consistency Voting (PCV).

        Args:
            bbox_preds (Tensors): bbox preds for proposals.
            labels (Tensors): assigned class label for each proposals.
                0-79 indicate fg, 80 indicates bg.
            proposals_list (list[Tensor]): proposals for each image.
            pos_gt_bboxes_list (list[Tensor]): label assignent results
            pos_assigned_gt_inds_list (list[Tensor]): label assignent results

        Returns:
            reg_weights (Tensors): Regression weights for proposals.
        """

        nums = [_.shape[0] for _ in proposals_list]
        labels = labels.split(nums, dim=0)
        bbox_preds = bbox_preds.split(nums, dim=0)

        reg_weights_list = []

        for bbox_pred, label, proposals, pos_gt_bboxes, pos_assigned_gt_inds in zip(
                bbox_preds, labels, proposals_list, pos_gt_bboxes_list, pos_assigned_gt_inds_list):

            pos_inds = ((label >= 0) &
                        (label < self.student.roi_head.bbox_head.num_classes)).nonzero().reshape(-1)
            reg_weights = proposals.new_zeros(bbox_pred.shape[0], 4)
            pos_proposals = proposals[pos_inds]
            if len(pos_inds):
                pos_reg_weights = proposals.new_zeros(pos_inds.shape[0], 4)
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1, 4)[
                    pos_inds, label[pos_inds]
                ]
                decoded_bboxes = self.student.roi_head.bbox_head.bbox_coder.decode(
                    pos_proposals, pos_bbox_pred)

                gt_inds_set = torch.unique(pos_assigned_gt_inds)

                ious = bbox_overlaps(
                    decoded_bboxes,
                    pos_gt_bboxes,
                    is_aligned=True)

                for gt_ind in gt_inds_set:
                    idx_per_gt = (pos_assigned_gt_inds == gt_ind).nonzero().reshape(-1)
                    if idx_per_gt.shape[0] > 0:
                        pos_reg_weights[idx_per_gt] = ious[idx_per_gt].mean()
                reg_weights[pos_inds] = pos_reg_weights

            reg_weights_list.append(reg_weights)
        reg_weights = torch.cat(reg_weights_list, 0)

        return reg_weights
