from typing import Optional, Tuple, Dict

import torch
import numpy as np
from torch import Tensor
from sklearn.mixture import GaussianMixture
from mmengine import MessageHub
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_project
from mmdet.models import SemiBaseDetector, SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class ConsistentTeacher(SemiBaseDetector):
    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(ConsistentTeacher, self).__init__(
            detector, semi_train_cfg, semi_test_cfg, data_preprocessor, init_cfg)
        if not isinstance(self.teacher, SingleStageDetector):
            raise UserWarning('ConsistentTeacher currently only supports single-stage detectors,'
                              ' using other types of detectors may cause errors.')

        # TODO: handle other cases
        self.num_classes = detector.bbox_head.num_classes
        self.scores_queue_size = self.semi_train_cfg.get('scores_queue_size', 100)
        self.register_buffer('scores_queue',
                             torch.zeros((self.num_classes, self.scores_queue_size)))
        self.iter = 0

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        self.teacher.eval()
        feats = self.teacher.extract_feat(batch_inputs)
        batch_results = self.teacher.bbox_head.predict(
            feats, batch_data_samples, rescale=False)

        # filter invalid box roughly
        filtered_results_list = []
        for res in batch_results:
            if len(res) == 0:
                filtered_results_list.append(res)
            else:
                dynamic_ratio = self.semi_train_cfg.get('dynamic_ratio', 1.0)
                scores = torch.sort(res.scores, descending=True)[0]
                num_gt = int(scores.sum() * dynamic_ratio + 0.5)
                num_gt = min(num_gt, len(scores) - 1)
                thr = scores[num_gt] - 1e-5
                filtered_results_list.append(res[res.scores > thr])
        batch_results = filtered_results_list

        # filter pseudo labels using thr generated by gmm
        scores = torch.cat([res.scores for res in batch_results])
        labels = torch.cat([res.labels for res in batch_results])
        thrs = torch.zeros_like(scores)
        for label in torch.unique(labels):
            scores_add = scores[labels == label]
            scores_new = torch.cat([scores_add, self.scores_queue[label]])[:self.scores_queue_size]
            self.scores_queue[label] = scores_new
            thr = self.gmm_policy(scores_new[scores_new > 0])
            thrs[labels == label] = thr
        mean_thr = thrs.mean()
        if len(thrs) == 0:
            mean_thr.fill_(0)
        batch_thrs = torch.split(thrs, [len(res.labels) for res in batch_results])

        batch_info = {
            'gmm_thr': mean_thr,
        }
        for data_samples, res, thrs in zip(batch_data_samples, batch_results, batch_thrs):
            data_samples.gt_instances = res[res.scores > thrs]
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)

        return batch_data_samples, batch_info

    def gmm_policy(self, scores: Tensor) -> float:
        default_thr = self.semi_train_cfg.get('default_gmm_thr', 0.0)
        if len(scores) < 4:
            return default_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)

        policy = self.semi_train_cfg.get('gmm_thr_policy', 'high')
        assert policy in ['high', 'middle']
        if (gmm_assignment == 1).any():
            if policy == 'high':
                gmm_scores[gmm_assignment == 0] = -np.inf
                ind = np.argmax(gmm_scores, axis=0)
                pos_inds = (gmm_assignment == 1) & (scores >= scores[ind]).squeeze()
                pos_thr = float(scores[pos_inds].min())
            else:
                pos_thr = float(scores[gmm_assignment == 1].min())
        else:
            pos_thr = default_thr
        return pos_thr

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

        origin_pseudo_data_samples, batch_info = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'],
            multi_batch_data_samples['unsup_teacher'])

        message_hub = MessageHub.get_current_instance()
        iter = message_hub.get_info('iter') + 1

        if iter > self.semi_train_cfg.get('burn_in_steps', -1):
            multi_batch_data_samples[
                'unsup_student'] = self.project_pseudo_instances(
                origin_pseudo_data_samples,
                multi_batch_data_samples['unsup_student'])
            losses.update(**self.loss_by_pseudo_instances(
                multi_batch_inputs['unsup_student'],
                multi_batch_data_samples['unsup_student'], batch_info))

        losses['gmm_thr'] = batch_info['gmm_thr']
        return losses
