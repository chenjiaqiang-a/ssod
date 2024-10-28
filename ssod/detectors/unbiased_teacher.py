from typing import Optional, Dict

from torch import Tensor

from mmengine import MessageHub
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.models import SemiBaseDetector
from mmdet.models.utils import filter_gt_instances, rename_loss_dict, reweight_loss_dict
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class UnbiasedTeacher(SemiBaseDetector):
    """Implementation of `Unbiased Teacher for Semi-Supervised
    Object Detection <https://arxiv.org/abs/2102.09480>`_

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        init_sup_cpt (str, optional): The path to checkpoint of
             initially trained supervised model on labeled data.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

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

        if cur_iter > self.semi_train_cfg.get('burn_in_steps', -1):
            origin_pseudo_data_samples, batch_info = self.get_pseudo_instances(
                multi_batch_inputs['unsup_teacher'],
                multi_batch_data_samples['unsup_teacher'])
            multi_batch_data_samples[
                'unsup_student'] = self.project_pseudo_instances(
                origin_pseudo_data_samples,
                multi_batch_data_samples['unsup_student'])
            losses.update(**self.loss_by_pseudo_instances(
                multi_batch_inputs['unsup_student'],
                multi_batch_data_samples['unsup_student'], batch_info))
        return losses

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """
        batch_data_samples = filter_gt_instances(
            batch_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)
        losses = self.student.loss(batch_inputs, batch_data_samples)
        pseudo_instances_num = sum([
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])

        if self.semi_train_cfg.get('use_box_reg', False) is False:
            for key in list(losses.keys()):
                if 'bbox' in key:
                    del losses[key]

        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))
