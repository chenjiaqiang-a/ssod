import torch
from torch import Tensor
from mmdet.registry import TASK_UTILS

from .pseudo_label_weight import PseudoLabelWeight


@TASK_UTILS.register_module()
class FixMatchWeight(PseudoLabelWeight):

    @torch.no_grad()
    def compute_weights(self, scores: Tensor):
        max_scores, _ = torch.max(scores, dim=-1)
        weights = max_scores.ge(self.threshold).to(max_scores.dtype)
        return weights
