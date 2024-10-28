from typing import Optional

import torch
import torch.distributed as dist
import numpy as np
from torch import Tensor

from ssod.utils.dist_utils import concat_all_gather


class DistAlignEMA:
    def __init__(self,
                 num_classes: int,
                 momentum: float = 0.999,
                 target_type: str = 'uniform',
                 target: Optional[Tensor] = None):
        self.num_classes = num_classes
        self.m = momentum

        self.p_target = self.set_target(target_type, target)
        self.p_model = None

    def set_target(self, target_type, p_target):
        assert target_type in ['uniform', 'gt']

        if target_type == 'uniform':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return p_target

    @torch.no_grad()
    def __call__(self, scores: Tensor):
        self.update(scores)

        scores_norm = scores * ((self.p_target + 1e-6) / (self.p_model + 1e-6)).unsqueeze(0)
        scores_norm = scores_norm / scores_norm.sum(dim=-1, keepdim=True)
        return scores_norm

    @torch.no_grad()
    def update(self, scores: Tensor):
        if dist.is_initialized():
            scores = concat_all_gather(scores)
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(scores.device)

        prob = torch.mean(scores, dim=0)
        if self.p_model is None:
            self.p_model = prob
        else:
            self.p_model = self.m * self.p_model + (1 - self.m) * prob
