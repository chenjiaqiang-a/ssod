from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from mmdet.registry import TASK_UTILS
from mmengine import MessageHub, print_log

from ssod.utils.dist_utils import concat_all_gather
from .pseudo_label_weight import PseudoLabelWeight
from .dist_align import DistAlignEMA


@TASK_UTILS.register_module()
class SoftMatchWeight(PseudoLabelWeight):
    def __init__(self,
                 num_classes: int,
                 n_sigma: int = 2,
                 momentum: float = 0.999,
                 per_class: bool = False,
                 dist_align: Optional[dict] = None):
        super().__init__(num_classes)
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum

        # initialize Gaussian mean and variance
        if self.per_class:
            self.mu_t = torch.ones((self.num_classes,)) / self.num_classes
            self.var_t = torch.ones((self.num_classes,))
        else:
            self.mu_t = torch.tensor(1. / self.num_classes)
            self.var_t = torch.tensor(1.)

        self.dist_align_cfg = dist_align
        if dist_align is not None:
            self.dist_align = DistAlignEMA(self.num_classes, **dist_align)


    @torch.no_grad()
    def update(self, scores):
        if dist.is_initialized():
            scores = concat_all_gather(scores)

        max_scores, max_idx = torch.max(scores, dim=-1)
        if self.per_class:
            mu_t = torch.zeros_like(self.mu_t)
            var_t = torch.ones_like(self.var_t)
            for i in range(self.num_classes):
                score = max_scores[max_idx == i]
                if len(score) > 1:
                    mu_t[i] = torch.mean(score)
                    var_t[i] = torch.var(score, unbiased=True)
        else:
            mu_t = torch.mean(max_scores)
            var_t = torch.var(max_scores, unbiased=True)
        self.mu_t = self.m * self.mu_t + (1 - self.m) * mu_t
        self.var_t = self.m * self.var_t + (1 - self.m) * var_t

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % 1000 == 0:
            print_log(f'mu_t:  {self.mu_t}')
            print_log(f'var_t: {self.var_t}')

    @torch.no_grad()
    def compute_weights(self, scores: Tensor):
        if not self.mu_t.is_cuda:
            self.mu_t = self.mu_t.to(scores.device)
        if not self.var_t.is_cuda:
            self.var_t = self.var_t.to(scores.device)

        if self.dist_align_cfg is not None:
            scores = self.dist_align(scores)
        self.update(scores)

        max_scores, max_idx = torch.max(scores, dim=-1)
        if self.per_class:
            mu = self.mu_t[max_idx]
            var = self.var_t[max_idx]
        else:
            mu = self.mu_t
            var = self.var_t
        weights = torch.exp(
            - (torch.clamp(max_scores - mu, max=0.0) ** 2) / (2. * var / (self.n_sigma ** 2)))

        return weights
