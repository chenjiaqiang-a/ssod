from copy import deepcopy
from collections import Counter

import torch
from torch import Tensor
from mmdet.registry import TASK_UTILS

from .pseudo_label_weight import PseudoLabelWeight

@TASK_UTILS.register_module()
class FlexMatchWeight(PseudoLabelWeight):
    def __init__(self,
                 num_classes: int,
                 threshold: float = 0.9,
                 dist_len: int = 50000,
                 thresh_warmup: bool = True):
        super().__init__(num_classes, threshold)
        self.dist_len = dist_len
        self.thresh_warmup = thresh_warmup
        self.selected_label = torch.ones((self.dist_len,), dtype=torch.long) * -1
        self.point = 0
        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update(self):
        label_counter = Counter(self.selected_label.tolist())
        if max(label_counter.values()) < self.dist_len:
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = label_counter[i] / max(label_counter.values())
            else:
                wo_negative_one = deepcopy(label_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    self.classwise_acc[i] = label_counter[i] / max(wo_negative_one.values())

    @torch.no_grad()
    def compute_weights(self, scores: Tensor):
        if not self.selected_label.is_cuda:
            self.selected_label = self.selected_label.to(scores.device)
        if not self.classwise_acc.is_cuda:
            self.classwise_acc = self.classwise_acc.to(scores.device)

        max_scores, max_idx = torch.max(scores, dim=-1)
        weights = max_scores.ge(self.threshold * (self.classwise_acc[max_idx] / (2.0 - self.classwise_acc[max_idx])))
        weights = weights.to(max_scores.dtype)

        select = max_scores.ge(self.threshold)
        if select.any():
            labels = max_idx[select]
            if self.point + len(labels) < self.dist_len:
                self.selected_label[self.point:self.point+len(labels)] = labels
            else:
                self.selected_label[:len(labels)] = labels
                self.point = len(labels)
        self.update()

        return weights

