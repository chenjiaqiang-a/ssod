from torch import Tensor
from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class PseudoLabelWeight:
    def __init__(self, num_classes: int, threshold: float = 0.9):
        self.num_classes = num_classes
        self.threshold = threshold

    def update(self):
        pass

    def compute_weights(self, scores: Tensor):
        return scores.new_ones((scores.size(0),))
