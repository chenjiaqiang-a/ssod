from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.registry import HOOKS
from mmdet.structures.bbox import bbox_project
import cv2
import numpy as np
from mmengine.structures import InstanceData


@HOOKS.register_module()
class PlottingHook(Hook):
    def __init__(self, interval: int = 500):
        self.interval = interval

    def before_train(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        if (runner.iter + 1) % self.interval != 0:
            return
        data = runner.model.data_preprocessor(data_batch, True)
        teacher = runner.model.teacher
        student = runner.model.student

        # 获取teacher Proposals
        batch_inputs, batch_data_samples =\
            data['inputs']['unsup_teacher'], data['data_samples']['unsup_teacher']
        x = teacher.extract_feat(batch_inputs)
        rpn_results_list = teacher.rpn_head.predict(
            x, batch_data_samples, rescale=False)
        results_list = teacher.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=False)
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        predictions_list = teacher.roi_head.predict_bbox(
            x, batch_img_metas, rpn_results_list, rcnn_test_cfg=None, rescale=False)

        # 获取数据：original/weak/strong
        img = data_batch['inputs']['unsup_teacher'][0]
        weak_aug = data_batch['data_samples']['unsup_teacher'][0]
        strong_aug = data_batch['data_samples']['unsup_student'][0]
        if weak_aug.flip is True:
            img = img.flip(-1)
        img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rpn_results, results, predictions = rpn_results_list[0], results_list[0], predictions_list[0]
        rpn_results.bboxes = bbox_project(
            rpn_results.bboxes,
            torch.from_numpy(weak_aug.homography_matrix).inverse().to(
                teacher.data_preprocessor.device), weak_aug.ori_shape)
        results.bboxes = bbox_project(
            results.bboxes,
            torch.from_numpy(weak_aug.homography_matrix).inverse().to(
                teacher.data_preprocessor.device), weak_aug.ori_shape)

        out_dir = runner.work_dir
        bboxes = results.bboxes.cpu().numpy()
        labels = np.ones((len(bboxes),))
        plot_bboxes(img, bboxes, labels, out_dir + f'/iter_{runner.iter+1}_ILPLs_0.05.png')

        bboxes = rpn_results.bboxes.cpu().numpy()
        bg_ind = teacher.roi_head.bbox_head.num_classes
        _, preds = torch.max(predictions.scores, dim=-1)
        labels = np.ones((len(bboxes),))
        labels[preds.cpu().numpy() == bg_ind] = 0
        plot_bboxes(img, bboxes, labels, out_dir + f'/iter_{runner.iter+1}_teacher.png')

        student_rpn_results = InstanceData(priors=bbox_project(
            rpn_results.bboxes, torch.from_numpy(strong_aug.homography_matrix).to(
                student.data_preprocessor.device), strong_aug.img_shape))
        student_results = InstanceData(
            bboxes=bbox_project(results.bboxes, torch.from_numpy(strong_aug.homography_matrix).to(
                student.data_preprocessor.device), strong_aug.img_shape),
            labels=results.labels,
            scores=results.scores)

        for thr in [0.5, 0.7, 0.9]:
            results = results[results.scores > thr]
            student_results = student_results[student_results.scores > thr]

            bboxes = results.bboxes.cpu().numpy()
            labels = np.ones((len(bboxes),))
            plot_bboxes(img, bboxes, labels, out_dir + f'/iter_{runner.iter+1}_ILPLs_{thr:.1f}.png')

            assign_results = student.roi_head.bbox_assigner.assign(
                student_rpn_results, student_results)
            bboxes = rpn_results.bboxes.cpu().numpy()
            assigned_gt_inds = assign_results.gt_inds
            labels = np.ones((len(bboxes),))
            labels[assigned_gt_inds.cpu().numpy() == 0] = 0
            plot_bboxes(img, bboxes, labels, out_dir + f'/iter_{runner.iter+1}_student_{thr:.1f}.png')


def plot_bboxes(img, bboxes, labels, file_name):
    assert len(bboxes) == len(labels)
    bboxes = bboxes.astype(int)
    img = np.ascontiguousarray(img)
    for i in range(min(200, len(bboxes))):
        x1, y1 = bboxes[i][0], bboxes[i][1]
        x2, y2 = bboxes[i][2], bboxes[i][3]
        color = (220, 20, 60) if labels[i] == 1 else (0, 0, 230)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.imwrite(file_name, img)
