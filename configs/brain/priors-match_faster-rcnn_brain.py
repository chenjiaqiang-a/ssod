custom_imports = dict(
    imports=['ssod.detectors.priors_match', 'ssod.task_modules.semi_weights'],
    allow_failed_imports=False)

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/semi_brain_detection.py',
    '../_base_/schedules/semi_schedule_10k.py',
    '../_base_/default_runtime.py',
]

# model
detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[21.301, 19.474, 19.891],
    std=[24.485, 20.939, 21.263],
    bgr_to_rgb=True,
    pad_size_divisor=32)
detector.roi_head.bbox_head.num_classes = 3

model = dict(
    _delete_=True,
    type='PriorsMatch',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        rpn_pos_thr=0.9,
        rpn_neg_thr=0.1,
        with_background=True,
        semi_weight_cfg=dict(
            type='SoftMatchWeight',
            n_sigma=2,
            momentum=0.999,
            per_class=True,
            dist_align=dict(target_type='uniform'))),
    semi_test_cfg=dict(predict_on='teacher'))

# dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

# semi-supervised for 10% labeled brain tumor data
labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.0.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.0-unlabeled.json'
# labeled_dataset.ann_file = 'semi_anns/instances_train2017.2@10.0.json'
# unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.2@10.0-unlabeled.json'
# labeled_dataset.ann_file = 'semi_anns/instances_train2017.3@10.0.json'
# unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.3@10.0-unlabeled.json'
# labeled_dataset.ann_file = 'semi_anns/instances_train2017.4@10.0.json'
# unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.4@10.0-unlabeled.json'
# labeled_dataset.ann_file = 'semi_anns/instances_train2017.5@10.0.json'
# unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.5@10.0-unlabeled.json'

unlabeled_dataset.data_prefix = dict(img='train2017/')
train_dataloader = dict(dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# runtime
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook')]
