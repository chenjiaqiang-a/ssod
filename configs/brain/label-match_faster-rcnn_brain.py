custom_imports = dict(
    imports=['ssod.detectors.label_match'],
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
    mean=[22.461, 20.434, 20.892],
    std=[27.164, 23.229, 23.586],
    bgr_to_rgb=True,
    pad_size_divisor=32)
detector.roi_head.bbox_head.num_classes = 3

data_root = _base_.data_root
labeled_ann_file = 'semi_anns/instances_train2017.1@10.0.json'
unlabeled_ann_file = 'semi_anns/instances_train2017.1@10.0-unlabeled.json'
# labeled_ann_file = 'semi_anns/instances_train2017.2@10.0.json'
# unlabeled_ann_file = 'semi_anns/instances_train2017.2@10.0-unlabeled.json'
# labeled_ann_file = 'semi_anns/instances_train2017.3@10.0.json'
# unlabeled_ann_file = 'semi_anns/instances_train2017.3@10.0-unlabeled.json'
# labeled_ann_file = 'semi_anns/instances_train2017.4@10.0.json'
# unlabeled_ann_file = 'semi_anns/instances_train2017.4@10.0-unlabeled.json'
# labeled_ann_file = 'semi_anns/instances_train2017.5@10.0.json'
# unlabeled_ann_file = 'semi_anns/instances_train2017.5@10.0-unlabeled.json'

model = dict(
    _delete_=True,
    type='LabelMatch',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        labeled_ann_file=data_root + labeled_ann_file,
        cls_pseudo_thr=0.9,
        queue_num=32 * 100,
        reliable_percent=0.2,
        act_update_interval=1,
        use_rplm=True,
        rplm_score=0.8,
        rplm_iou=0.8,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

# semi-supervised for 10% labeled brain tumor data
labeled_dataset.ann_file = labeled_ann_file
unlabeled_dataset.ann_file = unlabeled_ann_file

unlabeled_dataset.data_prefix = dict(img='train2017/')
train_dataloader = dict(dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# runtime
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook')]
