custom_imports = dict(
    imports=['ssod.detectors.mean_teacher'],
    allow_failed_imports=False)

_base_ = [
    '../_base_/models/fcos_r50_fpn.py',
    '../_base_/datasets/semi_acne_detection.py',
    '../_base_/default_runtime.py',
]

# model
detector = _base_.model
detector.data_preprocessor.mean = [142.566, 104.840, 90.092]
detector.data_preprocessor.std = [78.085, 62.782, 57.795]
detector.bbox_head.num_classes = 10

model = dict(
    _delete_=True,
    type='MeanTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        cls_pseudo_thr=0.5,
        burn_in_steps=3000,
        filter_empty_image=False,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

# semi-supervised for 10% labeled acne data
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
train_dataloader = dict(
    batch_size=8,
    sampler=dict(batch_size=8, source_ratio=[4, 4]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# schedule
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=10000,
        by_epoch=False,
        milestones=[8000],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

# runtime
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook')]
