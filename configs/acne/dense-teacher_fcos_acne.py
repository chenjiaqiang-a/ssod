custom_imports = dict(
    imports=['ssod.detectors.dense_teacher'],
    allow_failed_imports=False)

_base_ = [
    '../_base_/models/fcos_r50_fpn.py',
    '../_base_/datasets/semi_acne_detection.py',
    '../_base_/default_runtime.py',
]

# model
detector = _base_.model
detector.data_preprocessor.mean = [150.058, 110.453, 95.142]
detector.data_preprocessor.std = [52.055, 44.240, 41.592]
detector.bbox_head.num_classes = 10

model = dict(
    _delete_=True,
    type='DenseTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        burn_in_steps=3000,
        distill_ratio=0.01,
        cls_weight=4.0,
        reg_weight=1.0,
        centerness_weight=1.0,
        warmup_steps=3000,
        suppress='linear'),
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

# remove geometric augmentation
weak_pipeline = [
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]
strong_pipeline = [
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type='RandAugment', aug_space=_base_.color_space, aug_num=1),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]
unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadEmptyAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='MultiBranch',
        branch_field=_base_.branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline)
]

unlabeled_dataset.data_prefix = dict(img='train2017/')
unlabeled_dataset.pipeline = unsup_pipeline
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
    optimizer=dict(
        type='SGD', lr=0.002, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

# runtime
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook')]
