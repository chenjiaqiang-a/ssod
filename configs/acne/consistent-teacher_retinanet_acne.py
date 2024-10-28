custom_imports = dict(
    imports=[
        'ssod.detectors.consistent_teacher',
        'ssod.dense_heads.fam3d_head',
        'ssod.task_modules.assigners.adaptive_sample_assigner',
    ],
    allow_failed_imports=False)

_base_ = [
    '../_base_/datasets/semi_acne_detection.py',
    '../_base_/default_runtime.py',
]

# model
detector = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[142.566, 104.840, 90.092],
        std=[78.085, 62.782, 57.795],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='FAM3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(type='AdaptiveSampleAssigner', topk=13, iou_weight=2.0),
        sampler=dict(type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

model = dict(
    type='ConsistentTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector['data_preprocessor']),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        burn_in_steps=3000,
        scores_queue_size=100,
        dynamic_ratio=1.0,
        default_gmm_thr=0.0,
        gmm_thr_policy='high',
        cls_pseudo_thr=0.0,
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
    optimizer=dict(
        type='SGD', lr=0.002, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

# runtime
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook')]
