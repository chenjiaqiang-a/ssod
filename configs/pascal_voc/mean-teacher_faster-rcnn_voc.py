_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/semi_voc_detection.py',
    '../_base_/default_runtime.py',
]

# model
detector = _base_.model
detector.roi_head.bbox_head.num_classes = 20

model = dict(
    _delete_=True,
    type='SemiBaseDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=2.0,
        cls_pseudo_thr=0.7,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# schedule
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=36000, val_interval=2000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36000,
        by_epoch=False,
        milestones=[24000],
        gamma=0.1),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2),
)

# runtime
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0002)]
resume = True
