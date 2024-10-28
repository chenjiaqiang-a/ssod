_base_ = [
    '../_base_/models/fcos_r50_fpn.py',
    '../_base_/datasets/acne_detection.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    data_preprocessor=dict(
        mean=[142.566, 104.840, 90.092],
        std=[78.085, 62.782, 57.795],
    ),
    bbox_head=dict(num_classes=10),
)

# dataset
# supervised for 10% labeled acne data
train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.1@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.2@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.3@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.4@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.5@10.0.json'))

# schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[18],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.002, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))
