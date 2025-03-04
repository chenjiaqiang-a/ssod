_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/acne_detection.py',
    '../_base_/schedules/schedule_15e.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    data_preprocessor=dict(
        mean=[150.058, 110.453, 95.142],
        std=[52.055, 44.240, 41.592],
    ),
    roi_head=dict(bbox_head=dict(num_classes=10)),
)

# dataset
# supervised for 10% labeled acne data
train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.1@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.2@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.3@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.4@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.5@10.0.json'))

# runtime
default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))
