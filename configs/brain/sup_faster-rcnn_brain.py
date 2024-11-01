_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/brain_detection.py',
    '../_base_/schedules/schedule_15e.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    data_preprocessor=dict(
        mean=[22.461, 20.434, 20.892],
        std=[27.164, 23.229, 23.586],
    ),
    roi_head=dict(bbox_head=dict(num_classes=3)),
)

# dataset
# supervised for 10% labeled brain tumor data
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.1@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.2@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.3@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.4@10.0.json'))
# train_dataloader = dict(dataset=dict(ann_file='semi_anns/instances_train2017.5@10.0.json'))
