# Brain Tumor Dataset
dataset_type = "CocoDataset"
data_root = '/root/datasets/BrainTumor/'

backend_args = None

metainfo = dict(
    classes=(
        "label0",
        "label1",
        "label2",
    ),
    palette=[
        (255, 192, 0),
        (0, 176, 240),
        (255, 0, 0),
    ],
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs",
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

batch_size = 4
num_workers = 4
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img="train2017/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

# val and test settings
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# voc evaluator
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
