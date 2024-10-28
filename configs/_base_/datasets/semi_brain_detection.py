# Brain Tumor Dataset for SSOD
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

color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]
geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

branch_field = ['sup', 'unsup_teacher', 'unsup_student']
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs')),
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type='RandomFlip', prob=0.5),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]
# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(type='RandomFlip', prob=0.5),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]
# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs",
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

labeled_dataset = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline)
unlabeled_dataset = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file='annotations/instances_unlabeled2017.json',
    data_prefix=dict(img='unlabeled2017/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline)

batch_size = 8
num_workers = 4
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[4, 4]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

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
