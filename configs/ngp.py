import logging

# model settings
model = dict(
    type="NGP",
    
)

train_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = "KittiDataset"
data_root = "/data/"

train_pipeline = [
    
]

test_pipeline = [
    
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        pipeline=test_pipeline,
    ),
)

optimizer = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999), weight_decay=2e-6)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    type="one_cycle", lr_max=0.004, div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

total_epochs = 20
log_level = "INFO"
work_dir = "./experiments/pointpillars"
load_from = None 
resume_from = None
workflow = [("train", 1), ("val", 1)]
