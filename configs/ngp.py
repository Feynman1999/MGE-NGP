import logging

embedder_view = dict(
    type = "PositionalEncoding",
    multires = 4,  #  log2 of max freq for positional encoding (2D direction)
    log_sampling = True, 
    include_input = True
)

# model settings
model = dict(
    type="Coarse_Fine_Nerf",
    coarse_net = dict(
        type = 'NGP',
        implicit_net = dict(
            type = "MLP",
            layers = 4, 
            channels = 128,
            input_dim = ,
            output_dim = ,
        ),
        hash_net = dict(

        ),
        embedder_view = embedder_view
    ),
    fine_net = dict(
        type = 'NGP',
        implicit_net = dict(
            type = "MLP",
            layers = 4, 
            channels = 128,
            input_dim = ,
            output_dim = ,
        ),
        hash_net = dict(
            
        ),
        embedder_view = embedder_view
    )
)

train_cfg = dict(
    near = 0.3,
    far = 10,
)

test_cfg = dict()

# dataset settings
dataset_type = "KittiDataset"
data_root = r"C:\Users\76397\Desktop\taichi\kitti\00"

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
        mode = 'train'
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        pipeline=test_pipeline,
        mode = 'eval'
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        pipeline=test_pipeline,
        mode = 'test'
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
