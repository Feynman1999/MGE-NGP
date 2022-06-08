import logging

embedder_view = dict(
    type = "SHEncoding",
    input_dim = 3, 
    degree = 4
)

bounding_box = [[-15, -5, 0], [15, 20, 80]]

# model settings
model = dict(
    type="Coarse_Fine_Nerf",
    coarse_net = dict(
        type = 'NGP',
        hash_net = dict(
            type = "HashEncoding",
            bounding_box = bounding_box, # 空间大小
            finest_resolution = 1024,
            log2_hashmap_size = 19,
        ),
        implicit_net = dict(
            type = "MLP",
            num_layers = 2,
            hidden_dim = 64,
            geo_feat_dim=15,
            num_layers_color=3,
            hidden_dim_color = 64
        ),
        embedder_view = embedder_view
    ),
    fine_net = dict(
        type = 'NGP',
        hash_net = dict(
            type = "HashEncoding",
            bounding_box = bounding_box, # 空间大小
            finest_resolution = 1024,
            log2_hashmap_size = 19,
        ),
        implicit_net = dict(
            type = "MLP",
            num_layers = 2,
            hidden_dim = 64,
            geo_feat_dim=15,
            num_layers_color=3,
            hidden_dim_color = 64
        ),
        embedder_view = embedder_view
    )
)

train_cfg = dict(
    near = 0.1,
    far = 30,
    bounding_box = bounding_box,
    N_samples = 128,  # number of coarse samples per ray
    N_importance = 128,  # number of additional fine samples per ray
    retraw=True,
    lindisp=False,
    perturb=1.,
    sparse_loss_weight = 1e-10,
    tv_loss_weight = 1e-6
)

test_cfg = dict()

# dataset settings
dataset_type = "KittiTrainingDataset"
data_root =  r"/data/dataset-3840/datasets/kitti/00" # "/data/dataset-3840/datasets/kitti/00" #

train_pipeline = [
    
]

test_pipeline = [
    
]

data = dict(
    samples_per_gpu=8, # 4 * 1024
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        pipeline=train_pipeline,
        rays_per_sample = 1024,
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

paramwise_cfg=dict(custom_keys={'implicit_net': dict(weight_decay=1e6)})
optimizer = dict(type='AdamW', lr=0.005, betas=(0.9, 0.99), eps = 1e-15, weight_decay=1e-12, paramwise_cfg = paramwise_cfg)


lr_config = dict(
    type="one_cycle", max_lr=0.005, div_factor=10.0, pct_start=0.1, final_div_factor = 1e2
)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50000,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

total_epochs = 5
log_level = "INFO"
work_dir = "./workdirs/ngp"
load_from = None 
resume_from = None
workflow = [("train", 1),]
