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
    lindisp=False,
    perturb=True,
    sparse_loss_weight = 1e-8,
    tv_loss_weight = 1e-8
)

test_cfg = dict()

# dataset settings
dataset_type = "KittiTrainingDataset"
data_root =  r"/data/dataset-3840/datasets/kitti/00" # "/data/dataset-3840/datasets/kitti/00" #

test_pipeline = [
    
]

data = dict(
    samples_per_gpu=8, # 4 * 1024
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        root_path=data_root,
        pipeline=test_pipeline,
        mode = 'test'
    ),
)

log_level = "INFO"
work_dir = "./workdirs/ngp_test"
load_from = None 
workflow = [("test", 1),]