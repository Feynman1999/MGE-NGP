

# dataset settings
dataset_type = "KittiTestingDataset"
data_root =  r"C:\Users\76397\Desktop\taichi\kitti/00" # "/data/dataset-3840/datasets/kitti/00" #


test_pipeline = [
    
]

data = dict(
    samples_per_gpu=1, # 4 * 1024
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        root_path=data_root,
        pipeline=test_pipeline
    ),
)

log_level = "INFO"