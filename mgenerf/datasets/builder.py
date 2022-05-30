from mgenerf.utils import build_from_cfg
from .registry import DATASETS
from megengine.data import SequentialSampler, RandomSampler, DataLoader


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        raise NotImplementedError("")
    elif cfg["type"] == "RepeatDataset":
        raise NotImplementedError("")
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset, batch_size, workers_per_gpu, shuffle = True):
    if shuffle:
        sampler = RandomSampler(dataset, batch_size, drop_last=True)
    else:
        sampler = SequentialSampler(dataset, batch_size, drop_last=False)
    data_loader = DataLoader(dataset, sampler, num_workers=workers_per_gpu)
    
    return data_loader
