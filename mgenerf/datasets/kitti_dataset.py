
from mgenerf.datasets.registry import DATASETS
from mgenerf.datasets.base_dataset import BaseDataset

@DATASETS.register_module
class KittiDataset()