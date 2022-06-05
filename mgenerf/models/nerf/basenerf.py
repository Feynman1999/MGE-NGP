import megengine.module as M
from abc import ABCMeta, abstractmethod

class Base_Nerf(M.Module):
    def __init__(self, train_cfg, test_cfg):
        super(Base_Nerf, self).__init__()
        # initial all modules
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, x):
        pass