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
	
    @abstractmethod
    def train_step(self, batchdata, now_epoch, now_iter):
        """All subclasses should overwrite this function"""

    @abstractmethod
    def test_step(self, batchdata, **kwargs):
        """All subclasses should overwrite this function"""

    @abstractmethod
    def cal_for_eval(self):
        """All subclasses should overwrite this function"""
