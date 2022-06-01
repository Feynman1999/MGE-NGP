from ..registry import NERFS
from ..builder import build_backbone
import megengine as mge
import megengine.module as M
from .basenerf import Base_Nerf


@NERFS.register_module
class Coarse_Fine_Nerf(Base_Nerf):
    def __init__(self, coarse_net, fine_net, train_cfg, test_cfg):
        super(Coarse_Fine_Nerf, self).__init__(train_cfg=train_cfg, test_cfg=test_cfg)
        
        self.coarse_net = build_backbone(coarse_net)
        self.fine_net = build_backbone(fine_net)

    def train_step(self, batchdata, now_epoch, now_iter):
        pass


    def test_step(self, batchdata, **kwargs):
        pass

    def cal_for_eval(self):
        pass
