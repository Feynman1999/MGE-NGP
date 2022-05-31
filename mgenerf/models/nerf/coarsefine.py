from ..registry import NERFS
from ..builder import build_backbone
import megengine as mge
import megengine.module as M


@NERFS.register_module
class Coarse_Fine_Nerf(M.Module):
    def __init__(self, coarse_net, fine_net):
        super(Coarse_Fine_Nerf, self).__init__()
        # initial all modules


    def train_step(self, batchdata, now_epoch, now_iter):
        pass


    def test_step(self, batchdata, **kwargs):
        pass

    def cal_for_eval(self):
        pass
