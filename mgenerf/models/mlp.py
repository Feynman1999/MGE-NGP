import megengine as mge
from .registry import BACKBONES
import megengine.module as M


@BACKBONES.register_module
class MLP(M.Module):
    def __init__(self, ):
        super(MLP, self).__init__()