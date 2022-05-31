import megengine as mge
from .registry import BACKBONES
import megengine.module as M


@BACKBONES.register_module
class NGP(M.Module):
    def __init__(self, ):
        super(NGP, self).__init__()