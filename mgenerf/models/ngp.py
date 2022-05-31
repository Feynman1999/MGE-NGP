import megengine as mge
from .registry import NERFS
import megengine.module as M


@NERFS.register_module
class NGP(M.Module):
    def __init__(self, ):
        super(NGP, self).__init__()