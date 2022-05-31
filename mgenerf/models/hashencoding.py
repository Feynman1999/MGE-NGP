import megengine as mge
from .registry import BACKBONES
import megengine.module as M


@BACKBONES.register_module
class HashEncoding(M.Module):
    def __init__(self, ):
        super(HashEncoding, self).__init__()