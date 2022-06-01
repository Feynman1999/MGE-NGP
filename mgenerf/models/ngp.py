import megengine as mge
from .registry import BACKBONES
import megengine.module as M
from .builder import build_backbone


@BACKBONES.register_module
class NGP(M.Module):
    def __init__(self, log2_hashmap_size, finest_res, hash_net, implicit_net, embedder_view):
        super(NGP, self).__init__()

        self.log2_hashmap_size = log2_hashmap_size
        self.finest_res = finest_res
        self.hash_net = build_backbone(hash_net)
        self.implicit_net = build_backbone(implicit_net)
        self.embedder_view  =  build_backbone(embedder_view)


    def forward(x):
        pass