from cv2 import VideoWriter
import megengine as mge
from .registry import BACKBONES
import megengine.module as M
from .builder import build_backbone
import megengine.functional as F


@BACKBONES.register_module
class NGP(M.Module):
    def __init__(self, hash_net, implicit_net, embedder_view):
        super(NGP, self).__init__()

        self.hash_net = build_backbone(hash_net)
        self.embedder_view  =  build_backbone(embedder_view)
        
        # 根据一些参数 计算mlp模型的一些输入维度 并更新字典
        implicit_net.update({
            'input_ch' : self.hash_net.out_dim,
            'input_ch_views': self.embedder_view.out_dim,
        })
        self.implicit_net = build_backbone(implicit_net)
        

    def forward(self, locations, viewdirs):
        # given location and viewdirs, return rgb and theta
        # [num_rays, num_samples, 3]   [num_rays, 3]
        """
            locations: [num_rays, num_samples, 3]
            viewdirs: [num_rays, 3]
            return: [num_rays, num_samples along ray, 4]
        """
        num_rays, num_samples, x = locations.shape
        locations = F.reshape(locations, (-1, x))
        locations = self.hash_net(locations)
        viewdirs = self.embedder_view(viewdirs) # [num_rays, dim]
        viewdirs = F.broadcast_to(F.expand_dims(viewdirs, axis=1), (num_rays, num_samples, self.embedder_view.out_dim))
        viewdirs = F.reshape(viewdirs, (-1, self.embedder_view.out_dim))
        output = self.implicit_net(locations, viewdirs)
        return output.reshape((num_rays, num_samples, output.shape[-1]))