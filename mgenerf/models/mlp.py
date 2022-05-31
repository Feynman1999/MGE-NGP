import megengine as mge
from .registry import BACKBONES
import megengine.module as M
import megengine.functional as F


@BACKBONES.register_module
class MLP(M.Module):
    def __init__(self,  num_layers, 
                        hidden_dim, 
                        geo_feat_dim, 
                        num_layers_color,
                        hidden_dim_color, **kwargs):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        # get input_ch and input_ch_views from kwargs
        self.input_ch = kwargs.get('input_ch', None)
        self.input_ch_views = kwargs.get('input_ch_views', None)

        assert self.input_ch is not None
        assert self.input_ch_views is not None

        self.sigma_net = []

        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 features for color
            else:
                out_dim = hidden_dim
            
            self.sigma_net.append(M.Linear(in_dim, out_dim, bias=False))

        self.color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 for rgb
            else:
                out_dim = hidden_dim
            
            self.color_net.append(M.Linear(in_dim, out_dim, bias=False))

    def forward(self, x):
        input_pts, input_views = F.split(x, [self.input_ch, self.input_ch_views], axis=-1)

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h)

        sigma, geo_feat = h[..., 0:1], h[..., 1:]
        
        # color
        h = F.concat([input_views, geo_feat], axis=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h)
            
        color = h
        outputs = F.concat([color, sigma], axis = -1)

        return outputs
