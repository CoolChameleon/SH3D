from locale import locale_alias
from grpc import local_channel_credentials
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace

class OccupancyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg["network"]['num_layers']
        hidden_size = cfg["network"]['hidden_dim']
        self.octaves_pe_points = cfg["network"]['octaves_pe_points']
        self.octaves_pe_views = cfg["network"]['octaves_pe_views']
        self.skips = cfg["network"]['skips']
        self.rescale = cfg["network"]['rescale']
        self.feat_size = cfg["network"]['feat_size']
        geometric_init = cfg["network"]['geometric_init']
        self.loacl_latent_dim = cfg["network"]["latent_dim"]
        self.global_latent_dim = cfg["network"]["global_latent_dim"]
        bias = cfg["network"]["bias"]
        self.latent_dim = self.loacl_latent_dim + self.global_latent_dim
        dim = 3
        dim_embed = dim * self.octaves_pe_points * 2 + dim
        dim_embed_view = dim + dim * self.octaves_pe_views * 2 + dim + dim + self.feat_size + self.loacl_latent_dim

        self.transform_points = PositionalEncoding(L=self.octaves_pe_points)
        self.transform_views = PositionalEncoding(L=self.octaves_pe_views)

        dims_geo = [dim_embed + self.latent_dim] + [hidden_size for _ in range(self.num_layers)] + [1 + self.feat_size]

        self.num_layers = len(dims_geo)

        for l in range(self.num_layers - 1):
            if l + 1 in self.skips:
                out_dim = dims_geo[l + 1] - dims_geo[0]
            else:
                out_dim = dims_geo[l + 1]
            
            lin = nn.Linear(dims_geo[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_geo[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif self.octaves_pe_points > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.octaves_pe_points > 0 and l in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_geo[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        
        self.softplus = nn.Softplus(beta=100)
        # set_trace()
        dims_view = [dim_embed_view] + [hidden_size for _ in range(0, 4)] + [3]

        self.num_layers_app = len(dims_view)

        for l in range(0, self.num_layers_app - 1):
            out_dim = dims_view[l + 1]
            lina = nn.Linear(dims_view[l], out_dim)
            lina = nn.utils.weight_norm(lina)
            setattr(self, "lina" + str(l), lina)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def infer_occupancy(self, p, latent):
        pe = self.transform_points(p / self.rescale)
        # print(latent.shape, "latent")
        latent = latent.permute(1, 2 ,0)
        x = torch.cat([pe, latent], -1)
        for l in range(self.num_layers - 1):
            lin = getattr(self, "lin{}".format(l))
            if l in self.skips:
                x = torch.cat([x, pe, latent], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x
    
    def gradient(self, p, local_latent, global_latent,**kwargs):
        with torch.enable_grad():
            _, n_rayss, n_pointss = local_latent.shape
            global_latent = global_latent.unsqueeze(1).unsqueeze(2).repeat(1, n_rayss, n_pointss)
            latent = torch.cat((local_latent, global_latent), dim=0)            
            p.requires_grad_(True)
            y = self.infer_occupancy(p, latent)[..., :1]
            if torch.isnan(y).sum() > 0:
                set_trace()
            d_output = torch.ones_like(
                y,
                requires_grad=False,
                device=y.device
            )
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )
            if len(gradients[0].unsqueeze(1).shape) != 4:
                set_trace()
            return gradients[0].unsqueeze(1)

    def forward(self, points, local_latent, ray_dirs=None, return_occupancy=True, return_feature_vec=False, return_rgb=False, global_latent=None, **kwargs):
        """
        Args:
            points: torch.tensor([n_rays, n_points, 3])
            latent: torch.tensor([latent_dim,n_rays, n_points])
            return_occupancy: bool. If true, return occupancy. Otherwise, return None.
            return_feature_vec: bool. If true, return feature_vec. Otherwise, return None.
            return_rgb: bool. If true, calculate rgb using ray_dirs and return. Otherwise, return None.
            ray_dirs: torch.tensor([n_rays, 3]) or None. Only used when return_rgb is true.
        
        Output:
            occupancy: torch.tensor([n_rays, n_points, 1]) or None
            feature_vec: torch.tensor([n_rays, n_points, feature_vec_dim]) or None.
            rgb: torch.tensor([n_rays, n_points, 3]) or None
        """
        # set_trace()
        _, n_rayss, n_pointss = local_latent.shape
        global_latent_r = global_latent.unsqueeze(1).unsqueeze(2).repeat(1, n_rayss, n_pointss)
        latent = torch.cat((local_latent, global_latent_r), dim=0)
        x = self.infer_occupancy(points, latent)
        if return_occupancy:
           occupancy = self.sigmoid(x[..., :1] * -10.0)
        #    set_trace()
           rgb = 0.0
        if return_rgb:
            input_views = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
            input_views = self.transform_views(input_views)
            normals = self.gradient(points, local_latent, global_latent)
            # set_trace()
            rgb = self.infer_app(points, normals, input_views, x[..., 1:], local_latent.permute(1, 2, 0))
            # rgb = self.infer_app(points, normals, input_views, latent.permute(1,2,0))
        return occupancy, x[..., 1:], rgb 

    def infer_app(self, points, normals, view_dirs, feature_vectors, latent):
        rendering_input = torch.cat([points, view_dirs.unsqueeze(0).permute(1,0,2), normals.squeeze(-3), feature_vectors, latent], dim=-1)
        x = rendering_input
        for l in range(0, self.num_layers_app - 1):
            lina = getattr(self, "lina" + str(l))
            x = lina(x)
            if l < self.num_layers_app - 2:
                x = self.relu(x)
        # set_trace()
        x = self.tanh(x) * 0.5 + 0.5
        return x
        

class PositionalEncoding(object):
    def __init__(self, L=10):
        self.L = L
    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p), 
             torch.cos((2 ** i) * pi * p)],
             dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)
