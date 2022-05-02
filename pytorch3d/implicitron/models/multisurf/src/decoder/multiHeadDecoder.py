import numpy as np
import torch
import torch.nn as nn
from ipdb import set_trace

from .unisurfDecoder import PositionalEncoding

class multiHeadDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg["network"]['num_layers']
        hidden_size = cfg["network"]['hidden_dim']
        self.hidden_size = hidden_size
        self.octaves_pe_points = cfg["network"]['octaves_pe_points']
        self.octaves_pe_views = cfg["network"]['octaves_pe_views']
        self.skips = cfg["network"]['skips']
        self.rescale = cfg["network"]['rescale']
        self.feat_size = cfg["network"]['feat_size']
        geometric_init = cfg["network"]['geometric_init']
        self.latent_dim = cfg["network"]["latent_dim"]
        bias = cfg["network"]["bias"]

        dim = 3
        dim_embed = dim * self.octaves_pe_points * 2 + dim
        dim_embed_view = dim + dim * self.octaves_pe_views * 2 + dim + dim + self.feat_size

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

        dims_view = [dim_embed_view]+ [ hidden_size for i in range(0, 4)] + [3]

        self.num_layers_app = len(dims_view)

        for l in range(0, self.num_layers_app - 1):
            out_dim = dims_view[l + 1]
            lina = nn.Linear(dims_view[l], out_dim)
            lina = nn.utils.weight_norm(lina)
            setattr(self, "lina" + str(l), lina)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.bias_dim = cfg["network"]["hidden_dim"]
        self.global_latent_dim = cfg["network"]["global_latent_dim"]
        self.bias_dim = self.bias_dim * 8
        self.bias_network = nn.Linear(self.global_latent_dim, self.bias_dim)

    def forward(self, points, local_latent, ray_dirs=None, return_occupancy=True, return_feature_vec=False, return_rgb=False, global_latent=None, **kwargs):
        output = self.infer_sdf(points, local_latent, global_latent)

        occupancy, feature_vec, rgb = None, None, None

        if return_occupancy:
            occupancy = self.sigmoid(output[..., :1] * -10.0)

        if return_rgb:
            input_views = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
            input_views = self.transform_views(input_views)
            normals = self.gradient(points, local_latent=local_latent, global_latent=global_latent)
            rgb = self.infer_app(points, normals, input_views, output[..., 1:])
        
        if return_feature_vec:
            feature_vec = output[..., 1:]
        return occupancy, feature_vec, rgb
        
    def infer_sdf(self, points, local_latent, global_latent, **kwargs):
        input = self._process_input(points, local_latent)
        bias = self._get_bias(global_latent)
        x = input
        for l in range(self.num_layers - 1):
            lin = getattr(self, "lin{}".format(l))
            if l in self.skips:
                x = torch.cat([x, input], -1) / np.sqrt(2)
                x = lin(x)
                x = x * bias[self.hidden_size *l: self.hidden_size*(l+1)]
            elif l + 1 in self.skips :
                x = lin(x)
            elif l == self.num_layers - 2:
                x = lin(x)
            else:
                x = lin(x)
                x = x * bias[self.hidden_size *l: self.hidden_size*(l+1)]
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def _process_input(self, points, local_latent):
        pe = self.transform_points(points / self.rescale)
        local_latent = local_latent.permute(1, 2, 0)
        return torch.cat([pe, local_latent], dim=-1)
        

    def _get_bias(self, global_latent):
        bias = self.bias_network(global_latent)
        return self.sigmoid(bias)

    def infer_app(self, points, normals, view_dirs, feature_vectors):
        rendering_input = torch.cat([points, view_dirs.unsqueeze(0).permute(1,0,2), normals.squeeze(-3), feature_vectors], dim=-1)
        x = rendering_input
        for l in range(0, self.num_layers_app - 1):
            lina = getattr(self, "lina" + str(l))
            x = lina(x)
            if l < self.num_layers_app - 2:
                x = self.relu(x)
        # set_trace()
        x = self.tanh(x) * 0.5 + 0.5
        return x

    def gradient(self, points, local_latent, global_latent, **kwargs):
        with torch.enable_grad():
            points.requires_grad_(True)
            y = self.infer_sdf(points, local_latent, global_latent)[..., :1]
            grad_outputs = torch.ones_like(
                y, requires_grad=False, device=y.device
            )
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=points,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )
            return gradients[0].unsqueeze(1)
