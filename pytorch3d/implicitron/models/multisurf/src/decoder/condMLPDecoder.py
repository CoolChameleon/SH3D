import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace

from .unisurfDecoder import PositionalEncoding

class condMLPDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.global_latent_dim = cfg["network"]["global_latent_dim"]
        self.num_heads = cfg["network"]["num_heads"]
        self.num_layers = cfg["network"]["num_layers"]

        self.hidden_size = cfg["network"]["hidden_dim"]
        self.octaves_pe_points = cfg["network"]['octaves_pe_points']
        self.octaves_pe_views = cfg["network"]['octaves_pe_views']
        self.skips = cfg["network"]['skips']
        self.feat_size = cfg["network"]["feat_size"]
        geometric_init = cfg["network"]["geometric_init"]
        self.local_latent_dim = cfg["network"]["latent_dim"]
        self.rescale = 1
        bias = cfg["network"]["bias"]
        self.latent_dim = self.local_latent_dim + self.global_latent_dim
        dim = 3
        dim_embed = dim * self.octaves_pe_points * 2 + dim
        dim_embed_view = dim + dim * self.octaves_pe_views * 2 + dim + dim + self.feat_size + self.local_latent_dim

        dims_geo = [dim_embed + self.latent_dim] + [self.hidden_size for _ in range(self.num_layers)] + [1 + self.feat_size]

        self.num_layers = len(dims_geo)

        self.transform_points = PositionalEncoding(L=self.octaves_pe_points)
        self.transform_views = PositionalEncoding(L=self.octaves_pe_views)

        for l in range(self.num_layers - 1):
            lin = nn.Linear(self.global_latent_dim, self.num_heads)
            setattr(self, "linp_l{}".format(l), lin)
        
        for l in range(self.num_layers - 1):
            if l + 1 in self.skips:
                out_dim = dims_geo[l + 1] - dims_geo[0]
            else:
                out_dim = dims_geo[l + 1]
            
            if geometric_init:
                weight_init = torch.zeros([out_dim, dims_geo[l]])
                bias_init = torch.zeros([out_dim])
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(weight_init, mean=np.sqrt(np.pi) / np.sqrt(dims_geo[l]), std=0.0001)
                    torch.nn.init.constant_(bias_init, -bias)
                elif self.octaves_pe_points > 0 and l == 0:
                    torch.nn.init.constant_(bias_init, 0.0)
                    torch.nn.init.constant_(weight_init[:, 3:], 0.0)
                    torch.nn.init.normal_(weight_init[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.octaves_pe_points > 0 and l in self.skips:
                    torch.nn.init.constant_(bias_init, 0.0)
                    torch.nn.init.normal_(weight_init, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(weight_init[:, -(dims_geo[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(bias_init, 0.0)
                    torch.nn.init.normal_(weight_init, 0.0, np.sqrt(2) / np.sqrt(out_dim))  

                for h in range(self.num_heads):
                    lin_weight = nn.Parameter(weight_init.clone().detach().requires_grad_(True))
                    lin_bias = nn.Parameter(bias_init.clone().detach().requires_grad_(True))

                    setattr(self, "lin_l{}_h{}_w".format(l, h), lin_weight)
                    setattr(self, "lin_l{}_h{}_b".format(l, h), lin_bias)
            else:
                for h in range(self.num_heads):
                    weight_init = torch.randn([out_dim, dims_geo[l]])
                    bias_init = torch.randn([out_dim])
                    lin_weight = nn.Parameter(weight_init.clone().detach().requires_grad_(True))
                    lin_bias = nn.Parameter(bias_init.clone().detach().requires_grad_(True))

                    setattr(self, "lin_l{}_h{}_w".format(l, h), lin_weight)
                    setattr(self, "lin_l{}_h{}_b".format(l, h), lin_bias)                 

        self.softplus = nn.Softplus(beta=100)

        dims_view = [dim_embed_view] + [self.hidden_size for i in range(0, 4)] + [3]

        self.num_layers_app = len(dims_view)

        for l in range(0, self.num_layers_app - 1):
            out_dim = dims_view[l + 1]
            lina = nn.Linear(dims_view[l], out_dim)
            lina = nn.utils.weight_norm(lina)
            setattr(self, "lina" + str(l), lina)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()    
        self.softmax = nn.Softmax(dim=0)    

    def infer_sdf(self, points, local_latent, global_latent):
        input = self._process_input(points, local_latent, global_latent)
        x = input
        for l in range(self.num_layers - 1):
            lin_p = getattr(self, "linp_l{}".format(l))
            param = self.softmax(lin_p(global_latent))
            weight_list = [getattr(self, f"lin_l{l}_h{h}_w") for h in range(self.num_heads)]
            bias_list = [getattr(self, f"lin_l{l}_h{h}_b") for h in range(self.num_heads)]

            weight = sum([param[h] * weight_list[h] for h in range(self.num_heads)])
            bias = sum([param[h] * bias_list[h] for h in range(self.num_heads)])

            if l in self.skips:
                x = torch.cat([x, input], -1) / np.sqrt(2)
                x = F.linear(x, weight, bias)
                # print("weight",weight,l)
            else:
                x = F.linear(x, weight, bias)
                # print("weight",weight,l)
                # print("bias", bias, l)
            # set_trace()  

            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        return x

    def _process_input(self, points, local_latent, global_latent):
        # set_trace()
        _, n_rayss, n_pointss = local_latent.shape
        pe = self.transform_points(points / self.rescale)
        local_latent = local_latent.permute(1, 2, 0)
        global_latent_r = global_latent.unsqueeze(1).unsqueeze(2).repeat(1, n_rayss, n_pointss).permute(1,2,0)
        # set_trace()
        return torch.cat([pe, local_latent, global_latent_r], dim=-1)

    def infer_app(self, points, normals, view_dirs, feature_vectors, latent):
        rendering_input = torch.cat([points, view_dirs.unsqueeze(0).permute(1,0,2), normals.squeeze(-3), feature_vectors,latent], dim=-1)
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

    def forward(self, points, local_latent, ray_dirs=None, return_occupancy=True, return_feature_vec=False, return_rgb=False, global_latent=None, **kwargs):
        # set_trace()
        output = self.infer_sdf(points, local_latent, global_latent)

        occupancy, feature_vec, rgb = None, None, None

        if return_occupancy:
            occupancy = self.sigmoid(output[..., :1] * -10.0)

        if return_rgb:
            input_views = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
            input_views = self.transform_views(input_views)
            normals = self.gradient(points, local_latent=local_latent, global_latent=global_latent)
            rgb = self.infer_app(points, normals, input_views, output[..., 1:], local_latent.permute(1, 2, 0))
        
        if return_feature_vec:
            feature_vec = output[..., 1:]
        return occupancy, feature_vec, rgb        

    def get_mh_loss(self):
        # set_trace()
        mh_loss = 0
        for l in range(self.num_layers - 1):
            weight_list = [getattr(self, f"lin_l{l}_h{h}_w") for h in range(self.num_heads)]
            all_weight = torch.stack(weight_list)
            # mh_loss += all_weight.var(dim=0).norm()
            mh_loss += self.sigmoid(all_weight.var(dim=0).mean())

            bias_list = [getattr(self, f"lin_l{l}_h{h}_b") for h in range(self.num_heads)]    
            all_bias = torch.stack(bias_list)
            # mh_loss += all_bias.var(dim=0).norm()
            mh_loss += self.sigmoid(all_bias.var(dim=0).mean())
 
        # return self.sigmoid(mh_loss)
        return mh_loss / ((self.num_layers - 1) * 2) 
        
