import torch
import torch.nn as nn
import numpy as np
from ipdb import set_trace

class DepthFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *input):
        ray0, ray_direction, d_pred, p_pred, local_latent, global_latent, decoder, mask = input[:8]
        ctx.save_for_backward(ray0, ray_direction, d_pred, p_pred, local_latent, global_latent)
        ctx.decoder = decoder
        ctx.mask = mask
        return d_pred

    @staticmethod
    def backward(ctx, grad_output):
        ray0, ray_direction, d_pred, p_pred, local_latent, global_latent = ctx.saved_tensors
        decoder = ctx.decoder
        mask = ctx.mask
        eps = 1e-3

        with torch.enable_grad():
            p_pred.requires_grad = True
            # set_trace()
            f_p, _, _ = decoder(p_pred.permute(1,0,2), local_latent, ray_dirs=None, return_occupancy=True,return_feature_vec=False, return_rgb=False, global_latent=global_latent) 
            f_p_sum = f_p.sum()
            grad_p = torch.autograd.grad(f_p_sum, p_pred, retain_graph=True)[0]
            grad_p_dot_v = (grad_p * ray_direction).sum(-1)

            if mask.sum() > 0:
                grad_p_dot_v[mask == 0] = 1.
                # Sanity
                grad_p_dot_v[abs(grad_p_dot_v) < eps] = eps
                grad_outputs = -grad_output.squeeze(-1)
                grad_outputs = grad_outputs / grad_p_dot_v
                grad_outputs = grad_outputs * mask.float()

            # Gradients for latent code
            if local_latent is None or local_latent.shape[-1] == 0 or mask.sum() == 0:
                gradLatent = None
            else:
                gradLatent = torch.autograd.grad(f_p.squeeze(1).permute(1,0), local_latent, retain_graph=True,
                                            grad_outputs=grad_outputs)[0]

            # Gradients for network parameters phi
            if mask.sum() > 0:
                # set_trace()
                # Accumulates gradients weighted by grad_outputs variable
                grad_phi = torch.autograd.grad(
                    f_p.squeeze(1).permute(1,0), [k for k in decoder.parameters()],
                    grad_outputs=grad_outputs, retain_graph=True, allow_unused= True)
            else:
                grad_phi = [None for i in decoder.parameters()]

        # Return gradients for latent, z, and network parameters and None
        # for all other inputs
        out = [None, None, None, None, None, None, None, None] + list(grad_phi)
        return tuple(out)
