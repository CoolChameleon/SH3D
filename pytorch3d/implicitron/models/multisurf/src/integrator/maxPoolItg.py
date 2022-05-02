from turtle import forward
import torch
import torch.nn as nn
import cv2
from utils.common import transform_to_camera_space
from ipdb import set_trace
import numpy as np
import torch.nn.functional as F

def maxpooling(features):
    features[features == 0] = -200000
    merge_feature, _ = features.max(dim=0)
    # set_trace()
    return merge_feature

class maxPoolItg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, all_latents):
        """
        Merge latents from all_views

        Args:
            all_latent: torch.tensor([n_views, n_rays, n_points, latent_dim])

        Output:
            latent: torch.tensor([n_rays, n_points, latent_dim])

        """
        # try:
        #     all_latents[all_latents == 0] = -200
        #     print(111111111111111111111111)
        # except:
        #     set_trace()
        # merge_feature, _ = all_latents.max(dim=0)
        merge_feature = all_latents.sum(dim=0)
        return merge_feature


class maxPoolGlobal(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    def forward(self, all_latents):
        """
        Merge latents from all_views

        Args:
            all_latent: torch.tensor([n_views, n_rays, n_points, latent_dim])

        Output:
            latent: torch.tensor([n_rays, n_points, latent_dim])
        """
        merge_feature, _ = all_latents.max(dim=0)
        return merge_feature
    
    def set_neighbor(self, neighbor_list):
        self.neighbor_list = neighbor_list