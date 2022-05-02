from ipdb import set_trace
import torch
import torch.nn.functional as F
class AngleWeightedfeatureaggregation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, sampled_ray_dirs, camera_source_world, points,  features):
        # set_trace()
        #points:[1024, 256,3] sampled_ray_dirs:[1,1024,3] camera_source_world:[49,1,3] features:[49,32,1024,256]
    #    set_trace()
        n_rays, n_pointsonray, _ = points.shape
        points = points.unsqueeze(0).reshape(1,-1,3)
        _, n_points, _ = points.shape
        n_views = camera_source_world.shape[0]
        points = points.repeat(n_views,1,1)
        camera_source_world = camera_source_world.repeat(1,n_points, 1)
        ray_vector = points - camera_source_world
        sampled_ray_dirs = sampled_ray_dirs.repeat(n_views, n_pointsonray, 1) 

        ray_vectors = ray_vector / ray_vector.norm(2, 2).unsqueeze(-1)
     
        ray_dir_dot_prods = (sampled_ray_dirs * ray_vectors).sum(dim = -1)

        angle_weight = (ray_dir_dot_prods ) *0.5 + 0.5
    #    set_trace()
        n_dim = features.shape[1]
        features = features.permute(0,2,3,1).reshape(n_views, -1, n_dim)
        feature_agg = (features * angle_weight[..., None]).sum(dim = 0) / angle_weight[..., None].sum(dim = 0 ).clamp(
        0.01)
        feature_agg = feature_agg.reshape(n_rays, n_pointsonray,n_dim)
    #    set_trace()
        return feature_agg.permute(2,0,1)



