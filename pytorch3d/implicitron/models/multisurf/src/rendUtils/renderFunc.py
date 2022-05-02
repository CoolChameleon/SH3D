import torch
import numpy as np
import torch.nn as nn
from utils.common import get_mask, transform_to_camera_space_new
from ipdb import set_trace
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from .depthFunc import DepthFunction
from torch.nn import functional as F

class renderFunc():
    def __init__(self, cfg):
        self.num_grid_points_ray_marching = cfg["rendering"]["ray_marching_steps"]
        self.big_radius = cfg["rendering"]["radius"]
        self.n_sampled_pix = cfg["model"]["n_points_sampled"]
        self.n_grid_points = cfg["rendering"]["ray_marching_steps"]
        self.ada_start = cfg["rendering"]['interval_start']
        self.ada_end = cfg["rendering"]['interval_end']
        self.ada_grad = cfg["rendering"]['interval_decay']
        self.steps = cfg["rendering"]['num_points_in']
        self.steps_outside = cfg["rendering"]['num_points_out']
        self.ray_steps = cfg["rendering"]['ray_marching_steps']
        self.n_max_network_queries = cfg["rendering"]["n_max_network_queries"]
        self.depth_range = cfg["model"]["depth_range"]
        self.white_background = cfg["rendering"]["white_background"]
        self.render_outer_points = cfg["rendering"].get("render_outer_points", True)
        self.depth_func = DepthFunction()

    def set_params(self, world_mat, camera_mat, scale_mat, all_world_mats, all_camera_mats, all_scale_mats, neighbor_list, camera_all, camera, camera_selected):
        self.world_mat = world_mat
        self.camera_mat = camera_mat
        self.scale_mat = scale_mat
        self.selected_world_mats = all_world_mats[neighbor_list, :, :]
        self.selected_camera_mats = all_camera_mats[neighbor_list, :, :]
        self.selected_scale_mats = all_scale_mats[neighbor_list, :, :]
        self.device = all_camera_mats.device
        self.camera = camera.to(self.device)
        self.camera_all = camera_all.to(self.device)
        self.camera_selected = camera_selected.to(self.device)
        # set_trace()

        # 初始化相机在世界坐标系中的坐标
        self._get_coords()
        
    # def _get_coords(self):
    #     """
    #     Calculate the coordinates of the camera in the world coordinate system
    #     """
    #     # 注意这个origin_to_world和写在common里面那个不一样，这个不接受n_points这个参数，拷贝n份用来相减的那个过程我觉得从逻辑上来说不应该放在这
    #     # 好像也没什么别的事情要干，似乎可以直接一起写在set_params这个函数里面
    #     self.camera_world = origin_to_world(
    #         self.camera_mat, self.world_mat, self.scale_mat
    #     )
    #     self.selected_camera_world = origin_to_world(
    #         self.selected_camera_mats, self.selected_world_mats, self.selected_scale_mats
    #     )
    #     set_trace()
    def _get_coords(self):
        """
        Calculate the coordinates of the camera in the world coordinate system
        """
        # 注意这个origin_to_world和写在common里面那个不一样，这个不接受n_points这个参数，拷贝n份用来相减的那个过程我觉得从逻辑上来说不应该放在这
        # 好像也没什么别的事情要干，似乎可以直接一起写在set_params这个函数里面
        self.camera_world = self.camera.get_camera_center().cuda()
        self.selected_camera_world = self.camera_selected.get_camera_center().cuda().unsqueeze(1)
        # set_trace()
    def convert_pixels_to_rays(self, sampled_pixels):
        """
        Convert sampled pixel to ray_dirs

        Args:
            sampled_pixels: torch.tensor([1, n_points, 2])

        Output:
            sampled_ray_dirs: torch.tensor([n_points, 3])
        """
        # 把输入的点的像素坐标转换成世界坐标，然后和self.camera_world相减后获得ray_dir
        # 这个函数还没测试，现在只是伪代码
        
        _, n_points, _ = sampled_pixels.shape
        xy_1 = torch.ones(1, n_points, 3).to(self.device)
        xy_1[:,:,:2] = sampled_pixels
        pixels_world = self.camera.unproject_points(xy_1)
        # set_trace()
        # aaa = transform_to_camera_space_new(pixels_world, self.camera)
        # set_trace()
        # pixels_world = image_points_to_world(
        #     sampled_pixels, self.camera_mat, self.world_mat, self.scale_mat
        # )
        camera_world = self.camera_world.unsqueeze(0).repeat(1, n_points, 1)
        ray_vector = pixels_world - camera_world
        sampled_ray_dirs = ray_vector / ray_vector.norm(2, 2).unsqueeze(-1)
        self.sampled_ray_dirs = sampled_ray_dirs
        
        return sampled_ray_dirs.squeeze(0), self.camera_world

    def find_surface(self, sampled_ray_dirs):
        depth_interval = self._sphere_intersect(sampled_ray_dirs, r = self.big_radius)
        surface_depth, surface_points, object_mask = self._ray_marching(sampled_ray_dirs, depth_interval)
        # set_trace()
        return surface_depth, surface_points, object_mask, depth_interval 
        
    def _ray_marching(self, sampled_ray_dirs, depth_interval):
        """
        Apply ray_marching to find surfaces on sampled_ray_dirs

        Args:
            sampled_ray_dirs: torch.tensor([n_rays, 3])
            depth_interval: torch.tensor([n_rays, 2])
        
        Output:
            surface_depths: torch.tensor([n_rays, 1])
            surface_points: torch.tensor([n_rays, 3])
            object_mask: torch.tensor([n_rays, 1], dtype=np.bool)
        """
        # self.num_grid_points_ray_marching是超参，表示ray_marching第一步打格点的过程中，每条射线选多少个点。要从cfg里读

        grid_points, grid_points_depths = self._sample_grid_points(sampled_ray_dirs, depth_interval, self.num_grid_points_ray_marching)

        all_local_latents = self._interpolate_latent(grid_points)
        # set_trace()
        local_latent = self.local_integrator(self.sampled_ray_dirs, self.selected_camera_world, grid_points, all_local_latents)

        global_latent = self.global_integrator(self.global_features)

        # latent = self._get_latent(local_latent, global_latent)

        occupancy_0, _, _ = self.decoder(grid_points, local_latent, return_occupancy=True, return_feature_vec=False, return_rgb=False, ray_dirs=None, global_latent=global_latent)
        occupancy = occupancy_0 - 0.5
        

        f_low, f_high, d_low, d_high, ray0_masked, ray_direction_masked, object_mask, mask_0_not_occupied = self._get_secant_interval(occupancy, grid_points_depths, sampled_ray_dirs)

        # if object_mask[object_mask == 0].shape[0] > 0:
        #     set_trace()
        surface_depths, surface_points, mask = self._secant(f_low, f_high, d_low, d_high, ray0_masked, ray_direction_masked, object_mask,mask_0_not_occupied)
        
        return surface_depths, surface_points, mask

    def _secant(self, f_low, f_high, d_low, d_high, ray0_masked, ray_direction_masked, mask, mask_0_not_occupied, tau = 0.5, it=0, n_secant_steps=8):
        """
        Args:
            ray_dirs: torch.tensor([n_rays, 3])
            secant_interval: torch.tensor([n_rays, 2])
            object_mask: torch.tensor([n_rays, 1], dtype=np.bool)

        Output:
            surface_points: torch.tensor([n_rays, 3])
            surface_depths: torch.tensor(1, [n_rays])
        """
        # set_trace()
        self.ray0_masked = ray0_masked
        self.ray_direction_masked = ray_direction_masked
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        for i in range(n_secant_steps):
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
            with torch.no_grad():
                all_local_latents = self._interpolate_latent(p_mid.unsqueeze(1))
                # set_trace()
                local_latent = self.local_integrator(ray_direction_masked.unsqueeze(0), self.selected_camera_world, p_mid.unsqueeze(1), all_local_latents)
                # local_latent = self.local_integrator(all_local_latents)
                global_latent = self.global_integrator(self.global_features)

                # latent = self._get_latent(local_latent, global_latent)
                f_mid, _, _ = self.decoder(p_mid.unsqueeze(1), local_latent, ray_dirs=None, return_occupancy=True, return_feature_vec=False, return_rgb=False, global_latent=global_latent) 

                f_mid = f_mid - tau
            ind_low = f_mid < 0
            # set_trace()
            ind_low = ind_low.squeeze()
            f_mid = f_mid.squeeze()
            if ind_low.sum() > 0:
                d_low[ind_low] = d_pred[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
                    # with torch.no_grad():
        # all_local_latents = self._interpolate_latent(p_mid.unsqueeze(1))
        # # set_trace()
        # local_latent = self.local_integrator(ray0_masked.unsqueeze(0), self.selected_camera_world, p_mid.unsqueeze(1), all_local_latents)
        # # local_latent = self.local_integrator(all_local_latents)
        # global_latent = self.global_integrator(self.global_features)

        # # latent = self._get_latent(local_latent, global_latent)
        # occupancy_output, _, _ = self.decoder(p_mid.unsqueeze(1), local_latent, ray_dirs=None, return_occupancy=True, return_feature_vec=False, return_rgb=False, global_latent=global_latent)
        # set_trace() 
        d_pred_out = torch.ones(self.batch_size, self.n_pts).to(self.device)
        d_pred_out[mask] = d_pred
        d_pred_out[mask == 0] = np.inf
        d_pred_out[mask_0_not_occupied == 0] = 0
        return d_pred_out, p_mid, mask
    
    def get_surface_occ(self, surface_points):
        all_local_latents = self._interpolate_latent(surface_points.unsqueeze(1))
        # set_trace()
        local_latent = self.local_integrator(self.ray_direction_masked.unsqueeze(0), self.selected_camera_world, surface_points.unsqueeze(1), all_local_latents)
        # local_latent = self.local_integrator(all_local_latents)
        global_latent = self.global_integrator(self.global_features)

        # latent = self._get_latent(local_latent, global_latent)
        occupancy_output, _, _ = self.decoder(surface_points.unsqueeze(1), local_latent, ray_dirs=None, return_occupancy=True, return_feature_vec=False, return_rgb=False, global_latent=global_latent)
        # set_trace() 
        return occupancy_output

    def volume_render(self, surface_depths, depth_interval, ray_vector, add_noise = True, eval_ = False, it = 0):
        epsilon = 1e-6
        # set_trace()
        mask_zero_occupied = surface_depths == 0
        surface_depths = surface_depths.detach()
        device = self.device
        # Get mask for predicted depth
        mask_pred = get_mask(surface_depths).detach()
        
        with torch.no_grad():
            dists =  torch.ones_like(surface_depths).to(device)
            dists[mask_pred] = surface_depths[mask_pred]
            dists[mask_zero_occupied] = 0.
            network_object_mask = mask_pred & ~mask_zero_occupied
            network_object_mask = network_object_mask[0]
            dists = dists[0]
        # set_trace()

        # Project depth to 3d poinsts
        camera_world = self.camera_world.reshape(-1, 3).repeat(self.n_pts, 1)
        ray_vector = self.sampled_ray_dirs.reshape(-1, 3)
        
        points = camera_world + ray_vector * dists.unsqueeze(-1)
        points = points.view(-1,3)

        # Define interval
        depth_interval[:,0] = torch.Tensor([0.0]).cuda() 
        dists_intersect = depth_interval.reshape(-1, 2)
        d_inter = dists[network_object_mask]
        d_sphere_surf = dists_intersect[network_object_mask][:,1]
        delta = torch.max(self.ada_start * torch.exp(-1 * self.ada_grad * it * torch.ones(1)),\
             self.ada_end * torch.ones(1)).cuda()

        dnp = d_inter - delta
        dfp = d_inter + delta
        dnp = torch.where(dnp < torch.tensor(self.depth_range[0]).to(device),\
            torch.tensor(self.depth_range[0]).to(device), dnp)
        dfp = torch.where(dfp >  d_sphere_surf,  d_sphere_surf, dfp)
        if (dnp!=0.0).all() and it > 5000:
            full_steps = self.steps+self.steps_outside
        else:
            full_steps = self.steps

        d_interval = torch.linspace(0., 1., steps=self.steps, device=self.device)
        d_interval = d_interval.view(1, 1, -1).repeat(self.batch_size, d_inter.shape[0], 1)        
        d_interval = (dnp).view(1, -1, 1) * (1. - d_interval) + (dfp).view(1, -1, 1) * d_interval

        if full_steps != self.steps:
            d_binterval = torch.linspace(0., 1., steps=self.steps_outside, device=device)
            d_binterval = d_binterval.view(1, 1, -1).repeat(self.batch_size, d_inter.shape[0], 1)
            d_binterval =  self.depth_range[0] * (1. - d_binterval) + (dnp).view(1, -1, 1)* d_binterval
            d1,_ = torch.sort(torch.cat([d_binterval, d_interval],dim=-1), dim=-1)
        else:
            d1 = d_interval

        if add_noise:
            di_mid = .5 * (d1[:, :, 1:] + d1[:, :, :-1])
            di_high = torch.cat([di_mid, d1[:, :, -1:]], dim=-1)
            di_low = torch.cat([d1[:, :, :1], di_mid], dim=-1)
            noise = torch.rand(self.batch_size, d1.shape[1], full_steps, device=device)
            d1 = di_low + (di_high - di_low) * noise 

        p_iter = camera_world[network_object_mask].unsqueeze(-2)\
             + ray_vector[network_object_mask].unsqueeze(-2) * d1.unsqueeze(-1)
        p_iter = p_iter.reshape(-1, 3)


        d_nointer = dists_intersect[~network_object_mask]

        d2 = torch.linspace(0., 1., steps=full_steps, device=device)
        d2 = d2.view(1, 1, -1).repeat(self.batch_size, d_nointer.shape[0], 1)
        d2 = self.depth_range[0] * (1. - d2) + d_nointer[:,1].view(1, -1, 1) * d2

        if add_noise:
            di_mid = .5 * (d2[:, :, 1:] + d2[:, :, :-1])
            di_high = torch.cat([di_mid, d2[:, :, -1:]], dim=-1)
            di_low = torch.cat([d2[:, :, :1], di_mid], dim=-1)
            noise = torch.rand(self.batch_size, d2.shape[1], full_steps, device=device)
            d2 = di_low + (di_high - di_low) * noise 
        p_noiter = camera_world[~network_object_mask].unsqueeze(-2) \
            + ray_vector[~network_object_mask].unsqueeze(-2) * d2.unsqueeze(-1)
        p_noiter = p_noiter.reshape(-1, 3)

        # Merge rendering points
        p_fg = torch.zeros(self.batch_size * self.n_pts, full_steps, 3, device=device)
        p_fg[~network_object_mask] = p_noiter.view(-1, full_steps, 3)
        p_fg[network_object_mask] = p_iter.view(-1, full_steps, 3)
        p_fg = p_fg.reshape(-1, 3)
        ray_vector_fg = ray_vector.unsqueeze(-2).repeat(1, 1, full_steps, 1)
        ray_vector_fg = -1*ray_vector_fg.reshape(-1, 3)

        # Run Network
        noise = not eval_
        rgb_fg, logits_alpha_fg = [], []
        for i in range(0, p_fg.shape[0], self.n_max_network_queries):
            
            all_local_latents = self._interpolate_latent(p_fg[i:i+self.n_max_network_queries].unsqueeze(1))
            # set_trace()
            local_latent = self.local_integrator(-1*ray_vector_fg[i:i+self.n_max_network_queries].unsqueeze(0), self.selected_camera_world, p_fg[i:i+self.n_max_network_queries].unsqueeze(1), all_local_latents)
            # local_latent = self.local_integrator(all_local_latents)
            global_latent = self.global_integrator(self.global_features)

            # latent = self._get_latent(local_latent, global_latent)
            logits_alpha_i, _, rgb_i = self.decoder(
                p_fg[i:i+self.n_max_network_queries].unsqueeze(1), local_latent,
                ray_vector_fg[i:i+self.n_max_network_queries], 
                return_rgb=True, global_latent=global_latent
            )
            rgb_fg.append(rgb_i.squeeze(0))
            logits_alpha_fg.append(logits_alpha_i.squeeze(0))
            #print(rgb_i.shape)

        rgb_fg = torch.cat(rgb_fg, dim=0)
        logits_alpha_fg = torch.cat(logits_alpha_fg, dim=0)
        
        rgb = rgb_fg.reshape(self.batch_size * self.n_pts, full_steps, 3)
        alpha = logits_alpha_fg.view(self.batch_size * self.n_pts, full_steps)
        # set_trace()
        rgb = rgb
        
        weights = alpha * torch.cumprod(torch.cat([torch.ones((rgb.shape[0], 1), device=device), 1.-alpha + epsilon], -1), -1)[:, :-1]
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)

        if not self.render_outer_points:
            # set_trace()
            rgb_values[~network_object_mask] = 0
        
        if not eval_:
            surface_mask = network_object_mask.view(-1)
            surface_points = points[surface_mask]
            N = surface_points.shape[0]
            surface_points_neig = surface_points.repeat(3, 1) + (torch.rand_like(surface_points.repeat(3, 1)) - 0.5) * 0.03      
            pp = torch.cat([surface_points, surface_points_neig], dim=0)
            all_local_latents = self._interpolate_latent(pp.unsqueeze(1))
            # if self.ray_direction_masked.shape[0] == 1000:
            #     set_trace()
            local_latent = self.local_integrator(self.ray_direction_masked.unsqueeze(0).repeat(1, 4, 1), self.selected_camera_world, pp.unsqueeze(1), all_local_latents)
            # local_latent = self.local_integrator(all_local_latents)
            global_latent = self.global_integrator(self.global_features)

            # latent = self._get_latent(local_latent, global_latent)
            # set_trace()
            g = self.decoder.gradient(pp.unsqueeze(1), local_latent, global_latent=global_latent).squeeze(1) # 还需要算一次latent
            #print(g.shape)
            # set_trace()
            if torch.isnan(g).sum() > 0:
                set_trace()
            if torch.isinf(g).sum() > 0:
                set_trace()
            # set_trace()
            normals_ = g[:, 0, :] / (g[:, 0, :].norm(2, dim=1).unsqueeze(-1) + 10**(-5))
            diff_norm = torch.norm(normals_[:N] - normals_[N: 2 * N], dim=-1)
            diff_norm_1 = torch.norm(normals_[:N] - normals_[2 * N : 3 * N], dim=-1)
            diff_norm_2 = torch.norm(normals_[:N] - normals_[3 * N : 4 * N], dim=-1)
            diff_norm = diff_norm + diff_norm_1 + diff_norm_2

        else:
            surface_mask = network_object_mask
            diff_norm = None

        if self.white_background:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map.unsqueeze(-1))

        # out_dict = {
        #     'rgb': rgb_values.reshape(self.batch_size, -1, 3),#(1 1024 3)
        #     'mask_pred': network_object_mask,#(1024)
        #     'normal': diff_norm,#[864]
        # }
        return rgb_values.reshape(self.batch_size, -1, 3), network_object_mask, diff_norm

    def phong_render(self, d_i, ray_vector):

        mask_zero_occupied = d_i == 0
        d_i = d_i.detach()
        device = self.device
        # Get mask for predicted depth
        mask_pred = get_mask(d_i).detach()
        light_source = self.camera_world[0]
        light = (light_source / light_source.norm(2)).unsqueeze(1).cuda()
    
        diffuse_per = torch.Tensor([0.7,0.7,0.7]).float()
        ambiant = torch.Tensor([0.3,0.3,0.3]).float() 

        
        with torch.no_grad():
            dists =  torch.ones_like(d_i).to(device)
            dists[mask_pred] = d_i[mask_pred]
            dists[mask_zero_occupied] = 0.
            network_object_mask = mask_pred & ~mask_zero_occupied
            network_object_mask = network_object_mask[0]
            dists = dists[0]
            camera_world = self.camera_world.reshape(-1, 3).repeat(self.n_pts, 1)
            ray_vector = self.sampled_ray_dirs.reshape(-1, 3)
        
            points = camera_world + ray_vector * dists.unsqueeze(-1)
            # set_trace()
            points = points.view(-1,3)
            view_vol = -1 * ray_vector.view(-1, 3)
            rgb_values = torch.ones_like(points).float().cuda()
            surface_points = points[network_object_mask]
            surface_view_vol = view_vol[network_object_mask]
            grad = []
            for pnts in torch.split(surface_points, 1000000, dim=0):
                all_local_latents = self._interpolate_latent(pnts.unsqueeze(1))
                # set_trace()
                local_latent = self.local_integrator(ray_vector[network_object_mask].unsqueeze(0), self.selected_camera_world, pnts.unsqueeze(1), all_local_latents)
                # local_latent = self.local_integrator(all_local_latents)
                global_latent = self.global_integrator(self.global_features)

                # latent = self._get_latent(local_latent, global_latent)
                grad_tmp = self.decoder.gradient(pnts.unsqueeze(1), local_latent, global_latent=global_latent)[:,0,0,:].detach()
                if len(grad_tmp.shape) < 2:
                    set_trace()
                grad.append(grad_tmp)
                torch.cuda.empty_cache()
            # grad = torch.cat(grad,0).squeeze(0)
            grad = torch.cat(grad,0)
            try:
                surface_normals = grad / grad.norm(2,1,keepdim=True)
            except:
                set_trace()
        # set_trace()
        diffuse = torch.mm(surface_normals.squeeze(1), light).clamp_min(0).repeat(1, 3) * diffuse_per.unsqueeze(0).cuda()
        rgb_values[network_object_mask] = (ambiant.unsqueeze(0).cuda() + diffuse).clamp_max(1.0)

        with torch.no_grad():
            rgb_val = torch.zeros(self.batch_size * self.n_pts, 3, device=self.device)
            all_local_latents = self._interpolate_latent(surface_points.unsqueeze(1))
            local_latent = self.local_integrator(ray_vector[network_object_mask].unsqueeze(0), self.selected_camera_world, surface_points.unsqueeze(1), all_local_latents)
            # local_latent = self.local_integrator(all_local_latents)

            global_latent = self.global_integrator(self.global_features)

            # latent = self._get_latent(local_latent, global_latent)
            # _,_,rgb_val = self.decoder(surface_points.unsqueeze(1), latent,surface_view_vol,return_rgb=True)
            # set_trace()
            _, _, rgb_val_1 = self.decoder(surface_points.unsqueeze(1), local_latent, surface_view_vol,return_rgb=True, global_latent=global_latent)
            rgb_val[network_object_mask] = rgb_val_1.squeeze(1)
            # set_trace()
        return rgb_values.reshape(self.batch_size, -1, 3), rgb_val.reshape(self.batch_size, -1, 3)

    def _get_secant_interval(self, occupancy_matrix, grid_points_depths, sampled_ray_dirs, n_steps = 256):
        """
        Args:
            occupancy_matrix: torch.tensor([n_rays, n_points, 1])
            grid_points_depths: torch.tensor([n_rays, n_points, 1])

        Output:
            secant_interval: torch.tensor([n_rays, 2])
            object_mask: torch.tensor([n_rays, 1], dtype=np.bool)
        """
        # set_trace()
        val = occupancy_matrix.permute(2,0,1)
        batch_size, n_pts, D = val.shape
        self.n_pts = n_pts
        self.batch_size = batch_size
        mask_0_not_occupied = val[:, :, 0] < 0
        #set_trace()
        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        # set_trace()
        try:
            sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:])- ((val[:, :, :-1] * val[:, :, 1:])==0).float(),
                                    torch.ones(batch_size, n_pts, 1).to(self.device)],
                                    dim=-1)
        except:
            set_trace()
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(self.device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),
                              torch.arange(n_pts).unsqueeze(-0), indices] < 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        n = batch_size * n_pts
        d_low = grid_points_depths.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = grid_points_depths.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_high = val.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]

        ray0_masked = self.camera_world.repeat(n,1).unsqueeze(0)[mask]
        ray_direction_masked = sampled_ray_dirs.unsqueeze(0)[mask]

        # set_trace()
        return f_low, f_high, d_low, d_high, ray0_masked, ray_direction_masked, mask, mask_0_not_occupied

    def _get_latent(self, local_latent, global_latent):
        """
        Args:
            local_latent: torch.tensor([n_rays, n_points, local_latent_dim])
            global_latent: torch.tensor([global_latent_dim])

        Output:
            latent: torch.tensor([n_rays, n_points, latent_dim])
        """
        return local_latent
        
    def _interpolate_latent(self, points):
        """
        Interpolate latents of points from feature_map

        Args:
            points: torch.tensor([n_rays, n_points, 3])

        Output:
            all_latents: torch.tensor([n_views, n_rays, n_points, latent_dim]) 
        """
        n_views = self.selected_world_mats.shape[0]
        n_rays, n_points, _ = points.shape
        points = points.unsqueeze(0).reshape(1, -1, 3).repeat(n_views, 1, 1)
        # cam_points = transform_to_camera_space(points, self.selected_camera_mats, self.selected_world_mats, self.selected_scale_mats)
        # set_trace()
        cam_points = transform_to_camera_space_new(points, self.camera_selected)
        # set_trace()
        # depths = cam_points[:, :, 2].unsqueeze(2)
        # pixel_coords = cam_points / depths
        n, c, h, w = self.feature_maps.shape
        
       
        #torch.gather
        # set_trace()
        # feature = self.feature_maps.view(n, c, h*w)#[49,32,1200*1600]
        # _, p, _ = pixel_coords.shape
        
        # pixel_coords = pixel_coords[:, :, 0] + pixel_coords[:, :, 1] * w
        # pixel_coords = pixel_coords.view(n, 1, p).expand(n, c, p)#[49,32,1024]
        # pixel_coords = pixel_coords.long()
        # pixel_feature = torch.gather(feature, index=pixel_coords, dim=2)#[49,32,1024]

        pixel_coords = cam_points[:, :, :2].float().unsqueeze(1)
        pixel_coords[:, :, :, 0] = (pixel_coords[:, :, :, 0] - int(w/2)) / int(w/2)
        pixel_coords[:, :, :, 1] = (pixel_coords[:, :, :, 1] - int(h/2)) / int(h/2)
        pixel_feature = torch.nn.functional.grid_sample(self.feature_maps, pixel_coords, mode="bilinear", padding_mode="zeros")
        pixel_feature = pixel_feature.permute(0, 1, 3, 2)
        pixel_feature = pixel_feature.reshape(n_views, c, n_rays, n_points)
        return pixel_feature

    def _sample_grid_points(self, sampled_ray_dirs, depth_interval, n_points, depth_range=[0.3, 2.4]):
        """
        Calculate the world coordinates of n points sampled in the given depth_interval on each ray

        Args:
            sampled_ray_dirs: torch.tensor([n_rays, 3])
            depth_interval: torch.tensor([n_rays, 2])
            n_points: int

        Output:
            grid_points: torch.tensor([n_rays, n_points, 3])
            grid_points_depths: torch.tensor([n_rays, n_points, 1])
        """
        
        depth_range = self.depth_range
        n_steps = n_points
        d_intersect = depth_interval[...,1]
        n_pix = d_intersect.shape[0]
        d_proposal = torch.linspace(
            0, 1, steps=n_steps).view(
                1, 1, n_steps, 1).to(self.device)
        d_proposal = depth_range[0] * (1. - d_proposal) + d_intersect.view(1, -1, 1,1)* d_proposal
        p_proposal = self.camera_world.unsqueeze(0).unsqueeze(0).repeat(1, n_pix, n_steps, 1) + \
            sampled_ray_dirs.unsqueeze(0).unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal 
        return p_proposal.squeeze(0), d_proposal.squeeze(0)

    def _sphere_intersect(self, sampled_ray_dirs, r = 4.0):
        """
        Output the depth of intersection between sampled_ray and the front and rear surfaces of the big ball

        Args:
            sampled_ray_dirs: torch.tensor([n_rays, 3])

        Output:
            depth_interval: torch.tensor([n_rays, 2])
        """
        sampled_ray_dirs = sampled_ray_dirs.unsqueeze(0)
        n_imgs, n_pix, _ = sampled_ray_dirs.shape
        cam_loc = self.camera_world.unsqueeze(-1)
        # ray_cam_dot = torch.bmm(sampled_ray_dirs, cam_loc).squeeze()
        ray_cam_dot = torch.bmm(sampled_ray_dirs, cam_loc).squeeze(0)
        under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

        under_sqrt = under_sqrt.reshape(-1)
        mask_intersect = under_sqrt > 0
        
        sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
        sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
        sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)
        sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
        sphere_intersections = sphere_intersections.clamp_min(0.0)
        mask_intersect = mask_intersect.reshape(n_imgs, n_pix)
        return sphere_intersections.squeeze(0)

    def register_modules(self, decoder, local_integrator, global_integrator):
        self.decoder = decoder
        self.local_integrator = local_integrator
        self.global_integrator = global_integrator

    def load_feature(self, feature_maps, global_features):
        self.feature_maps = feature_maps
        self.global_features = global_features

    def get_depth_w_grad(self, d_pred_o, p_pred_o, mask):
        # set_trace()
        d_pred = d_pred_o.clone()
        n_points = d_pred.shape[1]
        ray0 = self.camera_world.unsqueeze(0).repeat(1,n_points,1)
        ray_direction = self.sampled_ray_dirs
        # decoder = self.decoder

        # inputs = [ray0, ray_direction, d_pred, p_pred, latent, decoder, mask] + [k for k in decoder.parameters()]

        # return self.depth_func(*inputs)
        d_pred[mask == 0] = 20
        p_pred = ray0 + self.sampled_ray_dirs * d_pred.unsqueeze(-1)
        all_local_latents = self._interpolate_latent(p_pred)
        # set_trace()
        local_latent = self.local_integrator(self.sampled_ray_dirs, self.selected_camera_world, p_pred.permute(1,0,2), all_local_latents)

        global_latent = self.global_integrator(self.global_features)


        # ray0 = self.ray0_masked
        # ray_direction = self.ray_direction_masked
        decoder = self.decoder
        # d_pred_masked = d_pred[mask]

        inputs = [ray0, ray_direction, d_pred, p_pred, local_latent, global_latent, decoder, mask] + [k for k in decoder.parameters()]
        # set_trace()

        return self.depth_func.apply(*inputs)

    def get_depth_loss(self, depth):
        depth_0 = depth[:,:256]
        depth_1 = depth[:,256:512]
        depth_2 = depth[:,512:768]
        depth_3 = depth[:,768:1024]
        depth_loss = ((depth_0-depth_1).abs() + (depth_0-depth_2).abs() + (depth_0-depth_3).abs()).mean()
        return depth_loss

    def get_iou_loss(self, depth_grad, object_mask,mask_gt):
        # try:
        #     assert(surface_depths.unsqueeze(-1).shape == mask_gt.shape)
        # except:
        #     set_trace()
        # set_trace()
        # network_mask_inf = (surface_depths != np.inf) 
        # network_mask_0 = (surface_depths != 0)
        # network_mask = network_mask_inf & network_mask_0
        surface_depths = depth_grad
        network_mask = object_mask.clone()
        mask_gt = mask_gt.bool()
        occupancy_mask = network_mask & (~mask_gt.squeeze(-1))#-0该空没空
        freespace_mask = (~network_mask) & mask_gt.squeeze(-1)#-1该占没占

        n_points = surface_depths.shape[1]
        ray0 = self.camera_world.unsqueeze(0).repeat(1,n_points,1)
        ray_direction = self.sampled_ray_dirs

        p_pred = ray0 + ray_direction * surface_depths.unsqueeze(-1)
        p_pred_masked = p_pred[network_mask].unsqueeze(0).reshape(-1, 3)

        # all_local_latents = self._interpolate_latent(p_pred_masked.unsqueeze(1))
        # local_latent = self.local_integrator(self.sampled_ray_dirs, self.selected_camera_world, p_pred_masked.unsqueeze(1), all_local_latents)
        # global_latent = self.global_integrator(self.global_features)
        # occupancy_output_occ, _, _ = self.decoder(p_pred_masked.unsqueeze(1), local_latent, ray_dirs=None, return_occupancy=True, return_feature_vec=False, return_rgb=False, global_latent=global_latent)
        # occ = (occupancy_output_occ).reshape(1, -1, 1)[occupancy_mask] 

        occ = self.get_surface_occ(p_pred_masked).reshape(1, -1, 1)[occupancy_mask[network_mask].unsqueeze(0)]
        if occupancy_mask.sum() != 0:
            # occ_loss = (occ - torch.zeros_like(occ)).sum()
            occ_loss = F.binary_cross_entropy_with_logits(occ, torch.zeros_like(occ), size_average=False)
        else:
            occ_loss = torch.tensor(0)
        if torch.isnan(occ_loss):
            set_trace()
        # set_trace()
        n_step = 20
        d2 = torch.linspace(6., 12., steps=n_step, device=self.device)
        d2 = d2.view(1, 1, -1).repeat(1, int(freespace_mask.sum()), 1).permute(1,2,0)
        p_random = ray0[freespace_mask].unsqueeze(1).repeat(1,n_step,1) + ray_direction[freespace_mask].unsqueeze(1).repeat(1,n_step,1) * d2
        # p_random_masked = p_random.reshape(-1, 3)
        # set_trace()
        all_local_latents = self._interpolate_latent(p_random)
        local_latent = self.local_integrator(self.sampled_ray_dirs[freespace_mask].unsqueeze(0), self.selected_camera_world, p_random, all_local_latents)
        global_latent = self.global_integrator(self.global_features)
        occupancy_output, _, _ = self.decoder(p_random, local_latent, ray_dirs=None, return_occupancy=True, return_feature_vec=False, return_rgb=False, global_latent=global_latent)
        
        # occ_free = occupancy_output.reshape(1, -1, 1)[freespace_mask]


        # p_random = ray0 + ray_direction * torch.rand_like(surface_depths.unsqueeze(-1)) * self.depth_range[1]
        # p_random_masked = p_random.reshape(-1, 3)
        # # set_trace()
        # all_local_latents = self._interpolate_latent(p_random_masked.unsqueeze(1))
        # local_latent = self.local_integrator(self.sampled_ray_dirs, self.selected_camera_world, p_random_masked.unsqueeze(1), all_local_latents)
        # global_latent = self.global_integrator(self.global_features)
        # occupancy_output, _, _ = self.decoder(p_random_masked.unsqueeze(1), local_latent, ray_dirs=None, return_occupancy=True, return_feature_vec=False, return_rgb=False, global_latent=global_latent)
        # occ_free = occupancy_output.reshape(1, -1, 1)[freespace_mask] 
        # occ_free = self.get_surface_occ(p_random_masked).reshape(1, -1, 1)[freespace_mask]
        if freespace_mask.sum() != 0:
            free_loss = F.binary_cross_entropy_with_logits(torch.ones_like(occupancy_output),occupancy_output, size_average=False)
            # free_loss = F.binary_cross_entropy_with_logits(occ_free, torch.ones_like(occ_free), size_average=False)
        else:
            free_loss = torch.tensor(0)
        
        if torch.isnan(free_loss):
            set_trace()
        # set_trace()
        # return ( free_loss)/n_points
        return (occ_loss + 0.5*free_loss)#/n_points
        