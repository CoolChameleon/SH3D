import sys, os
# print(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
from decoder.multiHeadDecoder import multiHeadDecoder
from encoder.UNetEncoder import UNetEncoder
from integrator.maxPoolItg import maxPoolItg, maxPoolGlobal
from rendUtils.renderFunc import renderFunc
from utils import common
from utils.common import get_mask

from ipdb import set_trace

class BaseModel(nn.Module):
    def __init__(self, cfg, mode="train", device: str="cuda"):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.device = device
        self.feature_encoder = UNetEncoder(cfg).to(self.device)
        self.render_fn = renderFunc(cfg)
        self.n_points_sampled = cfg["model"]["n_points_sampled"]
        self.img_res = cfg["dataloading"].get("resize_res", cfg["dataloading"]["img_size"])
        self.psnr_path = cfg["training"]["psnr_path"]

        self.decoder = multiHeadDecoder(cfg).to(self.device)
        self.local_integrator = maxPoolItg(cfg).to(self.device)
        self.global_integrator = maxPoolGlobal(cfg).to(self.device)
        self.render_fn.register_modules(decoder=self.decoder, local_integrator=self.local_integrator, global_integrator=self.global_integrator)


    def forward(self, data, it):
        
        img, mask, world_mat, camera_mat, scale_mat, all_imgs, all_masks, all_world_mats, all_camera_mats, all_scale_mats, view_idx, neighbor_list = self._process_input_data(data)

        # set_trace()
        feature_maps, global_feature = self.feature_encoder(all_imgs, neighbor_list)

        self.render_fn.load_feature(feature_maps, global_feature)
        # set_trace()
        self.render_fn.set_params(world_mat, camera_mat, scale_mat, all_world_mats, all_camera_mats, all_scale_mats, neighbor_list)

        self.local_integrator.set_neighbor(neighbor_list)

        # 我没有看common里面是不是真的有这么个函数
        
        _, sampled_pixels = common.sample_patch_points(self.n_points_sampled, self.img_res, continuous=False)#(1 1024 2)
        sampled_pixels = sampled_pixels.to(self.device)
        
        rgb_gt = common.get_tensor_values(img, sampled_pixels.clone())

        with torch.no_grad():
            sampled_ray_dirs, _ = self.render_fn.convert_pixels_to_rays(sampled_pixels)

            surface_depths, surface_points, object_mask, depth_interval = self.render_fn.find_surface(sampled_ray_dirs)

        rgb, network_object_mask, normal = self.render_fn.volume_render(surface_depths, depth_interval, sampled_ray_dirs, add_noise = True, eval_ = False, it = 0)
        
        out_dict = {
            "rgb_pred": rgb,
            "mask_pred": network_object_mask,
            "normal": normal,
            "rgb_gt": rgb_gt
        }
        return out_dict

    def _process_input_data(self, data):
        """
        Output:
            img: torch.tensor([1, 3, h, w])
            mask: torch.tensor([1, 1, h, w])
            world_mat: torch.tensor([1, 4, 4])
            camera_mat: torch.tensor([1, 4, 4])
            scale_mat: torch.tensor([1, 4, 4])
            all_imgs: torch.tensor([n_views, 3, h, w])
            all_masks: torch.tensor([n_views, 1, h, w])
            all_world_mats: torch.tensor([n_views, 4, 4])
            all_camera_mats: torch.tensor([n_views, 4, 4])
            all_scale_mats: torch.tensor([n_views, 4, 4])
            view_idx: torch.tensor([1])
            neighbor_list: torch.tensor([num_neighbors])

        """
        all_imgs = data["all_imgs"].squeeze(0).to(self.device)
        all_masks = data["all_masks"].squeeze(0).to(self.device)
        all_world_mats = data["all_world_mats"].squeeze(0).to(self.device)
        all_camera_mats = data["all_camera_mats"].squeeze(0).to(self.device)
        all_scale_mats = data["all_scale_mats"].squeeze(0).to(self.device)
        view_idx = data["view_idx"]
        neighbor_list = data["neighbor_list"].squeeze(0).to(self.device)
        img = all_imgs[view_idx, ...]
        mask = all_masks[view_idx, ...]
        world_mat = all_world_mats[view_idx, ...]
        camera_mat = all_camera_mats[view_idx, ...]
        scale_mat = all_scale_mats[view_idx, ...]
        return img, mask, world_mat, camera_mat, scale_mat, all_imgs, all_masks, all_world_mats, all_camera_mats, all_scale_mats, view_idx, neighbor_list

    def _volume_render(self, surface_depths, surface_points, sampled_ray_dirs, object_mask, feature_maps, global_feature):
        """
        Args:
            surface_depths: torch.tensor([n_rays, 1])
            surface_points: torch.tensor([n_rays, 3])
            sampled_ray_dirs: torch.tensor([n_rays, 3])
            object_mask: torch.tensor([n_rays, 1])
            feature_maps: torch.tensor([n_views, feature_h, feature_w, feature_dim])
            global_feature: torch.tensor([n_views, global_feature_dim])

        Output:
            rgb: torch.tensor([n_rays, 3])
            normal: torch.tensor([n_rays, 3])
        """
        
        pass

    def plot(self, data, resolution,it, out_render_path):
        img, mask, world_mat, camera_mat, scale_mat, all_imgs, all_masks, all_world_mats, all_camera_mats, all_scale_mats, view_idx, neighbor_list = self._process_input_data(data)
        feature_maps, global_feature = self.feature_encoder(all_imgs, neighbor_list)
        self.render_fn.load_feature(feature_maps, global_feature)

        self.render_fn.set_params(world_mat, camera_mat, scale_mat, all_world_mats, all_camera_mats, all_scale_mats, neighbor_list)

        self.local_integrator.set_neighbor(neighbor_list)
        h, w = resolution
        
        p_loc, pixels = common.arange_pixels(resolution=(h, w))
        pixels = pixels.to(self.device)
        p_loc = p_loc.float().to(self.device)
        with torch.no_grad():
            mask_pred = torch.ones(pixels.shape[0], pixels.shape[1]).bool()
            rgb_pred_list = []
            rgb_pred = []
            phong_list = []
            for k, pixels_i in enumerate(torch.split(p_loc, 1024, dim=1)):
                #rgb_render
                sampled_ray_dirs, camera_world = self.render_fn.convert_pixels_to_rays(pixels_i)

                surface_depths, surface_points, object_mask, depth_interval = self.render_fn.find_surface(sampled_ray_dirs)
                rgb_pred, _, _ =self.render_fn.volume_render(surface_depths, depth_interval, sampled_ray_dirs, add_noise = False, eval_ = True, it = it)
                rgb_pred_list.append(rgb_pred)

                #phong_render 
                
                #self.model.eval() 这句没地方写
                rgb, rgb_surf = self.render_fn.phong_render(surface_depths,sampled_ray_dirs)
                phong_list.append(rgb) 
                

           
            rgb_pred = torch.cat(rgb_pred_list, dim=1).cpu()
            p_loc = p_loc.cpu().long()
            p_loc1 = p_loc[mask_pred]
            img_out = (255 * np.zeros((h, w, 3))).astype(np.uint8)
            img1 = (255 * np.zeros((h, w, 3))).astype(np.float)
            img1[p_loc1[:, 1], p_loc1[:, 0]] = rgb_pred
            
            # set_trace()
            psnr = self.calculate_psnr(img.squeeze(0).permute(1, 2, 0).cpu().numpy(), 
                img1, 
                mask.squeeze(0).permute(1, 2, 0).cpu()
            )
            with open(self.psnr_path, "a+") as f:
                f.write("Iteration {}, psnr = {}\n".format(it, psnr))
            if mask_pred.sum() > 0:
                rgb_hat = rgb_pred[mask_pred].detach().cpu().numpy()
                rgb_hat = (rgb_hat * 255).astype(np.uint8)
                img_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
            # set_trace()
            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, '_unisurf.png')
            )# %d是none


            
            phong_pred = torch.cat(phong_list, dim=1).cpu()
            p_loc = p_loc.cpu().long()
            p_loc1 = p_loc[mask_pred]
            img_out = (255 * np.zeros((h, w, 3))).astype(np.uint8)
            if mask_pred.sum() > 0:
                rgb_hat = phong_pred[mask_pred].detach().cpu().numpy()
                rgb_hat = (rgb_hat * 255).astype(np.uint8)
                img_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
        
            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, '2_phong.png')
            )
        return img_out.astype(np.uint8)
        
    def calculate_psnr(self, img1, img2, mask):
        # img1 and img2 have range [0, 1]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
        if mse == 0:
            return float('inf')
        return 20 * math.log10(1.0 / math.sqrt(mse))

