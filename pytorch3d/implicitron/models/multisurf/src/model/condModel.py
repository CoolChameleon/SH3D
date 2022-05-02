import sys, os
# print(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
# from decoder.unisurfDecoder import OccupancyNetwork
from decoder.condMLPDecoder import condMLPDecoder
from encoder.UNetEncoder import UNetEncoder
from integrator.maxPoolItg import maxPoolItg, maxPoolGlobal
from integrator.integrator import AngleWeightedfeatureaggregation as avg_agg
from rendUtils.renderFunc import renderFunc
from utils import common
from utils.common import get_mask
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.implicitron.tools.metric_utils import calc_psnr, eval_depth, iou, rgb_l1
import lpips

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

        self.decoder = condMLPDecoder(cfg).to(self.device)
        # self.local_integrator = maxPoolItg(cfg).to(self.device)
        self.local_integrator = avg_agg(cfg).to(self.device)
        self.global_integrator = maxPoolGlobal(cfg).to(self.device)
        self.render_fn.register_modules(decoder=self.decoder, local_integrator=self.local_integrator, global_integrator=self.global_integrator)


    def forward(self, data, it):
        
        img, mask, world_mat, camera_mat, scale_mat, all_imgs, all_masks, all_world_mats, all_camera_mats, all_scale_mats, view_idx, neighbor_list, camera_all, camera, camera_selected = self._process_input_data(data)

        # set_trace()
        feature_maps, global_feature = self.feature_encoder(all_imgs, neighbor_list, all_masks)

        self.render_fn.load_feature(feature_maps, global_feature)
        # set_trace()
        self.render_fn.set_params(world_mat, camera_mat, scale_mat, all_world_mats, all_camera_mats, all_scale_mats, neighbor_list, camera_all, camera, camera_selected)

        self.global_integrator.set_neighbor(neighbor_list)

        # 我没有看common里面是不是真的有这么个函数
        
        sampled_p_ndc, sampled_pixels = common.sample_patch_points(self.n_points_sampled, self.img_res, continuous=False)#(1 1024 2)
        sampled_p_ndc = sampled_p_ndc.to(self.device)
        sampled_pixels = sampled_pixels.to(self.device)
        
        rgb_gt = common.get_tensor_values(img, sampled_pixels.clone())
        # set_trace()
        mask_gt = common.get_tensor_values(mask.float(), sampled_pixels.clone())

        with torch.no_grad():
            sampled_ray_dirs, _ = self.render_fn.convert_pixels_to_rays(sampled_p_ndc.clone())
    
            surface_depths, surface_points, object_mask, depth_interval = self.render_fn.find_surface(sampled_ray_dirs)
        
        surface_occ = self.render_fn.get_surface_occ(surface_points)

        rgb, network_object_mask, normal = self.render_fn.volume_render(surface_depths, depth_interval, sampled_ray_dirs, add_noise = True, eval_ = False, it = 0)
        
        out_dict = {
            "rgb_pred": rgb,
            "mask_pred": object_mask,
            "surface_points_occ": surface_occ,
            "normal": normal,
            "rgb_gt": rgb_gt,
            "mask_gt": mask_gt
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
        all_focal_length = data["all_focal_length"].squeeze(0).squeeze(1).to(self.device)
        all_principal_point = data["all_principal_point"].squeeze(0).squeeze(1).to(self.device)
        all_R = data["all_R"].squeeze(0).squeeze(1).to(self.device)
        all_T = data["all_T"].squeeze(0).squeeze(1).to(self.device)
        
        camera_all = PerspectiveCameras(
            focal_length=all_focal_length,
            principal_point=all_principal_point,
            R=torch.tensor(all_R, dtype=torch.float),
            T=torch.tensor(all_T, dtype=torch.float),
            image_size=torch.tensor([200,200])[None],
        )
        # set_trace()
        img = all_imgs[view_idx, ...]
        mask = all_masks[view_idx, ...]
        world_mat = all_world_mats[view_idx, ...]
        camera_mat = all_camera_mats[view_idx, ...]
        scale_mat = all_scale_mats[view_idx, ...]
        focal_length  = all_focal_length[view_idx, ...]
        principal_point = all_principal_point[view_idx, ...]
        R = all_R[view_idx, ...]
        T = all_T[view_idx, ...]
        camera = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=torch.tensor(R, dtype=torch.float),
            T=torch.tensor(T, dtype=torch.float),
            image_size=torch.tensor([200,200])[None],
            # in_ndc=False,
        )
        selected_focal_length  = all_focal_length[neighbor_list, ...]
        selected_principal_point = all_principal_point[neighbor_list, ...]
        selected_R = all_R[neighbor_list, ...]
        selected_T = all_T[neighbor_list, ...]
        camera_selected = PerspectiveCameras(
            focal_length=selected_focal_length,
            principal_point=selected_principal_point,
            R=torch.tensor(selected_R, dtype=torch.float),
            T=torch.tensor(selected_T, dtype=torch.float),
            image_size=torch.tensor([200,200])[None],
            # in_ndc=False,
        )
        return img, mask, world_mat, camera_mat, scale_mat, all_imgs, all_masks, all_world_mats, all_camera_mats, all_scale_mats, view_idx, neighbor_list, camera_all, camera, camera_selected
        # return img, mask, world_mat, camera_mat, scale_mat, all_imgs, all_masks, all_world_mats, all_camera_mats, all_scale_mats, view_idx, neighbor_list

    def plot(self, data, resolution,it, out_render_path):
        img, mask, world_mat, camera_mat, scale_mat, all_imgs, all_masks, all_world_mats, all_camera_mats, all_scale_mats, view_idx, neighbor_list, camera_all, camera, camera_selected = self._process_input_data(data)

        feature_maps, global_feature = self.feature_encoder(all_imgs, neighbor_list)
        self.render_fn.load_feature(feature_maps, global_feature)

        self.render_fn.set_params(world_mat, camera_mat, scale_mat, all_world_mats, all_camera_mats, all_scale_mats, neighbor_list, camera_all, camera, camera_selected)

        self.global_integrator.set_neighbor(neighbor_list)
        h, w = resolution
        
        p_loc, pixels_ndc = common.arange_pixels(resolution=(h, w))
        pixels_ndc = pixels_ndc.to(self.device)
        p_loc = p_loc.float().to(self.device)
        with torch.no_grad():
            mask_pred = torch.ones(pixels_ndc.shape[0], pixels_ndc.shape[1]).bool()
            rgb_pred_list = []
            rgb_pred = []
            phong_list = []
            phong_list_mask = []
            mask_occ_list = []
            for  pixels_i, pixel_ndc_i in zip(torch.split(p_loc, 2048, dim=1), torch.split(pixels_ndc, 2048, dim=1)):
                #rgb_render
                # set_trace()
                sampled_ray_dirs, camera_world = self.render_fn.convert_pixels_to_rays(pixel_ndc_i)

                surface_depths, surface_points, object_mask, depth_interval = self.render_fn.find_surface(sampled_ray_dirs)

                # surface_occ = self.render_fn.get_surface_occ(surface_points)

                # surface_mask = surface_occ + 0.5
                mask_occ_list.append(object_mask)

                rgb_pred, _, _ =self.render_fn.volume_render(surface_depths, depth_interval, sampled_ray_dirs, add_noise = False, eval_ = True, it = it)
                rgb_pred_list.append(rgb_pred)

                #phong_render 
                
                #self.model.eval() 这句没地方写
                rgb, rgb_surf = self.render_fn.phong_render(surface_depths,sampled_ray_dirs)
                phong_list.append(rgb)
                phong_list_mask.append(rgb_surf) 
                

           
            rgb_pred = torch.cat(rgb_pred_list, dim=1).cpu()
            # set_trace()
            p_loc = p_loc.cpu().long()
            p_loc1 = p_loc[mask_pred]
            img_out = (255 * np.zeros((h, w, 3))).astype(np.uint8)
            img1 = (255 * np.zeros((h, w, 3))).astype(np.float)
            img1[p_loc1[:, 1], p_loc1[:, 0]] = rgb_pred
            
            # set_trace()
            # psnr = self.calculate_psnr(img.squeeze(0).permute(1, 2, 0).cpu().numpy(), 
            #     img1, 
            #     mask.squeeze(0).permute(1, 2, 0).cpu()
            # )
            # set_trace()
            loss_fn_alex = lpips.LPIPS(net='alex')
            img_target = img * 2 - 1
            img_source = (torch.tensor(img1) *2 - 1).permute(2,0,1).unsqueeze(0)
            d = loss_fn_alex(img_target.cpu(), img_source.cpu().float())
            psnr = calc_psnr(img.squeeze(0).permute(1, 2, 0).cpu(), 
                torch.tensor(img1), 
                mask.squeeze(0).permute(1, 2, 0).cpu())
            with open(self.psnr_path, "a+") as f:
                f.write("Iteration {}, psnr = {},  lpips={}\n".format(it, psnr, d))
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
            phong_pred_mask = torch.cat(phong_list_mask, dim=1).cpu()
            # set_trace()
            mask_occ = torch.cat(mask_occ_list, dim=1).unsqueeze(2).cpu()


            p_loc = p_loc.cpu().long()
            p_loc1 = p_loc[mask_pred]
            # set_trace()
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

            if mask_pred.sum() > 0:
                rgb_hat = phong_pred_mask[mask_pred].detach().cpu().numpy()
                rgb_hat = (rgb_hat * 255).astype(np.uint8)
                img_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
        
            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, '2_phong_mask.png')
            )
            if mask_pred.sum() > 0:
                rgb_hat = mask_occ[mask_pred].detach().cpu().numpy()
                rgb_hat = (rgb_hat * 255).astype(np.uint8)
                img_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
        
            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, 'mask_pred.png')
            )
            # set_trace()
            if mask_pred.sum() > 0:
                rgb_hat = mask.float().squeeze(0).permute(1,2,0).repeat(1,1,3).detach().cpu().numpy()
                rgb_hat = (rgb_hat * 255).astype(np.uint8)
                img_out = rgb_hat
        
            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, 'mask.png')
            )
        return img_out.astype(np.uint8)

    def plot_old(self, data, resolution,it, out_render_path):
        img, mask, world_mat, camera_mat, scale_mat, all_imgs, all_masks, all_world_mats, all_camera_mats, all_scale_mats, view_idx, neighbor_list, camera_all, camera, camera_selected = self._process_input_data(data)
        feature_maps, global_feature = self.feature_encoder(all_imgs, neighbor_list)
        self.render_fn.load_feature(feature_maps, global_feature)
        # set_trace()

        self.render_fn.set_params(world_mat, camera_mat, scale_mat, all_world_mats, all_camera_mats, all_scale_mats, neighbor_list,camera_all, camera, camera_selected)

        self.global_integrator.set_neighbor(neighbor_list)
        h, w = resolution
        
        p_loc, pixels_ndc = common.arange_pixels(resolution=(h, w))
        pixels_ndc = pixels_ndc.to(self.device)
        p_loc = p_loc.float().to(self.device)
        with torch.no_grad():
            mask_pred = torch.ones(pixels_ndc.shape[0], pixels_ndc.shape[1]).bool()
            rgb_pred_list = []
            rgb_pred = []
            phong_list = []
            for  pixels_i, pixel_ndc_i in zip(torch.split(p_loc, 2048, dim=1), torch.split(pixels_ndc, 2048, dim=1)):
                #rgb_render
                sampled_ray_dirs, camera_world = self.render_fn.convert_pixels_to_rays(pixel_ndc_i)

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

