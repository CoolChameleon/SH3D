import os
import torch
from collections import defaultdict
from utils.common import (
    get_tensor_values, sample_patch_points, arange_pixels
)
from tqdm import tqdm
import logging
from loss.loss import Loss
import numpy as np
logger_py = logging.getLogger(__name__)
from PIL import Image
from ipdb import set_trace
import math
class Trainer(object):
    def __init__(self, cfg, model, loss, optimizer):
        self.cfg = cfg
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
         
    def train_step(self, data, it):
        self.model.train()
        self.optimizer.zero_grad()

        out_dict = self.model(data, it)
        # loss_dict = self.loss(out_dict["rgb_pred"], out_dict["rgb_gt"], out_dict["normal"], out_dict['mask_pred'], out_dict["mask_gt"], out_dict["surface_points_occ"])
        loss_dict = self.loss(out_dict)
        loss = loss_dict["loss"]
        loss.backward()
        self.optimizer.step()
        return loss_dict

# def calculate_psnr(img1, img2, mask):
#     # img1 and img2 have range [0, 1]
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(1.0 / math.sqrt(mse))


# class Trainer_old(object):
#     ''' Trainer object for the UNISURF.

#     Args:
#         model (nn.Module): model
#         optimizer (optimizer): pytorch optimizer object
#         cfg (dict): config file
#         device (device): pytorch device
#     '''

#     def __init__(self, model, optimizer, cfg, device=None, **kwargs):
#         self.model = model
#         self.optimizer = optimizer
#         self.device = device
#         self.n_training_points = cfg['n_training_points']
#         self.n_eval_points = cfg['n_training_points']
#         self.overwrite_visualization = True

#         self.rendering_technique = cfg['type']
#         self.psnr_path = cfg["psnr_path"]

#         self.loss = Loss(
#             cfg['lambda_l1_rgb'], 
#             cfg['lambda_normals'],
#             cfg['lambda_occ_prob']
#         )

#     def evaluate(self, val_loader):
#         ''' Performs an evaluation.
#         Args:
#             val_loader (dataloader): pytorch dataloader
#         '''
#         eval_list = defaultdict(list)
        
#         for data in tqdm(val_loader):
#             eval_step_dict = self.eval_step(data)

#             for k, v in eval_step_dict.items():
#                 eval_list[k].append(v)

#         eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
#         return eval_dict

#     def train_step(self, data, it=None):
#         ''' Performs a training step.

#         Args:
#             data (dict): data dictionary
#             it (int): training iteration
#         '''
#         self.model.train()
#         self.optimizer.zero_grad()

#         loss_dict = self.compute_loss(data, it=it)
#         loss = loss_dict['loss']
#         loss.backward()
#         self.optimizer.step()
#         return loss_dict

#     def eval_step(self, data):
#         ''' Performs a validation step.

#         Args:
#             data (dict): data dictionary
#         '''
#         self.model.eval()
#         eval_dict = {}
#         #with torch.no_grad():
#         try:
#             eval_dict = self.compute_loss(
#                 data, eval_mode=True)
#         except Exception as e:
#             print(e)

#         for (k, v) in eval_dict.items():
#             eval_dict[k] = v.item()

#         return eval_dict
    
#     def render_visdata(self, data, resolution, it, out_render_path):
#         (img, mask, world_mat, camera_mat, scale_mat, imgs, world_mats, camera_mats, scale_mats,img_idx) = \
#             self.process_data_dict(data)
#         h, w = resolution
        
#         p_loc, pixels = arange_pixels(resolution=(h, w))

#         pixels = pixels.to(self.device)
#         p_loc = p_loc.float().to(self.device)

#         with torch.no_grad():
#             mask_pred = torch.ones(pixels.shape[0], pixels.shape[1]).bool()

#             rgb_pred = \
#                 [self.model(
#                     pixels_i, camera_mat, world_mat, scale_mat, imgs, world_mats, camera_mats, scale_mats, 'unisurf', 
#                     add_noise=False, eval_=True, it=it)['rgb']
#                     for ii, pixels_i in enumerate(torch.split(p_loc, 1024, dim=1))]
           
#             rgb_pred = torch.cat(rgb_pred, dim=1).cpu()
#             p_loc = p_loc.cpu().long()
#             p_loc1 = p_loc[mask_pred]
#             img_out = (255 * np.zeros((h, w, 3))).astype(np.uint8)
#             img1 = (255 * np.zeros((h, w, 3))).astype(np.float)
#             img1[p_loc1[:, 1], p_loc1[:, 0]] = rgb_pred
            
#             # set_trace()
#             psnr = calculate_psnr(img.squeeze(0).permute(1, 2, 0).cpu().numpy(), 
#                 img1, 
#                 mask.squeeze(0).permute(1, 2, 0).cpu()
#             )

#             with open(self.psnr_path, "a+") as f:
#                 f.write("Iteration {}, psnr = {}\n".format(it, psnr))

#             if mask_pred.sum() > 0:
#                 rgb_hat = rgb_pred[mask_pred].detach().cpu().numpy()
#                 rgb_hat = (rgb_hat * 255).astype(np.uint8)
#                 img_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
#             # set_trace()
#             img1 = Image.fromarray(
#                 (img_out).astype(np.uint8)
#             ).convert("RGB").save(
#                 os.path.join(out_render_path, '_unisurf.png')
#             )# %d是none

#         with torch.no_grad():
#             mask_pred = torch.ones(pixels.shape[0], pixels.shape[1]).bool()
#             p_loc = p_loc.float().to(self.device)

#             rgb_pred = \
#                 [self.model(
#                     pixels_i, camera_mat, world_mat, scale_mat,imgs, world_mats, camera_mats, scale_mats, 'phong_renderer', 
#                     add_noise=False, eval_=True, it=it)['rgb']
#                     for ii, pixels_i in enumerate(torch.split(p_loc, 1024, dim=1))]
           
#             rgb_pred = torch.cat(rgb_pred, dim=1).cpu()
#             p_loc = p_loc.cpu().long()
#             p_loc1 = p_loc[mask_pred]
#             img_out = (255 * np.zeros((h, w, 3))).astype(np.uint8)

#             if mask_pred.sum() > 0:
#                 rgb_hat = rgb_pred[mask_pred].detach().cpu().numpy()
#                 rgb_hat = (rgb_hat * 255).astype(np.uint8)
#                 img_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
        
#             img1 = Image.fromarray(
#                 (img_out).astype(np.uint8)
#             ).convert("RGB").save(
#                 os.path.join(out_render_path, '2_phong.png')
#             )
            
#         return img_out.astype(np.uint8)

#     def process_data_dict(self, data):
#         ''' Processes the data dictionary and returns respective tensors

#         Args:
#             data (dictionary): data dictionary
#         '''
#         device = self.device

#         # Get "ordinary" data
       
#         img = data.get('img').to(device)
#         img_idx = data.get('img.idx')
#         batch_size, _, h, w = img.shape
#         mask_img = data.get('img.mask', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)
#         world_mat = data.get('img.world_mat').to(device)
#         camera_mat = data.get('img.camera_mat').to(device)
#         scale_mat = data.get('img.scale_mat').to(device)
#         imgs = torch.stack(data.get('all_imgs')).squeeze(1).to(device)
#         world_mats = torch.stack(data.get('all_world_mat')).squeeze(1).to(device)
#         camera_mats = torch.stack(data.get('all_camera_mat')).squeeze(1).to(device)
#         scale_mats = torch.stack(data.get('all_scale_mat')).squeeze(1).to(device)

#         return (img, mask_img, world_mat, camera_mat, scale_mat, imgs, world_mats, camera_mats, scale_mats, img_idx)

#     def compute_loss(self, data, eval_mode=False, it=None):
#         ''' Compute the loss.

#         Args:
#             data (dict): data dictionary
#             eval_mode (bool): whether to use eval mode
#             it (int): training iteration
#         '''
#         n_points = self.n_eval_points if eval_mode else self.n_training_points
#         (img, mask_img, world_mat, camera_mat, scale_mat, imgs, world_mats, camera_mats, scale_mats, img_idx) = self.process_data_dict(data)

#         # Shortcuts
#         device = self.device
#         batch_size, _, h, w = img.shape
#         # h, w = 1200, 1600
#         #set_trace()

#         # Assertions
#         assert(((h, w) == mask_img.shape[2:4]) and
#                (n_points > 0))

#         # Sample pixels
#         if n_points >= h*w:
#             p = arange_pixels((h, w), batch_size)[1].to(device)
#             mask_gt = mask_img.bool().reshape(-1)
#             pix = None
#         else:
#             p, pix = sample_patch_points(batch_size, n_points,
#                                     patch_size=1.,
#                                     image_resolution=(h, w),
#                                     continuous=False,
#                                     )
#             p = p.to(device) 
#             pix = pix.to(device)
#             # pix范围为[1600, 1200]
#             # set_trace()
#             mask_gt = get_tensor_values(mask_img, pix.clone()).bool().reshape(-1)
#         out_dict = self.model(
#             pix, camera_mat, world_mat, scale_mat, imgs, world_mats, camera_mats, scale_mats,
#             self.rendering_technique, it=it, mask=mask_gt, 
#             eval_=eval_mode
#         )
        
#         rgb_gt = get_tensor_values(img, pix.clone())
#         # set_trace()
#         loss_dict = self.loss(out_dict['rgb'], rgb_gt, out_dict['normal'], out_dict['mask_pred'], mask_gt)
#         return loss_dict
