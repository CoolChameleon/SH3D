import torch
from torch import nn
from torch.nn import functional as F
from ipdb import set_trace

class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.full_weight = cfg["training"]["lambda_l1_rgb"]
        self.grad_weight = cfg["training"]["lambda_normals"]
        self.occ_prob_weight = cfg["training"]["lambda_occ_prob"]
        self.depth_diff_weight = cfg["training"]["lambda_depth_diff"]
        self.depth_loss_weight = cfg["training"]["lambda_iou"]
        self.mh_loss_weight = cfg["training"]["lambda_mh"]
        self.iou_loss_weight = cfg["training"]["lambda_iou"]
        self.l1_loss = nn.L1Loss(reduction='sum')
    
    def get_rgb_full_loss(self,rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(rgb_values.shape[1])
        return rgb_loss

    def get_rgb_mask_loss(self,rgb_values, rgb_gt, mask):
        rgb_mask_loss = self.l1_loss(rgb_values[mask], rgb_gt[mask]) / float(rgb_values[mask].shape[0] + 0.01)
        return rgb_mask_loss

    def get_mask_loss(self, mask_pred, mask_gt, mask_occ):
        # set_trace()
        mask_loss = self.l1_loss((mask_occ + 0.5).squeeze(1).squeeze(1), mask_gt[mask_pred.squeeze(0)]) / (mask_occ.shape[0] + 0.01)
        return mask_loss

    def get_smooth_loss(self, diff_norm):
        if diff_norm is None or diff_norm.shape[0]==0:
            return torch.tensor(0.0).cuda().float()
        else:
            return diff_norm.mean()

    # def forward(self, rgb_pred, rgb_gt, diff_norm, mask_pred, mask_gt, mask_occ):
    def forward(self, out_dict):
        rgb_gt = out_dict["rgb_gt"]
        mask_gt = out_dict["mask_gt"]
        mask_pred = out_dict['mask_pred']
        rgb_pred = out_dict["rgb_pred"]
        diff_norm = out_dict["normal"]
        mask_occ = out_dict["surface_points_occ"]
        
        mh_loss = out_dict["mh_loss"]


        rgb_gt = rgb_gt.cuda()
        mask_gt = mask_gt.squeeze(0).squeeze(1).cuda()
        # set_trace()

        mask = mask_gt.bool() & mask_pred
        
        if self.full_weight != 0.0:
            rgb_full_loss = self.get_rgb_full_loss(rgb_pred, rgb_gt)
            # rgb_full_loss = self.get_rgb_mask_loss(rgb_pred, rgb_gt, mask)           
        else:
            rgb_full_loss = torch.tensor(0.0).cuda().float()

        if diff_norm is not None and self.grad_weight != 0.0:
            grad_loss = self.get_smooth_loss(diff_norm)
        else:
            grad_loss = torch.tensor(0.0).cuda().float()
        mask_loss = self.get_mask_loss(mask_pred, mask_gt, mask_occ)
    
        depth_loss = out_dict["depth_loss"]
        iou_loss =  out_dict["iou_loss"]
        depth_pred = out_dict["depth_pred"]
        depth_gt = out_dict["depth_gt"]
        mask_gt = depth_gt != 0
        depth_gt_loss = (depth_pred[mask_gt] - depth_gt[mask_gt]).abs().sum()
        # set_trace()
        loss = self.full_weight * rgb_full_loss + \
               self.grad_weight * grad_loss -self.mh_loss_weight * mh_loss + self.iou_loss_weight * iou_loss + self.depth_diff_weight * depth_loss + self.depth_loss_weight * depth_gt_loss
        if torch.isnan(loss):
            breakpoint()

        return {
            'loss': loss,
            'fullrgb_loss': rgb_full_loss,
            'grad_loss': grad_loss,
            'mask_loss': mask_loss,
            "mh_loss": mh_loss,
            "depth_loss": depth_loss,
            "iou_loss": iou_loss,
            "depth_gt_loss": depth_gt_loss,
        }


