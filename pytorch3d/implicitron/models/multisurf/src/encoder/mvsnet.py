from ipdb import set_trace
import torch
import torch.nn as nn
from inplace_abn import InPlaceABN
from inplace_abn.abn import ABN
# from torch.nn import BatchNorm1d as InPlaceABN
import torch.nn.functional as F
import torchsnooper
# from Pytorch_Memory_Utils.gpu_mem_track import MemTracker
# gpu_tracker = MemTracker(path="./gpu_mem_track")

from kornia.utils import create_meshgrid

def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """
    # set_trace()

    if src_grid==None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
        
        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory



        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(src_feat, src_grid.view(B, D, W_pad * H_pad, 2),
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=ABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=ABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))
        #print(self.conv0[0].conv.weight.dtype)
        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        x = self.toplayer(x) # (B, 32, H//4, W//4)
        return x

class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=ABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        # x = self.conv12(x)
        return x

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=ABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))

class MVSNet(nn.Module):
    def __init__(self,
                 num_groups=1,
                 norm_act=ABN,
                 levels=1):
        super(MVSNet, self).__init__()
        self.levels = levels  # 3 depth levels
        self.n_depths = [128,32,8]
        self.G = num_groups  # number of groups in groupwise correlation
        self.feature = FeatureNet()

        self.N_importance = 0
        self.chunk = 1024

        self.cost_reg_2 = CostRegNet(32+9, norm_act)

    def build_volume_costvar(self, feats, proj_mats, depth_values, pad=0):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]

        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        del ref_feats

        in_masks = torch.ones((B, 1, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_feat, proj_mat) in enumerate(zip(src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks += in_mask#.double()

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / in_masks
        img_feat = volume_sq_sum * count - (volume_sum * count) ** 2
        del volume_sq_sum, volume_sum, count

        return img_feat, in_masks

    # @torchsnooper.snoop()
    def build_volume_costvar_img(self, imgs, feats, proj_mats, depth_values, pad=0):
        # feats: (B, V, C=32, h, w)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (1, D)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]
        # set_trace()
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        # ref_feats.shape = [B, C, h, w] src_feats = [B, V - 1, C, h, w]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)
            # pad = 24, ref_feats.shape = [B, C, 176, 208]
        
        # gpu_tracker.track()
        try:
            img_feat = torch.empty((B, 9 + 32, D, *ref_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        except:
            set_trace()
        # print(img_feat.shape)
        # [B, 9 + 32, D, 176, 208]
        app_feat = torch.empty((B, 32, D, *ref_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B * V, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, V,-1,H,W).permute(1, 0, 2, 3, 4)
        # gpu_tracker.track()
        # print(imgs.shape)
        # [V, B, 3, H, W]

        # print(imgs[0].unsqueeze(2).shape)
        # [B, 3, 1, H, W]
        # print(imgs[0].unsqueeze(2).expand(-1, -1, D, -1, -1).shape)
        # [B, 3, D, H, W]
        img_feat[:, :3, :, pad:H + pad, pad:W + pad] = imgs[0].unsqueeze(2).expand(-1, -1, D, -1, -1)
        # img_feat的前三个元素是reference图片
        # gpu_tracker.track()

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        # gpu_tracker.track()

        del ref_feats

        in_masks = torch.ones((B, V, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_img, src_feat, proj_mat) in enumerate(zip(imgs[1:], src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            # print(warped_volume.shape, grid.shape)
            # [B, C, D, H + 2 * pad, W + 2 * pad], [1, D, W + 2 * pad, H + 2 * pad, 2]
            img_feat[:, (i + 1) * 3:(i + 2) * 3], _ = homo_warp(src_img, proj_mat, depth_values, src_grid=grid, pad=pad)
            # img_feat的第456、789个元素分别是第1张src_img和第2张src_img经过homo_warp生成的image

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks[:, i + 1] = in_mask#.double()
            # print(in_mask.shape)
            # [1, 1, D, H + pad * 2, W + pad * 2]

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
            # gpu_tracker.track()
        del src_feats, proj_mats

        count = 1.0 / torch.sum(in_masks, dim=1, keepdim=True)
        # print(count)
        
        try:
            img_feat[:, -32:] = volume_sq_sum * count - (volume_sum * count) ** 2
            app_feat[0, :] = volume_sum * count
            # print(app_feat)
            # set_trace()
        except RuntimeError:
            # set_trace()
            pass
        # b = img_feat[:, -32:].cpu().numpy()
        # np.save("volume.npy",b)
        # with open('./log_volume.txt', 'w+') as f:
        #     #f.write(str(img_feat[0, -32:, 100, 170, 200]))
        #     #f.write(str(img_feat[:, -32:].shape))
        #     f.write(str(img_feat[:, -32:][0]))
            #f.write('\n==============================\n')
        # img_feat的后32维是ref_feature和（src_feature的warped_volume）的方差
        del volume_sq_sum, volume_sum, count
        # gpu_tracker.track()
        return img_feat, in_masks, app_feat

    def forward(self, imgs, proj_mats, near_far, pad=0,  return_color=False, lindisp=False):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        # near_far (B, V, 2)

        # print(pad)
        # pad = 24
        # print(imgs.shape)
        B, V, _, H, W = imgs.shape
        near_far = near_far[0]
        # print("!!!!!!!!!!!!!!!!!!!!!!!!v = ", V)
        # print(imgs.shape, proj_mats.shape, near_far.shape)
        # [1, 3, 3, 512, 640] [1, 3, 3, 4] [2]

        imgs = imgs.reshape(B * V, 3, H, W)
        feats = self.feature(imgs)  # (B*V, 8, H, W)->(B*V, 16, H//2, W//2)->(B*V, 32, H//4, W//4)

        imgs = imgs.view(B, V, 3, H, W)

        #h=H/4 w=W/4
        feats_l = feats  # (B*V, C, h, w)

        feats_l = feats_l.view(B, V, *feats_l.shape[1:])  # (B, V, C, h, w)


        D = 128
        # 切片数量
        t_vals = torch.linspace(0., 1., steps=D, device=imgs.device, dtype=imgs.dtype)  # (B, D)
        # print(t_vals.shape)
        # [128]
        near, far = near_far  # assume batch size==1
        if not lindisp:
            depth_values = near * (1. - t_vals) + far * (t_vals)
        else:
            depth_values = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        # print(depth_values.shape)
        # [128]

        depth_values = depth_values.unsqueeze(0)
        # print(depth_values.shape)
        # [1, 128]
        # volume_feat, in_masks = self.build_volume_costvar(feats_l, proj_mats, depth_values, pad=pad)
        volume_feat, in_masks,app_feat = self.build_volume_costvar_img(imgs, feats_l, proj_mats, depth_values, pad=pad)
        # print(volume_feat.shape)
        # [B, 41, D, H + 2p, W + 2p]
        if return_color:
            feats_l = torch.cat((volume_feat[:,:V*3].view(B, V, 3, *volume_feat.shape[2:]),in_masks.unsqueeze(2)),dim=2)


        volume_feat = self.cost_reg_2(volume_feat)  # (B, 1, D, h, w)
        # print(volume_feat.shape)
        # [B, 8, D, H + 2p, W + 2p]

        volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])
        #print(volume_feat.shape)[1,8,128,176,208]

        return volume_feat, feats_l, depth_values, app_feat
