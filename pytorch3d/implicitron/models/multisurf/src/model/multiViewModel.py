import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image

# from decoder.unisurfDecoder import OccupancyNetwork
# from decoder.baselineDecoder import OccupancyNetwork
from decoder.condMLPDecoder import condMLPDecoder
# from decoder.cllDecoder import OccupancyNetwork

from encoder.UNetEncoder import UNetEncoder

from integrator.maxPoolItg import maxPoolItg, maxPoolGlobal
from integrator.integrator import AngleWeightedfeatureaggregation as avg_agg

from rendUtils.renderFunc import renderFunc
from utils import rendUtils
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
        self.device = device

        assert(mode in ["train", "test"], f"Mode must be train/test, get {mode}")
        self.mode = mode

        self.n_points_sampled = cfg["model"]["n_points_sampled"]
        self.img_res = cfg["dataloading"].get("resize_res", cfg["dataloading"]["img_size"])        
        self.psnr_path = cfg["training"]["psnr_path"] if self.mode == "train" else cfg["testing"]["psnr_path"]

        self.feature_encoder = UNetEncoder(cfg).to(self.device)
        self.decoder = condMLPDecoder(cfg).to(self.device)
        self.local_integrator = avg_agg(cfg).to(self.device)
        self.global_integrator = maxPoolGlobal(cfg).to(self.device)

    def forward(self, data, it):
        all_images, all_masks, all_depths, all_cameras, neigbor_list = self._preprocess_data(data)

        feature_maps, global_feature = self.feature_encoder(all_images, neigbor_list, mask=all_masks)

        sampled_p_ndc, sampled_pixels = rendUtils.sample_pixels(n_points_sampled=self.n_points_sampled, img_res=self.img_res, mask=all_masks, view_idx_list=neigbor_list)


        

        pass

    def _preprocess_data(self, data):
        pass

    def plot(self):
        pass
