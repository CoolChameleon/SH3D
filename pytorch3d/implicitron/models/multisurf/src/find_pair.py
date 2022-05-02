from dataloading.configloading import load_config
cfg = load_config("/home/shj20/pytorch3d/pytorch3d/implicitron/models/multisurf/configs/unisurf_co3d/0425_multi_cond_debug_mhLoss.yaml")

from dataloading.co3d.dataset_zoo import dataset_zoo
from dataloading.CO3Dloader import get_dataloader, CO3DDataset
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from ipdb import set_trace
import os
# train_loader = get_dataloader(cfg, mode='train')
test_loader = get_dataloader(cfg, mode='test')
# for batch in train_loader:

#     set_trace()
for test in test_loader:
    set_trace()