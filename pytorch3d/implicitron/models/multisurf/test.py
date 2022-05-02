import os
import sys
import logging
import time
import shutil
import argparse
import copy

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from ipdb import set_trace

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.dataloading.CO3Dloader import get_dataloader
from src.dataloading.configloading import load_config
# from dataloading_l.configloading import load_config
# from dataloading_l.dataloader import get_dataloader
#import model as mdl

# from src.model.geomodel import BaseModel
# from src.model.baseModel import BaseModel
# from src.model.condModel import BaseModel
from src.model.maskModel import BaseModel
from src.model import checkpoints
from src.loss.loss import Loss
from src import trainer

# Fix seeds
np.random.seed(42)
torch.manual_seed(42)

# Arguments
parser = argparse.ArgumentParser(
    description='Testing of Multisurf model'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')


args = parser.parse_args()
cfg = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg["testing"].get("out_dir", "./test")
os.makedirs(out_dir, exist_ok=True)

cfg["dataloading"]["dataset_name"] = "co3d_singlesequence"
data_loader = get_dataloader(cfg, mode="test", shuffle=False)

# init network
model = BaseModel(cfg, device=device)

# init checkpoints and load
checkpoint_io = checkpoints.CheckpointIO(out_dir, model=model)
checkpoint_name = cfg["testing"].get("checkpoint_name", "model.pt")

try:
    load_dict = checkpoint_io.load(checkpoint_name)
except FileExistsError:
    print(f"Fatal Error: Checkpoint{checkpoint_name} Not Found!")
    exit(1)

it = load_dict.get('it', -1)

vis_resolution = cfg["testing"].get("vis_resolution", [200, 200])
for idx, data in enumerate(tqdm(data_loader, desc="Ploting Pic:")):
    render_path = os.path.join(out_dir, "frame%04d" % idx)
    os.makedirs(render_path, exist_ok=True)
    model.plot(data, cfg["testing"]["vis_resolution"], it, render_path)
