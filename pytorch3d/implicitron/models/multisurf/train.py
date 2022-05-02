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

logger_py = logging.getLogger(__name__)

# Fix seeds
np.random.seed(42)
torch.manual_seed(42)

# Arguments
parser = argparse.ArgumentParser(
    description='Training of UNISURF model'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')

args = parser.parse_args()
cfg = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# params
out_dir = cfg['training']['out_dir']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after
batch_size = cfg['training']['batch_size']
n_workers = cfg['dataloading']['n_workers']
lr = cfg['training']['learning_rate']

# init dataloader
train_loader = get_dataloader(cfg, mode='train')
if "dataset_name" not in cfg["dataloading"] or cfg["dataloading"]["dataset_name"] == "co3d_singlesequence":
    test_loader = get_dataloader(cfg, mode='test')
else:
    cfg_test = copy.deepcopy(cfg)
    cfg_test["dataloading"]["dataset_name"] = "co3d_singlesequence"
    test_loader = get_dataloader(cfg_test, mode='test')

iter_test = iter(test_loader)
# iter_test = iter(train_loader)
data_test = next(iter_test)

test_loader_trainset = get_dataloader(cfg, mode="train")
iter_train = iter(test_loader_trainset)
data_train = next(iter_train)

# init network
# model_cfg = cfg
# model = mdl.OccupancyNetwork(model_cfg)
# print(model)

model = BaseModel(cfg, device=device)
# set_trace()
loss = Loss(cfg)

# init renderer
# rendering_cfg = cfg['rendering']
# renderer = mdl.Renderer(model, rendering_cfg, device=device)

# init optimizer
weight_decay = cfg['training']['weight_decay']
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

# init training
training_cfg = cfg['training']
# trainer = trainer.Trainer(renderer, optimizer, training_cfg, device=device)
trainer = trainer.Trainer(cfg=cfg, model=model, loss=loss, optimizer=optimizer)

# init checkpoints and load
checkpoint_io = checkpoints.CheckpointIO(out_dir, model=model, optimizer=optimizer)

try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, cfg['training']['scheduler_milestones'],
    gamma=cfg['training']['scheduler_gamma'], last_epoch=epoch_it)

logger = SummaryWriter(os.path.join(out_dir, 'logs'))
    
# init training output
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
visualize_every = cfg['training']['visualize_every']
render_path = os.path.join(out_dir, 'rendering')
if visualize_every > 0:
    visualize_skip = cfg['training']['visualize_skip']
    visualize_path = os.path.join(out_dir, 'visualize')
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)


# Print model
nparameters = sum(p.numel() for p in model.parameters())
logger_py.info(model)
logger_py.info('Total number of parameters: %d' % nparameters)
t0b = time.time()


while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
        loss_dict = trainer.train_step(batch, it)
        loss = loss_dict['loss']
        rgb_loss = loss_dict['fullrgb_loss']
        normal_loss = loss_dict['grad_loss']
        mask_loss = loss_dict['mask_loss']
        mh_loss = loss_dict["mh_loss"]
        depth_loss = loss_dict["depth_loss"]
        iou_loss = loss_dict["iou_loss"]
        depth_gt_loss = loss_dict["depth_gt_loss"]
        metric_val_best = loss
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f, rgb=%.4f, normal=%.4f, mask=%.4f, mh=%.4f, depth=%.4f,iou=%.4f,depth_gt=%.4f time=%.4f'
                           % (epoch_it, it, loss, rgb_loss, normal_loss, mask_loss, mh_loss, depth_loss, iou_loss, depth_gt_loss,time.time() - t0b))
            logger_py.info('[Epoch %02d] it=%03d, loss=%.4f, time=%.4f'
                           % (epoch_it, it, loss, time.time() - t0b))
            t0b = time.time()
            for l, num in loss_dict.items():
                logger.add_scalar('train/'+l, num.detach().cpu(), it)
        
        if visualize_every > 0 and (it % visualize_every)==1:
            logger_py.info("Rendering")
            print("I'm going to huatu")

            out_render_path = os.path.join(render_path, '%04d_vis_train' % it)
            if not os.path.exists(out_render_path):
                os.makedirs(out_render_path)
            val_rgb = model.plot(data_train, cfg['training']['vis_resolution'], it, out_render_path)
            try:
                data_train = next(iter_train)
            except StopIteration:
                iter_train = iter(test_loader_trainset)
                # iter_test = iter(train_loader)
                data_train = next(iter_train)

            out_render_path = os.path.join(render_path, '%04d_vis_test' % it)
            if not os.path.exists(out_render_path):
                os.makedirs(out_render_path)
            val_rgb = model.plot(data_test, cfg['training']['vis_resolution'], it, out_render_path)
            try:
                data_test = next(iter_test)
            except StopIteration:
                iter_test = iter(test_loader)
                # iter_test = iter(train_loader)
                data_test = next(iter_test)
            # logger.add_image('rgb', val_rgb, it)
        
        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            logger_py.info('Saving checkpoint')
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            logger_py.info('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
    scheduler.step()