import os
import torch
import numpy as np
import glob
import pdb
from ipdb import set_trace
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class DTUDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, device: str="cuda"):
        self.root_dir = cfg["dataloading"]["path"]
        self.num_views = cfg["dataloading"]["n_views"]
        self.img_res = cfg["dataloading"]["img_size"]
        self.resize_res = cfg["dataloading"].get("resize_res", "None")
        
        self.scenes_path = sorted(glob.glob(os.path.join(self.root_dir, "scan*/")))
        self.num_scenes = len(self.scenes_path)
        self.num_total_imgs = self.num_scenes * self.num_views
        self.device = device

        self.scenes = []
        self.data = dict()

        for path in self.scenes_path:
            scene_idx = path.split("/")[-2]
            self.scenes.append(scene_idx)
            self.data[scene_idx] = {
                "imgs_num": None,
                "content": dict()
            }
        
        print("loading data....")
        for scene in tqdm(self.data.keys(), desc="Scenes loaded"):
            self._read_imgs(scene)
            self._read_masks(scene)
            self._read_cameras(scene)
            self._read_neighbors(scene)
    
    def _read_imgs(self, scene: str):
        """
        Load the images of a scene
        
        Args:
          scene (str): The scene name.
        """
        scene_path = os.path.join(self.root_dir, scene)
        imgs_path = sorted(glob.glob(os.path.join(
            scene_path, "image/0*.png"
        )))

        content = self.data[scene]["content"]
        for idx in range(self.num_views):
            content[idx] = {
                "img_path": imgs_path[idx],
                "image": self._load_single_img(imgs_path[idx])
            }

    def _read_masks(self, scene: str):
        """
        Load the masks of a scene
        
        Args:
          scene (str): The scene name.
        """
        scene_path = os.path.join(self.root_dir, scene)
        masks_path = sorted(glob.glob(os.path.join(
            scene_path, "mask/0*.png"
        )))

        content = self.data[scene]["content"]
        for idx in range(self.num_views):
            content[idx]["mask_path"] = masks_path[idx]
            content[idx]["mask"] = self._load_single_mask(masks_path[idx])

    def _read_cameras(self, scene: str):
        """
        Load the camera matrices from the .npz file
        
        Args:
          scene (str): The scene name.
        """
        scene_path = os.path.join(self.root_dir, scene)
        cameras = np.load(os.path.join(
            scene_path, "cameras.npz"
        ))

        content = self.data[scene]["content"]
        
        for idx in range(self.num_views):
            if self.resize_res is not None:
                scaling_factor = 1 / (self.img_res[0] / self.resize_res[0])
                world_mat_old = torch.from_numpy(cameras["world_mat_{}".format(idx)])
                world_mat = torch.tensor([[scaling_factor, 0, 0], [0, scaling_factor, 0], [0, 0, 1]]) @ world_mat_old[:3, :]
                world_mat = torch.cat([world_mat, torch.tensor([[0, 0, 0, 1]])], dim=0)
                world_mat = world_mat
            else:
                world_mat = torch.from_numpy(cameras["world_mat_{}".format(idx)])

            content[idx]["world_mat"] = world_mat
            content[idx]["scale_mat"] = torch.from_numpy(cameras["scale_mat_{}".format(idx)])
            content[idx]["camera_mat"] = torch.from_numpy(cameras["camera_mat_{}".format(idx)])

    def _read_neighbors(self, scene: str):
        pairs_path = os.path.join(self.root_dir, "pair.txt")
        content = self.data[scene]["content"]

        is_pairs_exist = os.path.exists(pairs_path)
        if not is_pairs_exist:
            print("No pair.txt found in {0}".format(self.root_dir))
            for idx in range(self.num_views):
                content[idx]["neighbor_list"] = torch.arange(self.num_views)                
        else:
            with open(pairs_path, "r") as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    idx = int(f.readline().rstrip())                    
                    content[idx]["neighbor_list"] = torch.tensor([int(x) for x in f.readline().rstrip().split()[1::2]])

    def _load_single_img(self, img_path: str):
        """
        Load a single image from a path, resize if necessary and convert it to a tensor
        
        Args:
          img_path (str): the path to the image file
        
        Returns:
          A tensor of shape (3, h, w) (after resize)
        """
        image = Image.open(img_path).convert("RGB")
        image = transforms.ToTensor()(image)
        if self.resize_res is not None:
            image = transforms.Resize(min(self.resize_res))(image)
        return image

    def _load_single_mask(self, mask_path: str):
        """
        Loads a single mask from a given path, resize if necessary and convert it to a tensor
        
        Args:
          mask_path (str): The path to the mask file.
        
        Returns:
          A tensor of shape (1, h, w) (after resize)
        """
        mask = Image.open(mask_path)
        mask = transforms.ToTensor()(mask)
        if self.resize_res is not None:
            mask = transforms.Resize(min(self.resize_res))(mask)
        mask = mask[:1, :, :]
        return mask

    def __len__(self):
        return self.num_total_imgs
    
    def __getitem__(self, idx: int):
        """
        Given a scene index, return a dictionary of data for that scene
        
        Args:
          idx (int): the index of the sample in the dataset
        
        Returns:
            view_idx: the index of the current view
            all_imgs: a tensor of shape (num_views, C, H, W)
            all_world_mats: a tensor of shape (num_views, 4, 4)
            all_camera_mats: a tensor of shape (num_views, 4, 4)
            all_scale_mats: a tensor of shape (num_views, 4, 4)
            all_masks: a tensor of shape (num_views, 1, H, W)
            neighbor_list: a tensor of shape (num_neighbors)
        """
        scene_id = idx // self.num_views
        view_id = idx % self.num_views

        content = self.data[self.scenes[scene_id]]["content"]

        all_imgs = torch.stack([content[v]["image"] for v in range(self.num_views)])
        all_world_mats = torch.stack([content[v]["world_mat"] for v in range(self.num_views)])
        all_camera_mats = torch.stack([content[v]["camera_mat"] for v in range(self.num_views)])
        all_scale_mats = torch.stack([content[v]["scale_mat"] for v in range(self.num_views)])
        all_masks = torch.stack([content[v]["mask"] for v in range(self.num_views)])
        neighbor_list = content[view_id]["neighbor_list"]

        data = {
            "view_idx": view_id,
            "all_imgs": all_imgs, 
            "all_world_mats": all_world_mats,
            "all_camera_mats": all_camera_mats,
            "all_scale_mats": all_scale_mats,
            "all_masks": all_masks,
            "neighbor_list": neighbor_list
        }

        return data

def get_dataloader(cfg, mode='train', shuffle=True, device: str="cuda"):
    """
    Given a configuration file, return a dataloader for the dataset
    
    Args:
      cfg: the config file
      mode: train or test. Defaults to train
      shuffle: If True, shuffle the dataset every epoch. Defaults to True
    
    Returns:
      A dataloader object that can be iterated over to get the input and target for each batch.
    """
    batch_size = cfg["dataloading"]["batch_size"]
    n_workers = cfg["dataloading"]["n_workers"]

    dataset = DTUDataset(cfg, device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=shuffle,
        collate_fn=collate_remove_none
    )

    return dataloader

def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)