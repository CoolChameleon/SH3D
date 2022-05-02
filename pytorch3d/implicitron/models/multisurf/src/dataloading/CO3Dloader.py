import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import glob
from ipdb import set_trace
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random

from co3d.dataset_zoo import dataset_zoo

class CO3DDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split: str="train"):
        self.category = cfg["dataloading"]["category"]
        self.num_views = cfg["dataloading"]["n_views"]
        self.num_sample_neighbor = cfg["dataloading"]["n_sample"]
        self.dataset_root = cfg["dataloading"]["path"]

        self.load_masked_image = cfg["dataloading"].get("load_masked_image", False)
        self.dataset_name = cfg["dataloading"].get("dataset_name", "co3d_singlesequence")
        if self.dataset_name == "co3d_singlesequence":
            self.test_restrict_sequence_id = cfg["dataloading"].get("test_restrict_sequence_id", 0)
        else:
            self.test_restrict_sequence_id = -1
        self.test_on_train = cfg["dataloading"].get("test_on_train", True)

        datasets = dataset_zoo(
            category=self.category,
            assert_single_seq=False,
            dataset_name=self.dataset_name,
            test_on_train=self.test_on_train,
            load_point_clouds=True,
            test_restrict_sequence_id=self.test_restrict_sequence_id
        )

        self.datasets = datasets[split]
        self.scene_loaded = []
        self.data = {}

        for i in range(len((self.datasets))):
            try:
                origin_data = self.datasets.frame_annots[i]["frame_annotation"]
                scene_name = origin_data.sequence_name
                if scene_name not in self.data:
                    self.data[scene_name] = {}
                    self.data[scene_name]["index_list"] = []

                if "infeasible_list" not in self.data[scene_name]:
                    self.data[scene_name]["infeasible_list"] = self._load_infeasible_list(scene_name)  

                infeasible_list = self.data[scene_name]["infeasible_list"]

                if origin_data.image.path.split("/")[-1] not in infeasible_list:
                    self.data[scene_name]["index_list"].append(i)
                
            except IOError:
                continue
        
        self.scene_list = []
        self.total_imgs = 0
        for scene, data in self.data.items():
            # if len(data["index_list"]) >= self.num_views:
            if len(data["index_list"]) >= self.num_sample_neighbor:
                self.scene_list.append(scene)
                self.total_imgs += len(data["index_list"])
        print(f"Using {self.total_imgs} images")
        print(f"Using {len(self.scene_list)} scenes:")
        # for scene in self.scene_list:
        #     print(f"\tscene {scene} has {len(self.data[scene]['index_list'])} imgs")

    def _load_infeasible_list(self, scene_name: str):
        # set_trace()
        infeasible_list_path = os.path.join(self.dataset_root, self.category, scene_name, "infeasible.txt")
        if not os.path.exists(infeasible_list_path):
            return []
        with open(infeasible_list_path, "r") as f:
            infeasible_str = f.readline()
            infeasible_list = infeasible_str.split(",")
        return infeasible_list

    def _get_camera_mats(self, camera, h: int, w: int):
        principal_point = camera.principal_point
        focal_length = camera.focal_length
        half_image_size_wh_orig = focal_length.new_tensor([[w / 2, h / 2]])
        principal_point_px = -1.0 * (principal_point - 1.0) * half_image_size_wh_orig
        focal_length_px = focal_length * half_image_size_wh_orig
        fx, fy = focal_length_px.unbind(1)
        px, py = principal_point_px.unbind(1)
        K = fx.new_zeros(1, 4, 4)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = px
        K[:, 1, 2] = py
        K[:, 2, 2] = 1.0
        K[:, 3, 3] = 1.0
        # print(principal_point)
        R = camera.R  # [1,3,3]
        T = camera.T  # [1,3]
        K_out = torch.cat((torch.cat((R, T.unsqueeze(2)), 2), T.new_zeros((1, 1, 4))), 1)
        K_out[0, -1, -1] = 1
        # print(K,"K")
        # print(K_out,"K_OUT")
        world_mat = torch.bmm(K, K_out).squeeze(0)
        camera_mat = torch.eye(4)
        scale_mat = torch.eye(4)
        return world_mat, camera_mat, scale_mat, camera, principal_point, focal_length, R, T

    def _load_data(self, scene_name: str):
        if "content_list" in self.data[scene_name]:
            return
        if scene_name in self.scene_loaded:
            return 
        self.data[scene_name]["content_list"] = []
        # for idx in random.sample(self.data[scene_name]["index_list"], self.num_views):
        for idx in self.data[scene_name]["index_list"]:
            origin_data = self.datasets[idx]
            processed_data = {}
            processed_data["image"] = origin_data.image_rgb
            processed_data["mask"] = origin_data.fg_probability > 0.5

            if self.load_masked_image:
                processed_data["image"][~processed_data["mask"].repeat(3, 1, 1)] = 0

            processed_data["depth_map"] = origin_data.depth_map     # [1,200,200]
            processed_data["depth_mask"] = origin_data.depth_mask   # [1,200,200]

            h, w = origin_data.image_rgb.shape[1:]
            world_mat, camera_mat, scale_mat, camera, principal_point, focal_length, R, T = self._get_camera_mats(origin_data.camera, h, w)

            processed_data["world_mat"] = world_mat
            processed_data["camera_mat"] = camera_mat
            processed_data["scale_mat"] = scale_mat
            processed_data["camera_ndc"] = camera
            processed_data["principal_point"] = principal_point
            processed_data["focal_length"] = focal_length
            processed_data["R"] = R
            processed_data["T"] = T
            self.data[scene_name]["content_list"].append(processed_data)
        self.scene_loaded.append(scene_name)
        #print("{} scenes have been loaded".format(len(self.scene_loaded)))

    def __getitem__(self, idx: int):
        # scene_id = idx // self.num_views
        # view_id = idx % self.num_views
        num_imgs_before = 0
        for scene_id in range(len(self.scene_list)):
            num_imgs_before += len(self.data[self.scene_list[scene_id]]["index_list"])
            if num_imgs_before > idx:
                view_id = idx - num_imgs_before + len(self.data[self.scene_list[scene_id]]["index_list"])
                break        
        
        self._load_data(self.scene_list[scene_id])
        content = self.data[self.scene_list[scene_id]]["content_list"]
        num_views = len(content)

        all_imgs = torch.stack([content[v]["image"] for v in range(num_views)])
        all_world_mats = torch.stack([content[v]["world_mat"] for v in range(num_views)])
        all_camera_mats = torch.stack([content[v]["camera_mat"] for v in range(num_views)])
        all_scale_mats = torch.stack([content[v]["scale_mat"] for v in range(num_views)])
        all_masks = torch.stack([content[v]["mask"] for v in range(num_views)])
        # all_camera_ndc = {content[v]["camera_ndc"] for v in range(self.num_views)}
        # all_camera_ndc = [content[v]["camera_ndc"] for v in range(self.num_views)]
        all_principal_point = torch.stack([content[v]["principal_point"] for v in range(num_views)])
        all_focal_length = torch.stack([content[v]["focal_length"] for v in range(num_views)])
        all_R = torch.stack([content[v]["R"] for v in range(num_views)])
        all_T = torch.stack([content[v]["T"] for v in range(num_views)])
        all_depth_map = torch.stack([content[v]["depth_map"] for v in range(num_views)])
        all_depth_mask = torch.stack([content[v]["depth_mask"] for v in range(num_views)])
        
        neighbor_list = self._random_sample_neighbor_list(scene_id, view_id)

        data = {
            "view_idx": view_id,
            "all_imgs": all_imgs, 
            "all_world_mats": all_world_mats,
            "all_camera_mats": all_camera_mats,
            "all_scale_mats": all_scale_mats,
            "all_masks": all_masks,
            "all_principal_point": all_principal_point,
            "all_focal_length": all_focal_length,
            "all_R": all_R,
            "all_T": all_T,
            "neighbor_list": neighbor_list,
            "all_depth_map": all_depth_map,
            "all_depth_mask": all_depth_mask,
        }

        return data


    def _random_sample_neighbor_list(self, scene_id:int, view_id: int):
        num_views = len(self.data[self.scene_list[scene_id]]["index_list"])
        if self.num_sample_neighbor <= num_views - 1:
            neighbor_list = random.sample([i for i in range(num_views) if i != view_id], self.num_sample_neighbor)
        else:
            neighbor_list = [i for i in range(num_views) if i != view_id]
        return torch.tensor(neighbor_list)

    def __len__(self):
        # return len(self.scene_list) * self.num_views
        return self.total_imgs



def get_dataloader(cfg, mode, shuffle=True):
    """
    This is a function that will return a dataloader for you
    
    Args:
      cfg: 
      shuffle: Defaults to True
    
    Returns:
      The data is being parsed into a pytorch dataloader.
    """
    batch_size = cfg["dataloading"]["batch_size"]
    n_workers = cfg["dataloading"]["n_workers"]

    dataset = CO3DDataset(cfg, split=mode)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=shuffle,
        collate_fn=collate_remove_none
    )

    return dataloader

def collate_remove_none(batch):
    """
    Given a batch of samples, it removes the samples that are None
    
    Args:
      batch: a list of items that will be collated together.
    
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    cfg = {"dataloading": {"category": "banana", "n_views": 50, "n_sample": 10, "batch_size": 1, "n_workers": 8}}
    dataloader = get_dataloader(cfg)
    train_data = iter(dataloader)
    data = next(train_data)
    set_trace()
# class CO3DDataset_old(torch.utils.data.Dataset):
#     def __init__(self, cfg, split: str="train"):
#         self.category = cfg["dataloading"]["category"]
#         self.num_views = cfg["dataloading"]["n_views"]
#         self.num_sample_neighbor = cfg["dataloading"]["n_sample"]
#         self.dataset_root = cfg["dataloading"]["path"]

#         self.load_masked_image = False if "load_masked_image" not in cfg["dataloading"] else cfg["dataloading"]["load_masked_image"]

#         self.dataset_name = "co3d_singlesequence" if "dataset_name" not in cfg["dataloading"] else cfg["dataloading"]["dataset_name"]

#         self.test_restrict_sequence_id = -1
#         if self.dataset_name == "co3d_singlesequence":
#             self.test_restrict_sequence_id = 0 if "test_restrict_sequence_id" not in cfg["dataloading"] else cfg["dataloading"]["test_restrict_sequence_id"]

#         self.test_on_train = True if "test_on_train" not in cfg["dataloading"] else cfg["dataloading"]["test_on_train"]

#         # datasets = dataset_zoo(
#         #     category=self.category,
#         #     assert_single_seq=False,
#         #     dataset_name="co3d_multisequence",
#         #     test_on_train=True,
#         #     load_point_clouds=True
#         # )
#         datasets = dataset_zoo(
#             category=self.category,
#             assert_single_seq=False,
#             dataset_name=self.dataset_name,
#             test_on_train=self.test_on_train,
#             load_point_clouds=True,
#             test_restrict_sequence_id=self.test_restrict_sequence_id
#         )

#         self.datasets = datasets[split]
#         # set_trace()
#         self.scene_loaded = []
#         self.data = {}

#         for i in tqdm(range(len((self.datasets)))):
#             try:
#                 origin_data = self.datasets.frame_annots[i]["frame_annotation"]
#                 scene_name = origin_data.sequence_name
#                 if scene_name not in self.data:
#                     self.data[scene_name] = {}
#                     self.data[scene_name]["index_list"] = []

#                 if "infeasible_list" not in self.data[scene_name]:
#                     self.data[scene_name]["infeasible_list"] = self._load_infeasible_list(scene_name)  

#                 infeasible_list = self.data[scene_name]["infeasible_list"]

#                 if origin_data.image.path.split("/")[-1] not in infeasible_list:
#                     self.data[scene_name]["index_list"].append(i)
#                 # self.data[scene_name]["index_list"].append(i)
                
#             except IOError:
#                 continue
        
#         self.scene_list = []
#         for scene, data in self.data.items():
#             if len(data["index_list"]) >= self.num_views:
#                 self.scene_list.append(scene)
#         print(f"Using {len(self.scene_list)} scenes: {self.scene_list}")

#     def _load_infeasible_list(self, scene_name: str):
#         # set_trace()
#         infeasible_list_path = os.path.join(self.dataset_root, self.category, scene_name, "infeasible.txt")
#         if not os.path.exists(infeasible_list_path):
#             return []
#         with open(infeasible_list_path, "r") as f:
#             infeasible_str = f.readline()
#             infeasible_list = infeasible_str.split(",")
#         return infeasible_list

#     def _get_camera_mats(self, camera, h: int, w: int):
#         principal_point = camera.principal_point
#         focal_length = camera.focal_length
#         half_image_size_wh_orig = focal_length.new_tensor([[w / 2, h / 2]])
#         principal_point_px = -1.0 * (principal_point - 1.0) * half_image_size_wh_orig
#         focal_length_px = focal_length * half_image_size_wh_orig
#         fx, fy = focal_length_px.unbind(1)
#         px, py = principal_point_px.unbind(1)
#         K = fx.new_zeros(1, 4, 4)
#         K[:, 0, 0] = fx
#         K[:, 1, 1] = fy
#         K[:, 0, 2] = px
#         K[:, 1, 2] = py
#         K[:, 2, 2] = 1.0
#         K[:, 3, 3] = 1.0
#         # print(principal_point)
#         R = camera.R  # [1,3,3]
#         T = camera.T  # [1,3]
#         K_out = torch.cat((torch.cat((R, T.unsqueeze(2)), 2), T.new_zeros((1, 1, 4))), 1)
#         K_out[0, -1, -1] = 1
#         # print(K,"K")
#         # print(K_out,"K_OUT")
#         world_mat = torch.bmm(K, K_out).squeeze(0)
#         camera_mat = torch.eye(4)
#         scale_mat = torch.eye(4)
#         return world_mat, camera_mat, scale_mat, camera, principal_point, focal_length, R, T

#     def _load_data(self, scene_name: str):
#         if "content_list" in self.data[scene_name]:
#             return
#         if scene_name in self.scene_loaded:
#             return 
#         self.data[scene_name]["content_list"] = []
#         # for idx in tqdm(self.data[scene_name]["index_list"][:self.num_views], desc="Loading scene {}".format(scene_name)):
#         # for idx in self.data[scene_name]["index_list"][:self.num_views]:
#         for idx in random.sample(self.data[scene_name]["index_list"], self.num_views):
#             origin_data = self.datasets[idx]
#             processed_data = {}
#             processed_data["image"] = origin_data.image_rgb
#             # print(origin_data.image_path)
#             # print(origin_data.bbox_xywh)
#             processed_data["mask"] = origin_data.fg_probability > 0.5
#             if self.load_masked_image:
#                 # set_trace()
#                 # processed_data["masked_image"] = processed_data["image"][processed_data["mask"]]
#                 processed_data["image"][~processed_data["mask"].repeat(3, 1, 1)] = 0
#             # print(origin_data.sequence_point_cloud.get_bounding_boxes().shape)
#             # set_trace()
#             processed_data["depth_map"] = origin_data.depth_map#[1,200,200]
#             processed_data["depth_mask"] = origin_data.depth_mask#[1,200,200]
#             point_box = origin_data.sequence_point_cloud.get_bounding_boxes()
#             center = ((point_box[:,:,0] + point_box[:,:,1]) *0.5).squeeze(0)
#             radius = (point_box[:,:,1] - point_box[:,:,0]) * 0.5
#             radius = torch.sqrt((radius*radius).sum())
#             radius = radius*0.6
#             # image_hw = origin_data.image_size_hw
#             h, w = origin_data.image_rgb.shape[1:]
#             world_mat, camera_mat, scale_mat, camera, principal_point, focal_length, R, T = self._get_camera_mats(origin_data.camera, h, w)
#             # scale_mat[0,0] = radius
#             # scale_mat[1,1] = radius
#             # scale_mat[2,2] = radius
#             # scale_mat[:3,3] = center
#             # print('scale_mat', scale_mat)
#             processed_data["world_mat"] = world_mat
#             # print('world_mat',world_mat)
#             # set_trace()
#             processed_data["camera_mat"] = camera_mat
#             processed_data["scale_mat"] = scale_mat
#             processed_data["camera_ndc"] = camera
#             processed_data["principal_point"] = principal_point
#             processed_data["focal_length"] = focal_length
#             processed_data["R"] = R
#             processed_data["T"] = T
#             self.data[scene_name]["content_list"].append(processed_data)
#         self.scene_loaded.append(scene_name)
#         #print("{} scenes have been loaded".format(len(self.scene_loaded)))

#     def __getitem__(self, idx: int):
#         scene_id = idx // self.num_views
#         view_id = idx % self.num_views        
        
#         self._load_data(self.scene_list[scene_id])
#         content = self.data[self.scene_list[scene_id]]["content_list"]

#         all_imgs = torch.stack([content[v]["image"] for v in range(self.num_views)])
#         all_world_mats = torch.stack([content[v]["world_mat"] for v in range(self.num_views)])
#         all_camera_mats = torch.stack([content[v]["camera_mat"] for v in range(self.num_views)])
#         all_scale_mats = torch.stack([content[v]["scale_mat"] for v in range(self.num_views)])
#         all_masks = torch.stack([content[v]["mask"] for v in range(self.num_views)])
#         # all_camera_ndc = {content[v]["camera_ndc"] for v in range(self.num_views)}
#         # all_camera_ndc = [content[v]["camera_ndc"] for v in range(self.num_views)]
#         all_principal_point = torch.stack([content[v]["principal_point"] for v in range(self.num_views)])
#         all_focal_length = torch.stack([content[v]["focal_length"] for v in range(self.num_views)])
#         all_R = torch.stack([content[v]["R"] for v in range(self.num_views)])
#         all_T = torch.stack([content[v]["T"] for v in range(self.num_views)])
#         all_depth_map = torch.stack([content[v]["depth_map"] for v in range(self.num_views)])
#         all_depth_mask = torch.stack([content[v]["depth_mask"] for v in range(self.num_views)])
#         neighbor_list = self._random_sample_neighbor_list(view_id)

#         data = {
#             "view_idx": view_id,
#             "all_imgs": all_imgs, 
#             "all_world_mats": all_world_mats,
#             "all_camera_mats": all_camera_mats,
#             "all_scale_mats": all_scale_mats,
#             "all_masks": all_masks,
#             "all_principal_point": all_principal_point,
#             "all_focal_length": all_focal_length,
#             "all_R": all_R,
#             "all_T": all_T,
#             "neighbor_list": neighbor_list,
#             "all_depth_map": all_depth_map,
#             "all_depth_mask": all_depth_mask,
#         }

#         return data

#     # def _random_sample_neighbor_list(self, view_id: int):
#     #     neighbor_list = random.sample(range(self.num_views), self.num_sample_neighbor)
#     #     if view_id not in neighbor_list:
#     #         neighbor_list[0] = view_id
#     #     return torch.tensor(neighbor_list)

#     def _random_sample_neighbor_list(self, view_id: int):
#         if self.num_sample_neighbor <= self.num_views - 1:
#             neighbor_list = random.sample([i for i in range(self.num_views) if i != view_id], self.num_sample_neighbor)
#         else:
#             neighbor_list = [i for i in range(self.num_views) if i != view_id]
#         return torch.tensor(neighbor_list)

#     def __len__(self):
#         return len(self.scene_list) * self.num_views


