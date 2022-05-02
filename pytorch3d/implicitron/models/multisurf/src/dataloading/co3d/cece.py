import os.path
from collections import defaultdict
from co3d.co3d_dataset import FrameData, Co3dDataset
from co3d.dataset_zoo import DATASET_ROOT, dataset_zoo
from co3d.dataloader_zoo import dataloader_zoo
import cv2
import math
import torch.utils.data.dataset
import numpy as np


class co3d_dataset(torch.utils.data.Dataset):
    def __init__(self, category="car", split="train"):
        datasets = dataset_zoo(
            category=category,
            assert_single_seq=False,
            dataset_name="co3d_multisequence",
            test_on_train=False,
            load_point_clouds=False, )
        self.datasets = datasets[split]

    def __len__(self):
        return self.datasets.__len__()

    def __getitem__(self, item):
        try:
            origin_data = self.datasets[item]
        except IOError:
            return None
        img = origin_data.image_rgb  # [3,h,w]
        img_mask = origin_data.fg_probability  # [1,h,w]
        depth_map = origin_data.depth_map  # [1,h,w]
        # depth_mask = origin_data.depth_mask  # [1,h,w]
        principal_point = origin_data.camera.principal_point
        focal_length = origin_data.camera.focal_length
        h, w = img.shape[1:]
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
        R = origin_data.camera.R  # [1,3,3]
        T = origin_data.camera.T  # [1,3]
        K_out = torch.cat((torch.cat((R, T.unsqueeze(2)), 2), T.new_zeros((1, 1, 4))), 1)
        K_out[0, -1, -1] = 1
        world_mat = torch.bmm(K, K_out).squeeze(0).cpu().numpy()
        name = origin_data.sequence_name
        return name, world_mat, img_mask.permute(1, 2, 0).cpu().numpy()


category = ["car", "banana"]
for c in category:
    datasets = co3d_dataset(category=c)
    root = os.path.join(DATASET_ROOT, c)
    result = defaultdict(list)
    mask = defaultdict(list)
    for num in range(len(datasets)):
        data = datasets[num]
        if data is not None:
            name, world_mat, mask_img = data
            result[name].append(world_mat)
            mask[name].append(mask_img)
            # break
        print(num * 100 / len(datasets))
    for k, v in result.items():
        save_path = os.path.join(root, k)
        img_path = os.path.join(save_path, 'resize_masks')
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        save_path = os.path.join(save_path, 'cameras.npz')
        res = {}
        for i, j in enumerate(v):
            res[f'world_mat_{i}'] = j
            cv2.imwrite(os.path.join(img_path, f'{i}.jpg'), mask[k][i] * 255)
        np.savez(save_path, **res)
        print(save_path)
