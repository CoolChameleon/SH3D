from dataloading.configloading import load_config
cfg = load_config("/home/shj20/pytorch3d/pytorch3d/implicitron/models/multisurf/configs/condconv/0411_plot_mask.yaml")

from dataloading.co3d.dataset_zoo import dataset_zoo
from dataloading.CO3Dloader import get_dataloader, CO3DDataset
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from ipdb import set_trace
import os

datasets = dataset_zoo(
    category="toybus",
    assert_single_seq=False,
    dataset_name="co3d_multisequence",
    test_on_train=True,
    load_point_clouds=True,
    test_restrict_sequence_id=-1
)

train_data = datasets["train"]

save_root = "../test/toy_bus_200_200"

for index in tqdm(range(0, len(train_data), 1)):
    points_cloud = train_data[index].sequence_point_cloud
    points = points_cloud.points_list()[0]

    camera = train_data[index].camera

    world_mat = camera.get_full_projection_transform().get_matrix()

    points_homo = torch.cat([points, torch.ones([points.shape[0], 1])], dim=1)
    camera_points = points_homo @ world_mat.squeeze(0)

    camera_points[:, 0] = camera_points[:, 0] / camera_points[:, 3]
    camera_points[:, 1] = camera_points[:, 1] / camera_points[:, 3]

    x = (-camera_points[:, 0].numpy() + 1) * 100
    y = (-camera_points[:, 1].numpy() + 1) * 100

    rgb = train_data[index].image_rgb

    plt.clf()
    
    plt.imshow(rgb.permute(1, 2, 0).numpy())
    plt.scatter(x, y, s=0.01)

    sequence_name = train_data[index].sequence_name
    save_dir = os.path.join(save_root, sequence_name)
    os.makedirs(save_dir, exist_ok=True)
    image_name = train_data[index].image_path.split("/")[-1]
    
    plt.savefig(os.path.join(save_dir, image_name))
    
    score_file_name = "quality_score.txt"
    if not os.path.exists(os.path.join(save_dir, score_file_name)):
        with open(os.path.join(save_dir, score_file_name), "w") as f:
            f.write(f"{train_data[index].camera_quality_score=}\n")
            f.write(f"{train_data[index].point_cloud_quality_score=}\n")


