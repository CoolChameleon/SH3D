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
    # category="toybus",
    category="teddybear",
    assert_single_seq=False,
    # dataset_name="co3d_multisequence",
    dataset_name="co3d_singlesequence",
    test_on_train=True,
    load_point_clouds=True,
    test_restrict_sequence_id=0
)

train_data = datasets["train"]

# save_root = "../test/toy_bus_200_200"
save_root = "../test/ratio_teddybear"

for index in tqdm(range(0, len(train_data), 1)):
    # set_trace()
    # print(train_data[i].sequence_name)
    points_cloud = train_data[index].sequence_point_cloud
    try:
        points = points_cloud.points_list()[0]
    except:
        continue

    camera = train_data[index].camera

    world_mat = camera.get_full_projection_transform().get_matrix()

    points_homo = torch.cat([points, torch.ones([points.shape[0], 1])], dim=1)
    camera_points = points_homo @ world_mat.squeeze(0)

    camera_points[:, 0] = camera_points[:, 0] / camera_points[:, 3]
    camera_points[:, 1] = camera_points[:, 1] / camera_points[:, 3]

    uv = ((-camera_points[:, :2] + 1) * 100).long()

    n_points_total = camera_points.shape[0]

    uv = uv[uv[:, 0] >= 0]
    uv = uv[uv[:, 0] < 200]
    uv = uv[uv[:, 1] >= 0]
    uv = uv[uv[:, 1] < 200]

    mask = (train_data[index].fg_probability > 0.5).squeeze(0)
    n_points_in_mask = (mask[uv[:, 1], uv[:, 0]]).sum()
    # n_points_in_mask = 0
    # for i in range(uv.shape[0]):
    #     if mask[uv[i, 0], uv[i, 1]]:
    #         n_points_in_mask += 1
    

    sequence_name = train_data[index].sequence_name
    save_dir = os.path.join(save_root, sequence_name)
    image_name = train_data[index].image_path.split("/")[-1]

    os.makedirs(save_dir, exist_ok=True)  

    ratio_file_name = "ratio.txt"
    with open(os.path.join(save_dir, ratio_file_name), "a+") as f:
        f.write(f"{image_name}\t{n_points_in_mask / n_points_total}\t\n")

    x = (-camera_points[:, 0].numpy() + 1) * 100
    y = (-camera_points[:, 1].numpy() + 1) * 100

    rgb = train_data[index].image_rgb

    plt.clf()
    
    # plt.imshow(mask.unsqueeze(-1).repeat(1, 1, 3).float().numpy(), cmap="gray")
    plt.imshow(rgb.permute(1, 2, 0).numpy())
    plt.scatter(x, y, s=0.01)

    # # sequence_name = train_data[index].sequence_name
    # # save_dir = os.path.join(save_root, sequence_name)
    # # os.makedirs(save_dir, exist_ok=True)
    # # image_name = train_data[index].image_path.split("/")[-1]
    
    plt.savefig(os.path.join(save_dir, image_name))
    
    # score_file_name = "quality_score.txt"
    # if not os.path.exists(os.path.join(save_dir, score_file_name)):
    #     with open(os.path.join(save_dir, score_file_name), "w") as f:
    #         f.write(f"{train_data[index].camera_quality_score=}\n")
    #         f.write(f"{train_data[index].point_cloud_quality_score=}\n")


