from dataloading.configloading import load_config
cfg = load_config("/home/shj20/pytorch3d/pytorch3d/implicitron/models/multisurf/configs/condconv/0411_plot_mask.yaml")

from dataloading.co3d.dataset_zoo import dataset_zoo
from dataloading.CO3Dloader import get_dataloader, CO3DDataset
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from ipdb import set_trace

datasets = dataset_zoo(
    category="toybus",
    assert_single_seq=False,
    dataset_name="co3d_singlesequence",
    test_on_train=True,
    load_point_clouds=True
)

train_data = datasets["train"]

co3d = CO3DDataset(cfg)

for index in tqdm(range(10)):
    points_cloud = train_data[index].sequence_point_cloud
    points = points_cloud.points_list()[0]

    world_mat, camera_mat, scale_mat, camera, principal_point, focal_length, R, T = co3d._get_camera_mats(train_data[index].camera, 200, 200)

    # K_out = torch.cat((torch.cat((R, T.unsqueeze(2)), 2), T.new_zeros((1, 1, 4))), 1)
    # set_trace()
    # temp_x = T[:,0].clone()
    # temp_y = T[:,1].clone()
    # temp_z = T[:,2]
    # T[:,0] = temp_y
    # T[:,1] = temp_x



    # K_out = torch.cat((torch.cat((R, T.unsqueeze(2)), 2), T.new_zeros((1, 1, 4))), 1)

    # K_out[0, -1, -1] = 1
    K_out = camera.get_world_to_view_transform().get_matrix()

    h, w = 200, 200
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
    # set_trace()

    K = camera.get_projection_transform().get_matrix()
    world_mat = torch.bmm(K, K_out).squeeze(0)
    world_mat = camera.get_full_projection_transform().get_matrix().squeeze(0)

    points_homo = torch.cat([points, torch.ones([points.shape[0], 1])], dim=1).T
    camera_points = world_mat @ points_homo
    # set_trace()
    # camera_points = K_out.squeeze(0) @ points_homo
    camera_points[0] = camera_points[0] / camera_points[2]
    camera_points[1] = camera_points[1] / camera_points[2]
    camera_points = camera_points[:2]


    x = camera_points[0].numpy()
    y = camera_points[1].numpy()
    rgb = train_data[index].image_rgb


    plt.clf()
    
    plt.imshow(rgb.permute(1, 2, 0).numpy())
    plt.scatter(x, y, s=0.01)
    plt.savefig(f"../test/0412/together_{index}.png")

