import os
from tqdm import tqdm

dataset_root = "/media/disk2/shj_data/co3d/teddybear"
ratio_root = "/home/shj20/pytorch3d/pytorch3d/implicitron/models/multisurf/test/ratio_teddybear"

infeasible_name = "infeasible.txt"
ratio_name = "ratio.txt"

for sequence in tqdm(os.listdir(ratio_root)):
    ratio_path = os.path.join(ratio_root, sequence, ratio_name)
    if not os.path.exists(ratio_path):
        print(f"ratio file {ratio_path} not found")
        continue
    dataset_seq_path = os.path.join(dataset_root, sequence)
    if not os.path.exists(dataset_seq_path):
        print(f"dataset {dataset_seq_path} not found")
        continue
    with open(ratio_path, "r") as f:
        ratio_list = f.readlines()
    infeasible_list = []
    for ratio_str in ratio_list:
        image_idx, ratio = ratio_str.split()
        if eval(ratio) < 0.85:
            infeasible_list.append(image_idx)
    print(f"sequence {sequence}, infeasible_list {infeasible_list}")
    with open(os.path.join(dataset_seq_path, infeasible_name), "w") as f:
        f.write(",".join(infeasible_list))