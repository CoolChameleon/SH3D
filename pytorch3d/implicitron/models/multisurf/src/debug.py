import sys, os
sys.path.append(os.path.join(os.getcwd(), "../"))

from dataloading.DTUloader import get_dataloader
from utils.configLoader import load_config

cfg = load_config("/home/shj20/unisurf/multisurf/configs/default.yaml")
dataloader = get_dataloader(cfg)

from model.baseModel import BaseModel

model = BaseModel(cfg)

for data in dataloader:
    model(data)

# dataloader = get_dataloader(cfg, device="cpu")

# iter_train = iter(dataloader)
# data = next(iter_train)

# for k in data.keys():
#     print(k, data[k].shape)