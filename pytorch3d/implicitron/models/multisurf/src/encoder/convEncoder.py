from ipdb import set_trace
import torch
import numpy as np
import cv2
import re
from torchvision.models import resnet18
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self, output_dim=128, pretrained=False):
        super().__init__()
        self.model = resnet18(pretrained=pretrained).cuda()
        if pretrained:
            for p in self.parameters():
                p.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, output_dim)

    def forward(self, x):
        x = self.model(x)        
        return x