import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import os
import codecs
import matplotlib.pyplot as plt

class NonLinearBatch(Dataset):
    def __init__(self, config, valid=False):
        self.config = config

    def func(self, x):
        # x = torch.stack([torch.abs(x[0]), x[1]**2, F.relu(x[2])], dim=0)
        x = F.relu(torch.abs(x)-0.5)*2
        return x

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        data = torch.rand((self.config['dim'],), dtype=torch.float)*2-1
        target = torch.rand((self.config['dim'],), dtype=torch.float)

        target = self.func(data)
        return data, target
