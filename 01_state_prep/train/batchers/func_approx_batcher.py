import numpy as np
import torch
from torch.utils.data import Dataset

class FuncApproxBatch(Dataset):
    def __init__(self, config, valid=False):
        self.config = config
        pass

    def func(self, x):
        return x

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        data = torch.rand((self.config['cModel_in_dim'],), dtype=torch.float)
        target = torch.rand((self.config['cModel_in_dim'],), dtype=torch.float)

        target = self.func(data)
        return data, target
