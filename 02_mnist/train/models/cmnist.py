import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CMnist(nn.Module):
    def __init__(self, config):
        super(CMnist, self).__init__()
        self.config = config
        self.h = self.config['image_h']
        self.w = self.config['image_w']
        self.out_dim = len(self.config['classes'])
        self.linear = self.config['classic']['linear']
        self.fc1 = nn.Linear(self.h*self.w, 16)
        self.act1 = F.relu
        self.fc2 = nn.Linear(16, self.out_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        if not self.linear:
            x = self.act1(x)
        x = self.fc2(x)
        return x
