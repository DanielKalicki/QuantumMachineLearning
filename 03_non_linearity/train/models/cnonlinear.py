import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNonLinear(nn.Module):
    def __init__(self, config):
        super(CNonLinear, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config['dim'], 16)
        self.act1 = F.relu
        self.fc2 = nn.Linear(16, self.config['dim'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x