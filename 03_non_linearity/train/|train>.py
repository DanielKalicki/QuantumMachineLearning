import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pennylane as qml
from batchers.nonlinear_batcher import NonLinearBatch
from models.cnonlinear import CNonLinear
from models.qnonlinear import QNonLinear, prepare_qState
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import math
import numpy as np
import sys

config = {
    'batch_size': 1,
    'dim': 3,
    'q_nwires': 5,
    # 'name': 'q_6w_mixV3_3dim_10kshots_relu((AbsX-.5)*2)'
    'name': 'q_5w_1xArbUnit_3dim_10kshots_relu((AbsX-.5)*2)'
    # 'name': 'c_3dim_16hdim_relu((AbsX-.5)*2)'
}

config['name'] += str(sys.argv[1])

writer = SummaryWriter(log_dir="./train/logs/"+config['name'])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device = "cpu"


def train(cModel, qModel, device, loader, optimizer, epoch):
    loss_ = []
    start = time.time()
    pbar = tqdm(total=len(loader), dynamic_ncols=True)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        ancila = torch.zeros(1, config['q_nwires']-1).to(device)
        # data = torch.cat([data, ancila], dim=1)
        data = torch.cat([data[:, 0].unsqueeze(1), 
                          ancila[:, 0].unsqueeze(1), 
                          data[:, 1].unsqueeze(1), 
                          ancila[:, 1].unsqueeze(1), 
                          data[:, 2].unsqueeze(1) 
                        #   ancila[:, 2].unsqueeze(1)
                         ], dim=1)
        # data = torch.cat([data, data, data], dim=1)
        optimizer.zero_grad()
        if cModel is not None:
            output = cModel(data)
        else:
            qState = prepare_qState(data)
            output = qModel(qState) # [:config['dim']]
            output = torch.stack((output[0], output[2], output[4]), dim=0)
            output = output.unsqueeze(0)
        loss = F.mse_loss(output, target, reduction='mean')
        loss.backward()
        optimizer.step()

        loss_.append(loss.detach())
        if batch_idx % 10 == 0:
            end = time.time()
            print('target:\t'+str(target))
            print('output:\t'+str(output))
            print("")
            loss_avr = sum(loss_)/float(len(loss_))
            print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(batch_idx, loss_avr))
            print('\t\tTraining time: {:.2f}'.format((end - start)))
            writer.add_scalar('loss/train', sum(loss_)/float(len(loss_)), batch_idx)
            writer.flush()
            loss_ = []
            start = time.time()

        if (batch_idx == 6000) and (cModel is not None):
            exit(1)
    
cModel = None
qModel = None

# cModel = CNonLinear(config)
# cModel.to(device)
qModel = QNonLinear(config)
qModel.to(device)

opt = None
if cModel is not None:
    opt = torch.optim.Adam(cModel.parameters(), lr=0.01)
else:
    opt = torch.optim.Adam(qModel.parameters(), lr=0.01)
dataset_train = NonLinearBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=0)

for epoch in range(0, 200):
    train(cModel, qModel, device, data_loader_train, opt, epoch)
