import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pennylane as qml
from batchers.mnist_batcher import MnistBatch
from models.cmnist import CMnist
from models.qmnist import QMnist, prepare_qState
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import math
import numpy as np
from configs import configs
import sys

print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]

writer = SummaryWriter(log_dir="./train/logs/"+config['name'])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device = "cpu"


def train(cModel, qModel, device, loader, optimizer, epoch):
    acc_ = []
    loss_ = []
    start = time.time()
    pbar = tqdm(total=len(loader), dynamic_ncols=True)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # one_hot_target = (target == torch.arange(config['class_num']).reshape(1, config['class_num'])).float()
        # one_hot_target = one_hot_target*2-1 # set to [-1, 1]
        if cModel is not None:
            output = cModel(data)
            loss = F.cross_entropy(output, target, reduction='mean')
        else:
            qState = prepare_qState(data)
            output = qModel(qState)[:len(config['classes'])]
            output = output.unsqueeze(0)
            loss = F.cross_entropy(output, target, reduction='mean')
        # loss = F.mse_loss(output, one_hot_target, reduction='mean')
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total = target.size(0)
        correct = (predicted == target).sum().item()
        pbar.update(1)

        acc_.append(correct/float(total))
        loss_.append(loss.detach())
        if batch_idx % 10 == 0:
            end = time.time()
            print(target)
            # print(one_hot_target)
            print(output)
            print("")
            loss_avr = sum(loss_)/float(len(loss_))
            acc_avr = sum(acc_)/float(len(acc_))
            print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(batch_idx, loss_avr))
            print('\t\tAccuracy: {:.2f}%'.format(acc_avr*100))
            print('\t\tTraining time: {:.2f}'.format((end - start)))
            writer.add_scalar('loss/train', sum(loss_)/float(len(loss_)), batch_idx)
            writer.add_scalar('acc/train', sum(acc_)/float(len(acc_)), batch_idx)
            writer.flush()
            acc_ = []
            loss_ = []
            start = time.time()

        if batch_idx == 3000:
            exit(1)
    
cModel = None
qModel = None
if config['network_type'] == 'classic':
    cModel = CMnist(config)
    cModel.to(device)
else:
    qModel = QMnist(config)
    qModel.to(device)
opt = None
if cModel is not None:
    opt = torch.optim.Adam(cModel.parameters(), lr=0.01)
else:
    opt = torch.optim.Adam(qModel.parameters(), lr=0.01)
dataset_train = MnistBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=0)

for epoch in range(0, 200):
    train(cModel, qModel, device, data_loader_train, opt, epoch)
