import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pennylane as qml
from batchers.func_approx_batcher import FuncApproxBatch
from torch.utils.tensorboard import SummaryWriter
import time

config = {
    'batch_size': 1,
    'cModel_in_dim': 2,
    'cModel_hidd_dim': 32,
    'qModel_params_num': 12,
    'name': 'circuit17'
}

writer = SummaryWriter(log_dir="./train/logs/"+config['name'])

dev = qml.device('default.qubit', wires=2, shots=10000, analytic=False)

def calc_loss(output, target):
    loss = 0
    for idx in range(output.shape[0]):
        loss += torch.abs(output[idx]-target[idx])**2
    return loss

@qml.qnode(dev, interface='torch')
def qModel(params):
    # qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    # qml.RZ(params[2], wires=0)

    # qml.RX(params[3], wires=1)
    qml.RY(params[4], wires=1)
    # qml.RZ(params[5], wires=1)

    # qml.CNOT(wires=[0, 1])

    # qml.RX(params[6], wires=0)
    qml.RY(params[7], wires=0)
    # qml.RZ(params[8], wires=0)

    # qml.RX(params[9], wires=1)
    qml.RY(params[10], wires=1)
    # qml.RZ(params[11], wires=1)

    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

class StatePrepCNetwork(nn.Module):
    def __init__(self, config):
        super(StatePrepCNetwork, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config['cModel_in_dim'], self.config['cModel_hidd_dim'])
        self.act1 = F.relu
        self.fc2 = nn.Linear(self.config['cModel_hidd_dim'], self.config['qModel_params_num'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def train(cModel, qModel, device, loader, optimizer, epoch):
    train_loss = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        qParams = cModel(data)[0]
        output = qModel(qParams)
        loss = calc_loss(output, target[0])
        loss.backward()
        optimizer.step()
        train_loss += loss.detach()
    end = time.time()
    print("")
    print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss/(batch_idx+1)))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    writer.add_scalar('loss/train', train_loss/(batch_idx+1), epoch)
    
cModel = StatePrepCNetwork(config)
opt = torch.optim.Adam(cModel.parameters(), lr = 0.01)
dataset_train = FuncApproxBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=0)

scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.1)
for epoch in range(0, 200):
    train(cModel, qModel, None, data_loader_train, opt, epoch)
    scheduler.step()
