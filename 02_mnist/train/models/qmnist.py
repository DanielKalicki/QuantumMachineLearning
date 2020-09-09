import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pennylane as qml
import math
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device = "cpu"

config = {
    'batch_size': 1,
    'image_h': 3,
    'image_w': 3,
    'qLayers_num': 2,
    'class_num': 5,
    'name': 'circuit02_3x3_5class_2xLayersCnotRev_1000shots'
}

def prepare_qState(input_):
    input_ = torch.flatten(input_, start_dim=1).squeeze_(0)
    state = []
    for qubit in input_:
        state.append(float(qubit)*math.pi)
    state = Variable(torch.tensor(np.array(state).astype(np.float32), device=device), requires_grad=False)
    return state

n_ancila = 2
n_wires = 4
# dev = qml.device('default.qubit', wires=n_wires+n_ancila, shots=1000, analytic=False)
dev = qml.device('qulacs.simulator', wires=n_wires+n_ancila, shots=1000, analytic=False, gpu=False)

@qml.qnode(dev)
def qMnist_node(inputs, params):
    for qIdx in range(n_wires):
        qml.RY(inputs[qIdx], wires=qIdx)

    # qml.templates.StronglyEntanglingLayers(params, wires=range(n_wires))

    # qml.templates.subroutines.ArbitraryUnitary(params[:4**n_wires-1], wires=range(n_wires))
    # qml.templates.subroutines.ArbitraryUnitary(params[4**n_wires-1:], wires=range(n_wires))

    # layer 1
    startIdx = 0
    for qIdx in range(n_wires):
        qml.U3(params[qIdx*3 + startIdx], params[qIdx*3 + 1 + startIdx], params[qIdx*3 + 2 + startIdx], wires=qIdx)

    for qIdx in range(n_wires-1):
        qml.CNOT(wires=[qIdx, qIdx+1])
    for qIdx in reversed(range(1, n_wires)):
        qml.CNOT(wires=[qIdx, qIdx-1])
    # for qIdx in range(n_wires-2):
    #     qml.Toffoli(wires=[qIdx, qIdx+1, qIdx+2])
    # for qIdx in reversed(range(2, n_wires)):
    #     qml.Toffoli(wires=[qIdx, qIdx-1, qIdx-2])

    startIdx = n_wires*3
    for qIdx in range(n_wires):
        qml.U3(params[qIdx*3 + startIdx], params[qIdx*3 + 1 + startIdx], params[qIdx*3 + 2 + startIdx], wires=qIdx)

    # layer 2
    for qIdx in range(n_wires-1):
        qml.CNOT(wires=[qIdx, qIdx+1])
    for qIdx in reversed(range(1, n_wires)):
        qml.CNOT(wires=[qIdx, qIdx-1])
    # for qIdx in range(n_wires-2):
    #     qml.Toffoli(wires=[qIdx, qIdx+1, qIdx+2])
    # for qIdx in reversed(range(2, n_wires)):
    #     qml.Toffoli(wires=[qIdx, qIdx-1, qIdx-2])

    startIdx = 2*n_wires*3
    for qIdx in range(n_wires):
        qml.U3(params[qIdx*3 + startIdx], params[qIdx*3 + 1 + startIdx], params[qIdx*3 + 2 + startIdx], wires=qIdx)

    # # layer 3
    # for qIdx in range(n_wires-1):
    #     qml.CNOT(wires=[qIdx, qIdx+1])
    # for qIdx in reversed(range(1, n_wires)):
    #     qml.CNOT(wires=[qIdx, qIdx-1])

    # startIdx = 3*n_wires*3
    # for qIdx in range(n_wires):
    #     qml.U3(params[qIdx*3 + startIdx], params[qIdx*3 + 1 + startIdx], params[qIdx*3 + 2 + startIdx], wires=qIdx)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

@qml.qnode(dev)
def qMnist_node2(inputs, params):
    for qIdx in range(n_wires):
        qml.RY(inputs[qIdx], wires=qIdx)

    for qIdx in range(n_wires-1):
        qml.templates.subroutines.ArbitraryUnitary(weights=params[0, qIdx], wires=[qIdx, qIdx+1])

    for qIdx in range(n_wires-1):
        qml.templates.subroutines.ArbitraryUnitary(weights=params[1, qIdx], wires=[qIdx, qIdx+1])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

@qml.qnode(dev)
def qMnist_node3(inputs, params):
    for qIdx in range(n_wires):
        qml.RY(inputs[qIdx], wires=qIdx+n_ancila)

    for aqIdx in range(n_ancila):
        for qIdx in range(n_wires):
            qml.templates.subroutines.ArbitraryUnitary(weights=params[aqIdx, qIdx], wires=[aqIdx, qIdx+n_ancila])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_ancila+n_wires)]

@qml.qnode(dev)
def qMnist_node3_2l(inputs, params):
    for qIdx in range(n_wires):
        qml.RY(inputs[qIdx], wires=qIdx+n_ancila)

    for aqIdx in range(n_ancila):
        for qIdx in range(n_wires):
            qml.templates.subroutines.ArbitraryUnitary(weights=params[0, aqIdx, qIdx], wires=[aqIdx, qIdx+n_ancila])

    for aqIdx in range(n_ancila):
        for qIdx in range(n_wires):
            qml.templates.subroutines.ArbitraryUnitary(weights=params[1, aqIdx, qIdx], wires=[aqIdx, qIdx+n_ancila])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_ancila+n_wires)]

@qml.qnode(dev)
def qMnist_node3_3l(inputs, params, params2):
    for qIdx in range(n_wires):
        qml.RY(inputs[qIdx], wires=qIdx+n_ancila)

    for aqIdx in range(n_ancila):
        for qIdx in range(n_wires):
            qml.templates.subroutines.ArbitraryUnitary(weights=params[0, aqIdx, qIdx], wires=[aqIdx, qIdx+n_ancila])

    for qIdx in range(n_wires-1):
        qml.templates.subroutines.ArbitraryUnitary(weights=params2[qIdx, qIdx+1], wires=[qIdx+n_ancila, qIdx+n_ancila+1])

    for aqIdx in range(n_ancila):
        for qIdx in range(n_wires):
            qml.templates.subroutines.ArbitraryUnitary(weights=params[1, aqIdx, qIdx], wires=[aqIdx, qIdx+n_ancila])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_ancila+n_wires)]

@qml.qnode(dev)
def qMnist_node4(inputs, params):
    for qIdx in range(n_wires):
        qml.RY(inputs[qIdx], wires=qIdx+n_ancila)

    for aqIdx in range(n_ancila):
        startIdx = aqIdx*n_wires*3
        for qIdx in range(n_wires):
            qml.U3(params[qIdx*3 + startIdx], params[qIdx*3 + 1 + startIdx], params[qIdx*3 + 2 + startIdx], wires=qIdx+n_ancila)

        for qIdx in range(n_wires):
            qml.CNOT(wires=[qIdx+n_ancila, aqIdx])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_ancila+n_wires)]

class QMnist(nn.Module):
    def __init__(self, config):
        super(QMnist, self).__init__()
        self.config = config
        # params_shape = {'params': (config['qLayers_num']+1)*config['image_h']*config['image_w']*3}
        # # params_shape = {'params': 2*(4**n_wires-1)}
        # qmnist_layer = qml.qnn.TorchLayer(qMnist_node, params_shape)

        # params_shape = {'params': (2, n_wires-1, 15)}
        # qmnist_layer2 = qml.qnn.TorchLayer(qMnist_node2, params_shape)
        # self.qmodel = torch.nn.Sequential(qmnist_layer2)

        if config['quantum']['layers'] == 1:
            params_shape = {'params': (n_ancila, n_wires, 15)}
            qmnist_layer3 = qml.qnn.TorchLayer(qMnist_node3, params_shape)
            self.qmodel = torch.nn.Sequential(qmnist_layer3)
        elif config['quantum']['layers'] == 2:
            params_shape = {'params': (2, n_ancila, n_wires, 15)}
            qmnist_layer3 = qml.qnn.TorchLayer(qMnist_node3_2l, params_shape)
            self.qmodel = torch.nn.Sequential(qmnist_layer3)
        elif config['quantum']['layers'] == 3:
            params_shape = {'params': (2, n_ancila, n_wires, 15), 'params2': (n_wires, n_wires, 15)}
            qmnist_layer3 = qml.qnn.TorchLayer(qMnist_node3_3l, params_shape)
            self.qmodel = torch.nn.Sequential(qmnist_layer3)

        # params_shape = {'params': (n_ancila*n_wires*3)}
        # qmnist_layer4 = qml.qnn.TorchLayer(qMnist_node4, params_shape)
        # self.qmodel = torch.nn.Sequential(qmnist_layer4)

    def forward(self, inputs):
        return self.qmodel(inputs)

