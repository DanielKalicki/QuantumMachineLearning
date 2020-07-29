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

def prepare_qState(input_):
    input_ = torch.flatten(input_, start_dim=1).squeeze_(0)
    state = []
    for qubit in input_:
        state.append(float(qubit)*math.pi)
    state = Variable(torch.tensor(np.array(state).astype(np.float32), device=device), requires_grad=False)
    return state

n_wires = 5
dev = qml.device('default.qubit', wires=n_wires, shots=10000, analytic=False)

@qml.qnode(dev)
def qNonLin_node(inputs, params):
    for qIdx in range(n_wires):
        qml.RY(inputs[qIdx], wires=qIdx)

    # layer 1
    startIdx = 0
    for qIdx in range(n_wires):
        qml.U3(params[qIdx*3 + startIdx], params[qIdx*3 + 1 + startIdx], params[qIdx*3 + 2 + startIdx], wires=qIdx)

    # for qIdx in range(n_wires-1):
    #     qml.CNOT(wires=[qIdx, qIdx+1])
    # for qIdx in reversed(range(1, n_wires)):
    #     qml.CNOT(wires=[qIdx, qIdx-1])
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[4,5])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[5,4])

    startIdx = n_wires*3
    for qIdx in range(n_wires):
        qml.U3(params[qIdx*3 + startIdx], params[qIdx*3 + 1 + startIdx], params[qIdx*3 + 2 + startIdx], wires=qIdx)

    # layer 2
    # for qIdx in range(n_wires-1):
    #     qml.CNOT(wires=[qIdx, qIdx+1])
    # for qIdx in reversed(range(1, n_wires)):
    #     qml.CNOT(wires=[qIdx, qIdx-1])
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[4,5])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[5,4])

    startIdx = 2*n_wires*3
    for qIdx in range(n_wires):
        qml.U3(params[qIdx*3 + startIdx], params[qIdx*3 + 1 + startIdx], params[qIdx*3 + 2 + startIdx], wires=qIdx)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

@qml.qnode(dev)
def qNonLin_node2(inputs, params):
    for qIdx in range(n_wires):
        qml.RY(inputs[qIdx], wires=qIdx)

    for qIdx in range(n_wires-1):
        qml.templates.subroutines.ArbitraryUnitary(weights=params[0, qIdx], wires=[qIdx, qIdx+1])
    # for qIdx in range(n_wires-1):
    #     qml.templates.subroutines.ArbitraryUnitary(weights=params[1, qIdx], wires=[qIdx, qIdx+1])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

class QNonLinear(nn.Module):
    def __init__(self, config):
        super(QNonLinear, self).__init__()
        self.config = config
        # params_shape = {'params': (2+1)*n_wires*3}
        # qlayer = qml.qnn.TorchLayer(qNonLin_node, params_shape)

        params_shape = {'params': (1, n_wires-1, 15)}
        qlayer = qml.qnn.TorchLayer(qNonLin_node2, params_shape)

        self.qmodel = torch.nn.Sequential(qlayer)

    def forward(self, inputs):
        return self.qmodel(inputs)

