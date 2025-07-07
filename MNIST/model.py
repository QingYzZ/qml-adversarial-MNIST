# MNIST/model.py

import torch
import torch.nn as nn
import torchquantum as tq
import torch.nn.functional as F

class CNOTRing(nn.Module):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires

    def forward(self, qdev):
        for i in range(self.n_wires):
            tq.CNOT()(qdev, wires=[i, (i + 1) % self.n_wires])
class CRZRing(nn.Module):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires

    def forward(self, qdev):
        for i in range(self.n_wires):
            tq.CRZ(has_params=True, trainable=True)(
                qdev, wires=[i, (i + 1) % self.n_wires]
            )

class CZRing(nn.Module):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires

    def forward(self, qdev):
        for i in range(self.n_wires):
            tq.CZ()(qdev, wires=[i, (i + 2) % self.n_wires])


import torch
import torch.nn as nn
import torchquantum as tq

class SimpleQNN(nn.Module):
    def __init__(self, n_wires=16, n_classes=10):
        super().__init__()
        self.n_wires = n_wires

        # Trainable layers
        self.ry_layer = tq.Op1QAllLayer(n_wires=n_wires, op=tq.RY, has_params=True, trainable=True)
        self.rx_layer = tq.Op1QAllLayer(n_wires=n_wires, op=tq.RX, has_params=True, trainable=True)
        self.rz_layer = tq.Op1QAllLayer(n_wires=n_wires, op=tq.RZ, has_params=True, trainable=True)

        # Entangling layers
        self.entangle_crz = CRZRing(n_wires=n_wires)
        self.entangle_cz = CZRing(n_wires=n_wires)
        self.entangle_cnot = CNOTRing(n_wires=n_wires)

        # Measurement & Readout
        self.measure = tq.MeasureAll(obs=tq.PauliZ)
        self.readout = nn.Linear(n_wires, n_classes)

    def forward(self, x):
        outputs = []

        for i in range(x.shape[0]):
            qdev = tq.QuantumDevice(n_wires=self.n_wires)

            # Input encoding: non-trainable RY
            for wire in range(self.n_wires):
                tq.RY(has_params=False)(qdev, wires=wire, params=x[i, wire].unsqueeze(0))

            # Add Hadamard gates to promote superposition
            for wire in range(self.n_wires):
                tq.Hadamard()(qdev, wires=wire)

            # Trainable layers
            self.ry_layer(qdev)
            self.rx_layer(qdev)

            # First round of entanglement
            self.entangle_crz(qdev)
            self.entangle_cz(qdev)

            # Final phase manipulation and CNOT ring
            self.rz_layer(qdev)
            self.entangle_cnot(qdev)

            # Measurement
            q_out = self.measure(qdev)
            outputs.append(q_out)

        q_outs = torch.cat(outputs, dim=0)
        q_outs = q_outs.to(self.readout.weight.device)
        return self.readout(q_outs)

