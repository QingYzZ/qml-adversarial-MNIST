import torch
import torch.nn as nn
import torchquantum as tq
import torch.nn.functional as F
import numpy as np

n_qubit = 4
q_depth = 3
nw_list_normal = list(range(2 ** n_qubit))

def generate_qubit_states_torch(n):
    return torch.tensor([list(map(int, f"{i:0{n}b}")) for i in range(2**n)], dtype=torch.float32)

class LewHybridNN(nn.Module):
    class QLayer(nn.Module):
        def __init__(self, n_blocks):
            super().__init__()
            self.n_wires = int(np.ceil(np.log2(len(nw_list_normal))))
            self.n_blocks = n_blocks
            self.u3_layers = tq.QuantumModuleList()
            self.cu3_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.u3_layers.append(tq.Op1QAllLayer(op=tq.U3, n_wires=self.n_wires, has_params=True, trainable=True))
                self.cu3_layers.append(tq.Op2QAllLayer(op=tq.CU3, n_wires=self.n_wires, has_params=True, trainable=True, circular=True))

        def forward(self):
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=next(self.parameters()).device)
            easy_scale_coeff = 2 ** (self.n_wires - 1)
            gamma, beta, alpha = 0.1, 0.8, 0.3
            for k in range(self.n_blocks):
                self.u3_layers[k](qdev)
                self.cu3_layers[k](qdev)
            state_mag = qdev.get_states_1d().abs()[0][:len(nw_list_normal)]
            x = torch.abs(state_mag) ** 2
            x = (beta * torch.tanh(gamma * easy_scale_coeff * x)) ** alpha
            x = x - torch.mean(x)
            return x

    class MappingModel(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size):
            super().__init__()
            self.input_layer = nn.Linear(input_size, hidden_sizes[0])
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
            self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        def forward(self, X):
            X = X.type_as(self.input_layer.weight)
            X = self.input_layer(X)
            for hidden in self.hidden_layers:
                X = hidden(X)
            output = self.output_layer(X)
            return output

    def __init__(self):
        super().__init__()
        self.MappingNetwork = self.MappingModel(n_qubit+1, [4, 20, 4], 1)
        self.QuantumNN = self.QLayer(q_depth)

        # CNN (fixed architecture, trainable parameters)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        device = x.device
        probs_ = self.QuantumNN()
        probs_ = probs_[:len(nw_list_normal)].to(device).reshape(-1, 1)

        qubit_states_torch = generate_qubit_states_torch(n_qubit)[:len(nw_list_normal)].to(device)
        combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=1).reshape(len(nw_list_normal), 1, n_qubit+1)

        modulator = self.MappingNetwork(combined_data_torch).mean().item()  # scalar modulator

        # Apply CNN with modulation (e.g., scaling conv1 output)
        x = F.relu(self.conv1(x)) * modulator
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
