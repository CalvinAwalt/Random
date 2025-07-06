import torch
import torch.nn as nn

class QuantumNet(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=8):
        super(QuantumNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.quantum_layer = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.quantum_layer(x))  # Simulated quantum layer
        return self.fc2(x)