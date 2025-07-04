# encoders.py
import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.2):
        super(MLPEncoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# gnn_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        support = self.linear(x)
        out = torch.matmul(adjacency, support)
        return F.relu(out)


class GeneGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(GeneGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))

    def forward(self, x, adjacency):
        for layer in self.layers:
            x = layer(x, adjacency)
        return x


# decoders.py
import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout=0.2):
        super(MLPDecoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

