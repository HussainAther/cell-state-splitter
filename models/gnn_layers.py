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

