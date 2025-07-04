import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout=0.2, use_residual=True):
        """
        MLP Decoder with optional residual connection.

        Args:
            input_dim (int): Input feature size (from encoder/GNN).
            output_dim (int): Output dimension (number of genes).
            hidden_dims (list): Hidden layer sizes.
            dropout (float): Dropout probability.
            use_residual (bool): Whether to add residual skip connection from input.
        """
        super(MLPDecoder, self).__init__()
        self.use_residual = use_residual and (input_dim == output_dim)

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.decoder = nn.Sequential(*layers)
        self.final_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        out = self.decoder(x)
        out = self.final_layer(out)

        if self.use_residual:
            out = out + x  # Add residual (only valid when input_dim == output_dim)
        return out

