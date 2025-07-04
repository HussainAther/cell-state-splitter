# decoders.py

import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout=0.2):
        """
        MLP Decoder to transform encoded cell state into predicted gene expression.

        Args:
            input_dim (int): Dimensionality of input features (from GNN or encoder).
            output_dim (int): Number of genes (output vector size).
            hidden_dims (list of int): Sizes of hidden layers.
            dropout (float): Dropout probability.
        """
        super(MLPDecoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))  # Final prediction layer
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

