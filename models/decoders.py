# decoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout=0.2):
        """
        MLP Decoder that outputs delta expression changes to be added to baseline.

        Args:
            input_dim (int): Dimension of encoded input (from encoder or GNN).
            output_dim (int): Number of genes (output dimension).
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

        self.decoder = nn.Sequential(*layers)
        self.final_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, encoded_input, baseline_expression):
        """
        Predict the perturbed expression by applying delta to baseline.

        Args:
            encoded_input (Tensor): Latent features from encoder/GNN [batch_size, input_dim].
            baseline_expression (Tensor): Pre-perturbation expression profile [batch_size, output_dim].

        Returns:
            Tensor: Predicted perturbed expression [batch_size, output_dim].
        """
        delta = self.final_layer(self.decoder(encoded_input))
        return baseline_expression + delta

