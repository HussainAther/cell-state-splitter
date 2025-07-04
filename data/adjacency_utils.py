import torch

def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric normalized adjacency matrix for GCN.
    Formula: D^(-1/2) * (A + I) * D^(-1/2)

    Args:
        adj (torch.Tensor): Adjacency matrix (num_nodes x num_nodes), unnormalized.

    Returns:
        torch.Tensor: Normalized adjacency matrix.
    """
    num_nodes = adj.size(0)
    adj_with_self_loops = adj + torch.eye(num_nodes, device=adj.device)
    
    degree = adj_with_self_loops.sum(dim=1)
    deg_inv_sqrt = torch.pow(degree, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    return D_inv_sqrt @ adj_with_self_loops @ D_inv_sqrt

