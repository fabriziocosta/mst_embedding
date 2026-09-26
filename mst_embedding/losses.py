"""Default differentiable losses for IMSTE."""

import torch


def log_attraction_loss(
    positive_squared_distances: torch.Tensor,
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean rank-weighted ``log(1 + squared_distance)`` over graph edges."""
    return torch.mean(edge_weights * torch.log1p(positive_squared_distances))


def inverse_distance_repulsion_loss(
    negative_squared_distances: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Mean inverse-distance penalty over sampled graph non-edges."""
    return torch.mean(1.0 / (1.0 + negative_squared_distances + epsilon))
