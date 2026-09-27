"""Default differentiable losses for IMSTE."""

import torch


def log_attraction_loss(
    positive_squared_distances: torch.Tensor,
    edge_weights: torch.Tensor,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Mean rank-weighted Bernoulli negative log-likelihood for graph edges."""
    return torch.mean(
        edge_weights * torch.log1p(positive_squared_distances + epsilon)
    )


def bernoulli_repulsion_loss(
    negative_squared_distances: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Mean Bernoulli negative log-likelihood over sampled graph non-edges."""
    return torch.mean(torch.log1p(1.0 / (negative_squared_distances + epsilon)))
