"""Default differentiable losses for IMSTE."""

import torch


def log_attraction_loss(
    positive_squared_distances: torch.Tensor,
    edge_weights: torch.Tensor,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Mean rank-weighted log penalty on squared positive distances."""
    return torch.mean(
        edge_weights * torch.log1p(positive_squared_distances + epsilon)
    )


def logistic_repulsion_loss(
    negative_squared_distances: torch.Tensor,
    epsilon: float,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Negative-pair logistic loss, high below the margin and low above it."""
    del epsilon
    return torch.mean(
        torch.nn.functional.softplus(
            (margin - negative_squared_distances) / temperature
        )
    )


def bernoulli_repulsion_loss(
    negative_squared_distances: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Mean Bernoulli loss over sampled graph non-edges."""
    return torch.mean(torch.log1p(1.0 / (negative_squared_distances + epsilon)))


def inverse_distance_repulsion_loss(
    negative_squared_distances: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Mean inverse-distance penalty over sampled graph non-edges."""
    return torch.mean(1.0 / (1.0 + negative_squared_distances + epsilon))
