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

