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


def euclidean_attraction_loss(
    positive_squared_distances: torch.Tensor,
    edge_weights: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Mean rank-weighted, smoothed Euclidean distance for positive edges."""
    distance = torch.sqrt(positive_squared_distances + epsilon) - epsilon**0.5
    return torch.mean(edge_weights * distance)


def squared_distance_attraction_loss(
    positive_squared_distances: torch.Tensor,
    edge_weights: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Mean rank-weighted squared Euclidean distance for positive edges."""
    del epsilon
    return torch.mean(edge_weights * positive_squared_distances)


def huber_attraction_loss(
    positive_squared_distances: torch.Tensor,
    edge_weights: torch.Tensor,
    epsilon: float = 1e-4,
    delta: float = 1.0,
) -> torch.Tensor:
    """Mean rank-weighted Huber penalty on smoothed Euclidean distance."""
    distance = torch.sqrt(positive_squared_distances + epsilon) - epsilon**0.5
    quadratic = 0.5 * distance.square()
    linear = delta * (distance - 0.5 * delta)
    penalty = torch.where(distance <= delta, quadratic, linear)
    return torch.mean(edge_weights * penalty)


def bernoulli_repulsion_loss(
    negative_squared_distances: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Mean Bernoulli negative log-likelihood over sampled graph non-edges."""
    return torch.mean(torch.log1p(1.0 / (negative_squared_distances + epsilon)))


def inverse_distance_repulsion_loss(
    negative_squared_distances: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Mean inverse-distance penalty over sampled graph non-edges."""
    return torch.mean(1.0 / (1.0 + negative_squared_distances + epsilon))
