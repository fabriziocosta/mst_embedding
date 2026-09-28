"""Default differentiable losses for IMSTE."""

import torch


def log_attraction_loss(
    positive_squared_distances: torch.Tensor,
    edge_weights: torch.Tensor,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Mean rank-weighted log penalty on the configured positive distance."""
    return torch.mean(
        edge_weights * torch.log1p(positive_squared_distances + epsilon)
    )


def direct_attraction_loss(
    positive_distances: torch.Tensor,
    edge_weights: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Mean rank-weighted raw distance or squared distance for positive edges."""
    del epsilon
    return torch.mean(edge_weights * positive_distances)


def logistic_attraction_loss(
    positive_distances: torch.Tensor,
    edge_weights: torch.Tensor,
    epsilon: float,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Positive-pair logistic loss using the complement of the repulsion link."""
    scaled_margin = margin / temperature
    baseline = torch.nn.functional.softplus(
        positive_distances.new_tensor(-scaled_margin)
    )
    loss = torch.nn.functional.softplus(
        (positive_distances - margin) / temperature
    ) - baseline
    return torch.mean(edge_weights * loss)


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
    """Mean rank-weighted Huber penalty on the selected distance value."""
    del epsilon
    quadratic = 0.5 * positive_squared_distances.square()
    linear = delta * (positive_squared_distances - 0.5 * delta)
    penalty = torch.where(positive_squared_distances <= delta, quadratic, linear)
    return torch.mean(edge_weights * penalty)


def logistic_repulsion_loss(
    negative_distances: torch.Tensor,
    epsilon: float,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Negative-pair logistic loss, high below the margin and low above it."""
    del epsilon
    return torch.mean(
        torch.nn.functional.softplus((margin - negative_distances) / temperature)
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
