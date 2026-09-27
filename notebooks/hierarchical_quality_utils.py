"""Small reusable helpers for the hierarchical quality notebook."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def run_trials(
    configurations: Iterable[Mapping[str, Any]],
    run_one: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every configuration, collecting failures without aborting the sweep."""
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for configuration in configurations:
        try:
            results.append(run_one(configuration))
        except Exception as exc:
            failures.append(
                {
                    **configuration,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return results, failures
