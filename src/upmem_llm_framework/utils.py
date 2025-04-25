"""Utility functions for the UPMEM LLM framework."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TypeVar

import torch

T = TypeVar("T")


def add_dictionaries(dict1: dict[T, float], dict2: dict[T, float]) -> dict[T, float]:
    """Add two dictionaries together, summing the values of matching keys."""
    result = defaultdict(float, dict1)
    for key, value in dict2.items():
        result[key] += value
    return dict(result)


@dataclass
class LayerProfile:
    """Class to store profiling information for a layer."""

    id: int
    name: str
    n_layer: int
    context: str
    dim_in: int
    dim_out: int
    exec_time: float = 0
    exec_nums: int = 0
    energy: dict = field(default_factory=dict)
    obj: torch.nn.Module | None = None
