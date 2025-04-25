"""Utility functions for the UPMEM LLM framework."""

from collections import defaultdict
from typing import TypeVar

T = TypeVar("T")


def add_dictionaries(dict1: dict[T, float], dict2: dict[T, float]) -> dict[T, float]:
    """Add two dictionaries together, summing the values of matching keys."""
    result = defaultdict(float, dict1)
    for key, value in dict2.items():
        result[key] += value
    return dict(result)
