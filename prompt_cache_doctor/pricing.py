"""Anthropic model pricing (per million tokens), as of 2026.

Cache writes are billed at 1.25x the input rate; cache reads at 0.10x.
Override defaults by passing a custom pricing dict to `analyze()`.
"""

from typing import Dict

PRICING_PER_MTOK: Dict[str, Dict[str, float]] = {
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.00},
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def input_rate(model: str, pricing: Dict[str, Dict[str, float]]) -> float:
    """Best-effort lookup with prefix fallback (e.g. claude-sonnet-4-6-20251022)."""
    if model in pricing:
        return pricing[model]["input"]
    for known in pricing:
        if model.startswith(known):
            return pricing[known]["input"]
    return 3.00
