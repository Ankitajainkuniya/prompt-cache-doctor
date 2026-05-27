"""prompt-cache-doctor — diagnose Anthropic prompt cache hit rate."""

from prompt_cache_doctor.analyzer import (
    CallRecord,
    RouteStats,
    AnalysisReport,
    analyze,
)
from prompt_cache_doctor.recorder import record

__version__ = "0.1.0"
__all__ = [
    "CallRecord",
    "RouteStats",
    "AnalysisReport",
    "analyze",
    "record",
    "__version__",
]
