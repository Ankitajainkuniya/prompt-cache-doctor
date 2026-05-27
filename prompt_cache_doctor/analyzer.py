"""Core analyzer — turns a stream of CallRecords into an AnalysisReport.

The analyzer is intentionally simple: aggregate per-route, detect a handful of
known cache-miss patterns, and quantify their cost. It's not magic — it's a
checklist of mistakes that real Anthropic API users make, applied at scale.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from prompt_cache_doctor.pricing import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICING_PER_MTOK,
    input_rate,
)


# ─────────────────────────── data model ───────────────────────────

@dataclass
class CallRecord:
    """One Anthropic API call: the request shape + the usage numbers."""

    timestamp: Optional[datetime]
    route: str
    model: str
    system: str
    tool_names: Tuple[str, ...]
    first_user_msg_hash: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int

    @classmethod
    def from_log_line(cls, line: Dict[str, Any]) -> "CallRecord":
        req = line.get("request", {}) or {}
        usage = line.get("usage", {}) or {}

        ts_raw = line.get("timestamp")
        ts: Optional[datetime] = None
        if ts_raw:
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = None

        system = req.get("system") or ""
        if isinstance(system, list):
            system = "\n".join(
                (b.get("text", "") if isinstance(b, dict) else str(b)) for b in system
            )

        tools = req.get("tools") or []
        tool_names = tuple(
            (t.get("name") if isinstance(t, dict) else str(t)) for t in tools
        )

        messages = req.get("messages") or []
        first_user = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, list):
                    first_user = "\n".join(
                        (b.get("text", "") if isinstance(b, dict) else str(b))
                        for b in content
                    )
                else:
                    first_user = str(content or "")
                break

        return cls(
            timestamp=ts,
            route=line.get("route", "default"),
            model=req.get("model", "claude-sonnet-4-6"),
            system=str(system),
            tool_names=tool_names,
            first_user_msg_hash=_short_hash(first_user),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        )


@dataclass
class MissReason:
    label: str
    explanation: str
    fix: str
    est_savings_usd_per_month: float


@dataclass
class RouteStats:
    route: str
    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    paid_usd: float = 0.0
    no_cache_usd: float = 0.0
    miss_reasons: List[MissReason] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        total_input = self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens
        if total_input == 0:
            return 0.0
        return self.cache_read_tokens / total_input

    @property
    def savings_so_far(self) -> float:
        return max(self.no_cache_usd - self.paid_usd, 0.0)

    @property
    def max_possible_savings(self) -> float:
        # If hit rate were 100% the entire input would cost 0.10x base.
        return self.no_cache_usd * (1.0 - CACHE_READ_MULTIPLIER)


@dataclass
class AnalysisReport:
    total_calls: int = 0
    routes: List[RouteStats] = field(default_factory=list)

    @property
    def overall_hit_rate(self) -> float:
        total_input = sum(
            r.input_tokens + r.cache_creation_tokens + r.cache_read_tokens for r in self.routes
        )
        if total_input == 0:
            return 0.0
        return sum(r.cache_read_tokens for r in self.routes) / total_input

    @property
    def total_paid_usd(self) -> float:
        return sum(r.paid_usd for r in self.routes)

    @property
    def total_no_cache_usd(self) -> float:
        return sum(r.no_cache_usd for r in self.routes)

    @property
    def total_savings_so_far(self) -> float:
        return max(self.total_no_cache_usd - self.total_paid_usd, 0.0)

    @property
    def total_max_possible_savings(self) -> float:
        return self.total_no_cache_usd * (1.0 - CACHE_READ_MULTIPLIER)

    @property
    def total_potential_remaining(self) -> float:
        return max(self.total_max_possible_savings - self.total_savings_so_far, 0.0)


# ─────────────────────────── helpers ───────────────────────────

def _short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ─────────────────────────── miss detectors ───────────────────────────

def _detect_dynamic_system_prefix(calls: List[CallRecord]) -> Optional[MissReason]:
    """Flag the case where system prompts differ in the first few chars."""
    systems = [c.system for c in calls if c.system]
    if len(systems) < 2:
        return None

    # Pairwise: look at consecutive systems and how soon they diverge.
    early_divergences = 0
    short_divergences: List[int] = []
    base_len = max(len(systems[0]), 1)
    for i in range(1, len(systems)):
        cpl = _common_prefix_len(systems[i - 1], systems[i])
        if cpl < 200 and base_len > 500:
            early_divergences += 1
            short_divergences.append(cpl)

    if early_divergences >= max(2, len(systems) // 5):
        avg_div = int(sum(short_divergences) / max(len(short_divergences), 1))
        return MissReason(
            label="System prompt prefix changes per call",
            explanation=(
                f"Consecutive system prompts diverge after only ~{avg_div} characters. "
                "Anthropic caches by exact prefix — anything dynamic at the top of "
                "your system prompt invalidates the entire cache."
            ),
            fix="Move dynamic content (timestamps, user IDs, session state) into the LAST user message, after your cache_control breakpoint.",
            est_savings_usd_per_month=0.0,
        )
    return None


def _detect_unsorted_tools(calls: List[CallRecord]) -> Optional[MissReason]:
    """Flag if tool definitions are present but in different orders across calls."""
    tool_calls = [c for c in calls if c.tool_names]
    if len(tool_calls) < 2:
        return None

    sorted_fingerprints = {tuple(sorted(c.tool_names)) for c in tool_calls}
    raw_fingerprints = {c.tool_names for c in tool_calls}

    if len(sorted_fingerprints) == 1 and len(raw_fingerprints) > 1:
        return MissReason(
            label="Tool definitions in different order between calls",
            explanation=(
                "The tool list is logically identical but the order differs across calls. "
                "Anthropic's cache key is order-sensitive."
            ),
            fix="Sort your tools alphabetically (or by any stable key) before passing to client.messages.create().",
            est_savings_usd_per_month=0.0,
        )
    return None


def _detect_tiny_cached_payload(calls: List[CallRecord]) -> Optional[MissReason]:
    """Flag when cache writes happen but the cached payload is tiny."""
    writes = [c.cache_creation_input_tokens for c in calls if c.cache_creation_input_tokens > 0]
    if not writes:
        return None
    avg_write = sum(writes) / len(writes)
    if avg_write < 1024 and len(writes) >= 3:
        return MissReason(
            label="Cache breakpoint captures very little",
            explanation=(
                f"Average cached payload is only ~{int(avg_write)} tokens. The cache write "
                "premium (1.25x) only pays off if the cached chunk is large and reused."
            ),
            fix="Place cache_control deeper into your prompt — typically on the last system block or after your largest static context (RAG docs, schemas, examples).",
            est_savings_usd_per_month=0.0,
        )
    return None


def _detect_no_cache_at_all(calls: List[CallRecord]) -> Optional[MissReason]:
    has_any_cache = any(
        (c.cache_creation_input_tokens > 0 or c.cache_read_input_tokens > 0) for c in calls
    )
    if has_any_cache:
        return None
    big_calls = [c for c in calls if c.input_tokens >= 2048]
    if len(big_calls) >= 3:
        return MissReason(
            label="No cache_control set on any call",
            explanation=(
                "This route has not used prompt caching at all but is sending "
                f">2k input tokens on at least {len(big_calls)} calls — prime caching territory."
            ),
            fix="Add cache_control={'type': 'ephemeral'} to your largest static block (system prompt, tools, or RAG context).",
            est_savings_usd_per_month=0.0,
        )
    return None


def _detect_ttl_expiry(calls: List[CallRecord]) -> Optional[MissReason]:
    """Flag when consecutive cache writes happen >5min apart with no hits between."""
    timed = [c for c in calls if c.timestamp is not None]
    if len(timed) < 3:
        return None

    writes_only = [c for c in timed if c.cache_creation_input_tokens > 0 and c.cache_read_input_tokens == 0]
    if len(writes_only) < 3:
        return None

    gaps_over_5min = 0
    for i in range(1, len(writes_only)):
        delta = (writes_only[i].timestamp - writes_only[i - 1].timestamp).total_seconds()
        if 300 < delta < 3600:
            gaps_over_5min += 1

    if gaps_over_5min >= 2:
        return MissReason(
            label="Repeated cache writes >5 min apart",
            explanation=(
                "You're paying the cache-write premium repeatedly because the 5-minute default TTL "
                f"keeps expiring between calls ({gaps_over_5min} instances detected)."
            ),
            fix="Either batch calls closer together, or opt into the 1-hour TTL with cache_control={'type': 'ephemeral', 'ttl': '1h'}.",
            est_savings_usd_per_month=0.0,
        )
    return None


MISS_DETECTORS = [
    _detect_no_cache_at_all,
    _detect_dynamic_system_prefix,
    _detect_unsorted_tools,
    _detect_tiny_cached_payload,
    _detect_ttl_expiry,
]


# ─────────────────────────── main entry point ───────────────────────────

def analyze(
    records: Iterable[CallRecord],
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> AnalysisReport:
    """Aggregate records into a report with per-route stats + miss reasons."""
    pricing = pricing or PRICING_PER_MTOK
    by_route: Dict[str, List[CallRecord]] = {}
    for r in records:
        by_route.setdefault(r.route, []).append(r)

    report = AnalysisReport(total_calls=sum(len(v) for v in by_route.values()))

    for route, calls in by_route.items():
        stats = RouteStats(route=route, call_count=len(calls))
        for c in calls:
            stats.input_tokens += c.input_tokens
            stats.output_tokens += c.output_tokens
            stats.cache_creation_tokens += c.cache_creation_input_tokens
            stats.cache_read_tokens += c.cache_read_input_tokens

            rate = input_rate(c.model, pricing)
            # Actual cost (per Mtok): full input + output + cache write premium + cache read discount.
            paid = (
                (c.input_tokens * rate)
                + (c.cache_creation_input_tokens * rate * CACHE_WRITE_MULTIPLIER)
                + (c.cache_read_input_tokens * rate * CACHE_READ_MULTIPLIER)
            ) / 1_000_000
            # Hypothetical cost without caching: all cached tokens billed at full input rate.
            no_cache = (
                (c.input_tokens * rate)
                + (c.cache_creation_input_tokens * rate)
                + (c.cache_read_input_tokens * rate)
            ) / 1_000_000
            # Output portion (unaffected by caching, but we include in both totals so $$ is realistic).
            output_rate = pricing.get(c.model, {}).get("output", rate * 5)
            output_cost = (c.output_tokens * output_rate) / 1_000_000
            stats.paid_usd += paid + output_cost
            stats.no_cache_usd += no_cache + output_cost

        for detector in MISS_DETECTORS:
            reason = detector(calls)
            if reason:
                stats.miss_reasons.append(reason)

        max_savings = stats.no_cache_usd * (1.0 - CACHE_READ_MULTIPLIER)
        leaked = max(max_savings - stats.savings_so_far, 0.0)
        unflagged_reasons = [r for r in stats.miss_reasons if r.est_savings_usd_per_month == 0.0]
        if unflagged_reasons and leaked > 0:
            per_reason = leaked / len(unflagged_reasons)
            for r in unflagged_reasons:
                r.est_savings_usd_per_month = per_reason

        report.routes.append(stats)

    report.routes.sort(
        key=lambda r: (r.no_cache_usd - r.paid_usd) - r.savings_so_far, reverse=True
    )
    return report


def load_records(path: str) -> List[CallRecord]:
    """Load records from a JSONL file."""
    records: List[CallRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(CallRecord.from_log_line(json.loads(line)))
            except Exception:
                continue
    return records
