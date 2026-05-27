"""Tests for the prompt-cache-doctor analyzer."""

from pathlib import Path

from prompt_cache_doctor.analyzer import (
    CallRecord,
    analyze,
    load_records,
)


def _rec(**kwargs):
    defaults = dict(
        timestamp=None,
        route="/x",
        model="claude-sonnet-4-6",
        system="",
        tool_names=tuple(),
        first_user_msg_hash="x",
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    defaults.update(kwargs)
    return CallRecord(**defaults)


def test_hit_rate_basic():
    records = [
        _rec(input_tokens=0, cache_creation_input_tokens=1000, cache_read_input_tokens=0),
        _rec(input_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=1000),
        _rec(input_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=1000),
    ]
    report = analyze(records)
    assert abs(report.overall_hit_rate - (2000 / 3000)) < 1e-6


def test_no_cache_detector_fires_when_input_is_large():
    records = [
        _rec(route="/big", input_tokens=3000) for _ in range(4)
    ]
    report = analyze(records)
    route = report.routes[0]
    assert any("No cache_control set" in m.label for m in route.miss_reasons)


def test_no_cache_detector_silent_for_tiny_calls():
    records = [_rec(route="/tiny", input_tokens=100) for _ in range(4)]
    report = analyze(records)
    route = report.routes[0]
    assert not any("No cache_control set" in m.label for m in route.miss_reasons)


def test_unsorted_tools_detector():
    records = [
        _rec(route="/agent", tool_names=("a", "b", "c"), cache_creation_input_tokens=2000),
        _rec(route="/agent", tool_names=("b", "a", "c"), cache_creation_input_tokens=2000),
        _rec(route="/agent", tool_names=("c", "a", "b"), cache_creation_input_tokens=2000),
    ]
    report = analyze(records)
    labels = [m.label for r in report.routes for m in r.miss_reasons]
    assert any("different order" in label.lower() for label in labels)


def test_unsorted_tools_silent_when_consistent():
    records = [
        _rec(route="/agent", tool_names=("a", "b", "c")) for _ in range(3)
    ]
    report = analyze(records)
    labels = [m.label for r in report.routes for m in r.miss_reasons]
    assert not any("different order" in label.lower() for label in labels)


def test_tiny_cached_payload_detector():
    records = [
        _rec(cache_creation_input_tokens=400, input_tokens=5000) for _ in range(5)
    ]
    report = analyze(records)
    labels = [m.label for r in report.routes for m in r.miss_reasons]
    assert any("captures very little" in label.lower() for label in labels)


def test_dynamic_system_prefix_detector():
    long_static_tail = "x" * 800
    records = [
        _rec(
            system=f"Current time: 2026-05-27 10:0{i}:00 UTC. Session: abc{i}.\n\n" + long_static_tail,
            cache_creation_input_tokens=1000,
        )
        for i in range(5)
    ]
    report = analyze(records)
    labels = [m.label for r in report.routes for m in r.miss_reasons]
    assert any("prefix changes" in label.lower() for label in labels)


def test_demo_fixture_loads_and_analyzes():
    fixture = Path(__file__).resolve().parent.parent / "prompt_cache_doctor" / "fixtures" / "demo_logs.jsonl"
    records = load_records(str(fixture))
    assert len(records) >= 15
    report = analyze(records)
    assert report.total_calls == len(records)
    assert report.total_paid_usd > 0
    assert report.total_no_cache_usd > 0
    all_miss_labels = [m.label for r in report.routes for m in r.miss_reasons]
    assert all_miss_labels, "demo fixture should surface at least one anti-pattern"


def test_callrecord_from_log_line_handles_list_system():
    line = {
        "route": "/x",
        "request": {
            "model": "claude-sonnet-4-6",
            "system": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        },
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 5},
    }
    rec = CallRecord.from_log_line(line)
    assert "hello" in rec.system and "world" in rec.system


def test_well_tuned_route_has_high_hit_rate():
    fixture = Path(__file__).resolve().parent.parent / "prompt_cache_doctor" / "fixtures" / "demo_logs.jsonl"
    report = analyze(load_records(str(fixture)))
    well_tuned = next((r for r in report.routes if r.route == "/well-tuned"), None)
    assert well_tuned is not None
    assert well_tuned.hit_rate > 0.7
