#!/usr/bin/env python3
"""Compare old vs new Long Video clip-length/cache-interval worker logs."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(normalize_event(event))
    return events


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Accept raw worker JSONL and app progress-debug JSONL.

    Raw worker logs use `type=diagnostic`. The macOS app wraps those as
    `event=worker-diagnostic`. For this comparison, keep the same top-level
    fields so both formats can be analyzed together.
    """
    if event.get("event") == "worker-diagnostic":
        normalized = dict(event)
        normalized["type"] = "diagnostic"
        return normalized
    if event.get("event") == "worker-log":
        normalized = dict(event)
        normalized["type"] = "log"
        return normalized
    if event.get("event") == "worker-error":
        normalized = dict(event)
        normalized["type"] = "error"
        return normalized
    if event.get("event") == "worker-completed":
        normalized = dict(event)
        normalized["type"] = "completed"
        return normalized
    return event


def diagnostics_by_stage(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("type") != "diagnostic":
            continue
        stage = str(event.get("stage", "unknown"))
        by_stage.setdefault(stage, []).append(event)
    return by_stage


def mean_field(records: list[dict[str, Any]], field: str, skip_first: bool = True) -> float | None:
    values = [float(r[field]) for r in records if isinstance(r.get(field), (int, float))]
    if skip_first and len(values) > 1:
        values = values[1:]
    if not values:
        return None
    return statistics.mean(values)


def find_terminal_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("type") in {"error", "cancelled", "completed"}:
            return event
    return events[-1] if events else None


def crash_or_memory_signals(events: list[dict[str, Any]]) -> list[str]:
    signals: list[str] = []
    for event in events:
        etype = event.get("type")
        message = str(event.get("message", ""))
        lowered = message.lower()
        if etype == "error":
            signals.append(f"error: {message or '<no message>'}")
        if etype == "log":
            if "aborted inside mpsgraph" in lowered:
                signals.append(f"MPSGraph abort: {message}")
            elif "retrying" in lowered and "clip length" in lowered:
                signals.append(f"clip-length retry: {message}")
            elif "out of memory" in lowered or "oom" in lowered or "insufficient memory" in lowered:
                signals.append(f"possible OOM: {message}")
            elif "cannot form weak reference" in lowered and "mpsgraph" in lowered:
                signals.append(f"MPSGraph weak-reference crash: {message}")
        if etype == "completed" and event.get("fallbackDevice"):
            signals.append(f"completed via fallback device: {event.get('fallbackDevice')}")
    return signals


def summarize_run(label: str, path: Path) -> dict[str, Any]:
    events = load_events(path)
    by_stage = diagnostics_by_stage(events)
    detect = by_stage.get("detect-window", [])
    export = by_stage.get("export-window", [])
    clip_restore_window = by_stage.get("clip-restore-window", [])
    clip_create = by_stage.get("clip-create", [])
    observed_clip_lengths = [
        float(r["clipLength"])
        for r in clip_create
        if isinstance(r.get("clipLength"), (int, float))
    ]

    return {
        "label": label,
        "path": str(path),
        "total_events": len(events),
        "detect_windows": len(detect),
        "clip_restore_windows": len(clip_restore_window),
        "clip_create_events": len(clip_create),
        "avg_detect_fps": mean_field(detect, "framesPerSecond"),
        "avg_detect_inference_pct": mean_field(detect, "inferencePercent"),
        "avg_export_fps": mean_field(export, "framesPerSecond"),
        "avg_export_wait_pct": mean_field(export, "waitForRestoredFramePercent"),
        "avg_clip_restore_fps": mean_field(clip_restore_window, "clipFramesPerSecond"),
        "max_observed_clip_length": max(observed_clip_lengths) if observed_clip_lengths else None,
        "terminal_event": find_terminal_event(events),
        "signals": crash_or_memory_signals(events),
    }


def pct_str(value: float | None) -> str:
    return f"{value * 100:5.1f}%" if value is not None else "  n/a"


def fps_str(value: float | None) -> str:
    return f"{value:6.2f}" if value is not None else "   n/a"


def print_comparison(old: dict[str, Any], new: dict[str, Any]) -> None:
    print()
    print("=" * 76)
    print("CLIP-LENGTH / CACHE-INTERVAL A/B COMPARISON")
    print("=" * 76)
    print(f"{'':34s} {'old (45 / interval=3)':>20s} {'new (75 / interval=8)':>20s}")
    print(f"{'diagnostic detect windows':34s} {old['detect_windows']:>20d} {new['detect_windows']:>20d}")
    print(f"{'diagnostic restore windows':34s} {old['clip_restore_windows']:>20d} {new['clip_restore_windows']:>20d}")
    print(f"{'detect fps avg':34s} {fps_str(old['avg_detect_fps']):>20s} {fps_str(new['avg_detect_fps']):>20s}")
    print(f"{'detect inferencePercent avg':34s} {pct_str(old['avg_detect_inference_pct']):>20s} {pct_str(new['avg_detect_inference_pct']):>20s}")
    print(f"{'export fps avg':34s} {fps_str(old['avg_export_fps']):>20s} {fps_str(new['avg_export_fps']):>20s}")
    print(f"{'export waitForRestoredFrame avg':34s} {pct_str(old['avg_export_wait_pct']):>20s} {pct_str(new['avg_export_wait_pct']):>20s}")
    print(f"{'clip-restore fps avg':34s} {fps_str(old['avg_clip_restore_fps']):>20s} {fps_str(new['avg_clip_restore_fps']):>20s}")
    print(f"{'clip-create events':34s} {old['clip_create_events']:>20d} {new['clip_create_events']:>20d}")
    print(f"{'max observed clipLength':34s} {str(old['max_observed_clip_length']):>20s} {str(new['max_observed_clip_length']):>20s}")
    print()

    for run in (old, new):
        terminal = run["terminal_event"]
        if terminal:
            print(f"[{run['label']}] terminal event: {terminal.get('type', terminal.get('event', 'unknown'))}")
        if run["signals"]:
            print(f"[{run['label']}] crash/memory signals:")
            for signal in run["signals"]:
                print(f"  - {signal}")
    print()

    print("-" * 76)
    print("VERDICT")
    print("-" * 76)

    if new["signals"]:
        print("New settings produced crash or memory signals.")
        print("Recommendation: test a midpoint such as 60 before keeping 75 as the Long Video default.")
        return

    print("No crash/memory signals were detected in the new-settings run.")
    old_fps = old["avg_detect_fps"]
    new_fps = new["avg_detect_fps"]
    improvement: float | None = None
    if old_fps is not None and new_fps is not None and old_fps > 0:
        improvement = (new_fps / old_fps - 1) * 100
        print(f"Detect throughput changed by {improvement:+.1f}% vs old settings.")

    old_clips = old["clip_create_events"]
    new_clips = new["clip_create_events"]
    if old_clips > 0:
        clip_reduction = (1 - new_clips / old_clips) * 100
        print(f"Clip-create events changed by {clip_reduction:+.1f}% fewer vs old settings.")

    if improvement is not None and improvement > 10:
        print("Recommendation: keep the 75/8 Long Video tuning, then validate once on a longer or lower-memory-machine run.")
    else:
        print("Recommendation: 75/8 appears safe here, but not a clear speed win. The two-pass restructure remains the next likely big lever.")


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} old_settings.raw.jsonl new_settings.raw.jsonl", file=sys.stderr)
        return 2

    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])
    for path in (old_path, new_path):
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 2

    old = summarize_run("old_settings", old_path)
    new = summarize_run("new_settings", new_path)
    print_comparison(old, new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
