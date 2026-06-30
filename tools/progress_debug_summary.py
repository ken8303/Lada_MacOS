#!/usr/bin/env python3
"""Summarize a Lada Mac progress-debug JSONL log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to *.progress-debug.jsonl")
    parser.add_argument(
        "--stall-window",
        type=float,
        default=300,
        help="Seconds without stable progress movement to classify as a stall.",
    )
    return parser.parse_args()


def load_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            events.append({
                "event": "decode-error",
                "line": line_number,
                "message": str(error),
            })
    return events


def progress_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [event for event in events if event.get("event") == "progress"]


def heartbeat_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [event for event in events if event.get("event") == "heartbeat"]


def worker_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        event for event in events
        if isinstance(event.get("event"), str) and str(event["event"]).startswith("worker-")
    ]


def diagnostic_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        event for event in events
        if event.get("event") == "worker-diagnostic"
    ]


def summarize(events: list[dict[str, object]], stall_window: float) -> dict[str, object]:
    progress = progress_events(events)
    heartbeats = heartbeat_events(events)
    workers = worker_events(events)
    diagnostics = diagnostic_events(events)
    cpu_fallback_events = cpu_fallback_worker_events(workers)
    raw_regressions = [
        event for event in progress
        if isinstance(event.get("rawProgress"), (int, float))
        and isinstance(event.get("stableProgress"), (int, float))
        and float(event["rawProgress"]) < float(event["stableProgress"])
    ]
    huge_raw_eta = [
        event for event in progress
        if isinstance(event.get("rawRemainingSeconds"), (int, float))
        and float(event["rawRemainingSeconds"]) >= 24 * 3600
    ]
    stable_eta_increases_while_stalled = count_stable_eta_increases_while_stalled(progress)
    stable_eta_increases = count_stable_eta_increases(progress)

    first = progress[0] if progress else None
    last = progress[-1] if progress else None
    max_stable_progress = max(
        [float(event["stableProgress"]) for event in progress if isinstance(event.get("stableProgress"), (int, float))],
        default=0.0,
    )

    last_progress_change_elapsed = None
    previous_stable_progress = None
    for event in progress:
        stable = event.get("stableProgress")
        elapsed = event.get("elapsedSeconds")
        if not isinstance(stable, (int, float)) or not isinstance(elapsed, (int, float)):
            continue
        if previous_stable_progress is None or float(stable) > previous_stable_progress:
            last_progress_change_elapsed = float(elapsed)
            previous_stable_progress = float(stable)

    elapsed_last = float(last["elapsedSeconds"]) if last and isinstance(last.get("elapsedSeconds"), (int, float)) else None
    stalled_seconds = (
        elapsed_last - last_progress_change_elapsed
        if elapsed_last is not None and last_progress_change_elapsed is not None
        else None
    )
    throughput = throughput_summary(progress)

    return {
        "events": len(events),
        "progress_events": len(progress),
        "heartbeat_events": len(heartbeats),
        "worker_events": len(workers),
        "diagnostic_events": len(diagnostics),
        "cpu_fallback_events": len(cpu_fallback_events),
        "first_progress": first,
        "last_progress": last,
        "last_heartbeat": heartbeats[-1] if heartbeats else None,
        "last_worker_event": workers[-1] if workers else None,
        "max_stable_progress": max_stable_progress,
        "raw_progress_regressions": len(raw_regressions),
        "huge_raw_eta_events": len(huge_raw_eta),
        "stable_eta_increases": stable_eta_increases,
        "stable_eta_increases_while_stalled": stable_eta_increases_while_stalled,
        "last_progress_change_elapsed_seconds": last_progress_change_elapsed,
        "stalled_seconds": stalled_seconds,
        "throughput": throughput,
        "diagnostics": diagnostics_summary(diagnostics),
        "is_stalled": stalled_seconds is not None and stalled_seconds >= stall_window,
        "recommendation": recommendation(
            progress_events_count=len(progress),
            raw_regressions_count=len(raw_regressions),
            huge_raw_eta_count=len(huge_raw_eta),
            stable_eta_increases=stable_eta_increases,
            stable_eta_increases_while_stalled=stable_eta_increases_while_stalled,
            stalled_seconds=stalled_seconds,
            stall_window=stall_window,
        ),
    }


def diagnostics_summary(diagnostics: list[dict[str, object]]) -> dict[str, object]:
    if not diagnostics:
        return {}

    by_stage: dict[str, int] = {}
    for event in diagnostics:
        stage = str(event.get("stage") or "unknown")
        by_stage[stage] = by_stage.get(stage, 0) + 1

    export_windows = [
        event for event in diagnostics
        if event.get("stage") == "export-window"
    ]
    detect_windows = [
        event for event in diagnostics
        if event.get("stage") == "detect-window"
    ]
    clip_restores = [
        event for event in diagnostics
        if event.get("stage") == "clip-restore"
    ]
    clip_restore_windows = [
        event for event in diagnostics
        if event.get("stage") == "clip-restore-window"
    ]
    clip_creates = [
        event for event in diagnostics
        if event.get("stage") == "clip-create"
    ]
    dense_filters = [
        event for event in diagnostics
        if event.get("stage") == "dense-filter"
    ]

    return {
        "events_by_stage": by_stage,
        "dense_filter": dense_filter_summary(dense_filters),
        "last_export_window": export_windows[-1] if export_windows else None,
        "slowest_export_window": max_by(export_windows, "framesPerSecond", lowest=True),
        "last_detect_window": detect_windows[-1] if detect_windows else None,
        "slowest_detect_window": max_by(detect_windows, "framesPerSecond", lowest=True),
        "clip_create_events": len(clip_creates),
        "clip_restore_events": len(clip_restores),
        "slowest_clip_restore": max_by(clip_restores, "durationSeconds"),
        "last_clip_restore_window": clip_restore_windows[-1] if clip_restore_windows else None,
        "slowest_clip_restore_window": max_by(clip_restore_windows, "clipFramesPerSecond", lowest=True),
    }


def dense_filter_summary(events: list[dict[str, object]]) -> dict[str, object]:
    if not events:
        return {
            "events": 0,
            "total_dropped_detections": 0,
        }
    total_dropped = sum(
        int(event.get("droppedDetections") or 0)
        for event in events
    )
    total_original = sum(
        int(event.get("originalDetections") or 0)
        for event in events
    )
    total_kept = sum(
        int(event.get("detectionsInLastBatch") or 0)
        for event in events
    )
    return {
        "events": len(events),
        "total_original_detections": total_original,
        "total_kept_detections": total_kept,
        "total_dropped_detections": total_dropped,
        "last_event": events[-1],
    }


def max_by(
    events: list[dict[str, object]],
    field: str,
    lowest: bool = False,
) -> dict[str, object] | None:
    candidates = [
        event for event in events
        if isinstance(event.get(field), (int, float))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda event: float(event[field]), reverse=not lowest)[0]


def throughput_summary(progress: list[dict[str, object]]) -> dict[str, object]:
    if not progress:
        return {}
    last = progress[-1]
    last_progress = last.get("stableProgress")
    last_elapsed = last.get("elapsedSeconds")
    if not isinstance(last_progress, (int, float)) or not isinstance(last_elapsed, (int, float)):
        return {}
    last_progress = float(last_progress)
    last_elapsed = float(last_elapsed)
    windows = {}
    for window_seconds in (300, 600, 1200, 1800):
        windows[f"{window_seconds // 60}m"] = window_eta(progress, last_progress, last_elapsed, window_seconds)
    average_eta = (
        last_elapsed * (1 - last_progress) / last_progress
        if last_progress > 0
        else None
    )
    return {
        "elapsed_minutes": last_elapsed / 60,
        "stable_progress_percent": last_progress * 100,
        "average_progress_percent_per_hour": (last_progress / last_elapsed * 3600 * 100) if last_elapsed > 0 else None,
        "eta_hours_from_average_speed": (average_eta / 3600) if average_eta is not None else None,
        "eta_hours_by_recent_windows": windows,
    }


def window_eta(
    progress: list[dict[str, object]],
    last_progress: float,
    last_elapsed: float,
    window_seconds: int,
) -> float | None:
    cutoff = last_elapsed - window_seconds
    previous = next(
        (
            event for event in reversed(progress)
            if isinstance(event.get("elapsedSeconds"), (int, float))
            and float(event["elapsedSeconds"]) <= cutoff
            and isinstance(event.get("stableProgress"), (int, float))
        ),
        None,
    )
    if previous is None:
        return None
    previous_progress = float(previous["stableProgress"])
    previous_elapsed = float(previous["elapsedSeconds"])
    delta_progress = last_progress - previous_progress
    delta_elapsed = last_elapsed - previous_elapsed
    if delta_progress <= 0 or delta_elapsed <= 0:
        return None
    return (1 - last_progress) * delta_elapsed / delta_progress / 3600


def cpu_fallback_worker_events(workers: list[dict[str, object]]) -> list[dict[str, object]]:
    cpu_events = []
    for event in workers:
        note = str(event.get("note", "")).lower()
        if "device=cpu" in note or "fallbackdevice=cpu" in note:
            cpu_events.append(event)
    return cpu_events


def count_stable_eta_increases(progress: list[dict[str, object]]) -> int:
    count = 0
    previous_stable_eta = None
    for event in progress:
        stable_eta = event.get("stableRemainingSeconds")
        if not isinstance(stable_eta, (int, float)):
            continue
        stable_eta = float(stable_eta)
        if previous_stable_eta is not None and stable_eta > previous_stable_eta + 1:
            count += 1
        previous_stable_eta = stable_eta
    return count


def count_stable_eta_increases_while_stalled(progress: list[dict[str, object]]) -> int:
    count = 0
    previous_stable_progress = None
    previous_stable_eta = None
    for event in progress:
        stable_progress = event.get("stableProgress")
        stable_eta = event.get("stableRemainingSeconds")
        if not isinstance(stable_progress, (int, float)) or not isinstance(stable_eta, (int, float)):
            continue
        stable_progress = float(stable_progress)
        stable_eta = float(stable_eta)
        if (
            previous_stable_progress is not None
            and previous_stable_eta is not None
            and stable_progress <= previous_stable_progress + 0.000_001
            and stable_eta > previous_stable_eta + 1
        ):
            count += 1
        previous_stable_progress = stable_progress
        previous_stable_eta = stable_eta
    return count


def recommendation(
    progress_events_count: int,
    raw_regressions_count: int,
    huge_raw_eta_count: int,
    stable_eta_increases: int,
    stable_eta_increases_while_stalled: int,
    stalled_seconds: float | None,
    stall_window: float,
) -> str:
    if progress_events_count == 0:
        return "No progress events were recorded. Check whether the worker started or emitted JSON progress."
    if stable_eta_increases_while_stalled:
        return "Displayed ETA increased while stable progress was stalled. The UI ETA stabilizer needs review."
    if stable_eta_increases:
        return "Displayed ETA increased during the job. This can be valid when current processing speed is slower than the earlier best-speed estimate."
    if stalled_seconds is not None and stalled_seconds >= stall_window:
        return "Stable progress has stalled. Inspect worker logs and backend processing around the last progress event."
    if raw_regressions_count or huge_raw_eta_count:
        return "Worker progress/ETA is unstable, but the UI stabilizer is protecting the displayed values."
    return "Progress events look stable in this log."


def main() -> int:
    args = parse_args()
    if not args.log.exists():
        print(json.dumps({"log": str(args.log), "exists": False}, indent=2))
        return 2
    print(json.dumps(summarize(load_events(args.log), args.stall_window), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
