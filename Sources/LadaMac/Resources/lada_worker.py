#!/usr/bin/env python3
"""JSON-lines adapter between the native macOS app and Lada's CLI.

The protocol remains stable while packaging work moves from a development
checkout to a fully bundled Python/PyTorch/FFmpeg runtime.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def emit(event_type: str, **payload: object) -> None:
    print(json.dumps({"type": event_type, **payload}), flush=True)


def apply_bundled_runtime_defaults() -> None:
    resources = pathlib.Path(__file__).resolve().parent
    runtime = resources / "runtime"
    python_candidates = (
        runtime / "python" / "bin" / "python3.12",
        runtime / "python" / "bin" / "python3",
        runtime / "python" / "bin" / "python",
    )
    python = next((path for path in python_candidates if path.exists()), None)
    site_packages = runtime / "site-packages"
    source_root = resources / "lada"
    model_dir = source_root / "model_weights"

    if python and "LADA_PYTHON" not in os.environ:
        os.environ["LADA_PYTHON"] = str(python)
    if site_packages.exists() and "LADA_SITE_PACKAGES" not in os.environ:
        os.environ["LADA_SITE_PACKAGES"] = str(site_packages)
    if (source_root / "lada" / "__init__.py").exists() and "LADA_SOURCE_ROOT" not in os.environ:
        os.environ["LADA_SOURCE_ROOT"] = str(source_root)
    if model_dir.exists() and "LADA_MODEL_DIR" not in os.environ:
        os.environ["LADA_MODEL_DIR"] = str(model_dir)
    os.environ.setdefault("LADA_RETRY_REDUCED_MPS_ON_MPS_CRASH", "1")
    os.environ.setdefault("LADA_RETRY_CPU_ON_MPS_CRASH", "0")


def find_lada_command() -> list[str] | None:
    bundled_python = os.environ.get("LADA_PYTHON")
    source_root = os.environ.get("LADA_SOURCE_ROOT")
    if bundled_python and source_root and pathlib.Path(bundled_python).exists():
        return [bundled_python, "-m", "lada.cli.main"]

    executable = shutil.which("lada-cli")
    if executable:
        return [executable]

    if not source_root:
        return None

    candidates = (
        pathlib.Path(source_root) / ".venv" / "bin" / "lada-cli",
        pathlib.Path(source_root) / "dist" / "cli" / "lada-cli",
    )
    executable = next((str(path) for path in candidates if path.exists()), None)
    return [executable] if executable else None


def worker_performance_defaults(memory_mode: str) -> dict[str, str]:
    """Return conservative-by-mode defaults for the production Python engine.

    These remain environment defaults rather than hard overrides so advanced
    users can still experiment from the outside when profiling a specific Mac.
    """
    # Parallel MPS (serialize_mps="0") was tried across modes but reliably
    # crashes PyTorch's MPS backend with a thread-safety bug (arange_mps_out /
    # NSMutableDictionary race) once detection and restoration overlap on the
    # GPU -- confirmed on a real HUNTC-619 run, not just a synthetic test.
    # The crash-retry fallback catches this, but it restarts lada-cli from
    # frame 0 with no checkpoint, so relying on it as the normal path wastes
    # all already-completed work on every crash. Serialize unconditionally
    # until the two-pass (detect-then-restore) restructure removes the need
    # for concurrent MPS access, or the upstream PyTorch race is fixed.
    if memory_mode == "Performance":
        cpu_threads = "4"
        mps_empty_cache_interval = "12"
        serialize_mps = "1"
    elif memory_mode == "Long Video":
        cpu_threads = "2"
        mps_empty_cache_interval = "8"
        serialize_mps = "1"
    elif memory_mode == "Conservative":
        cpu_threads = "1"
        mps_empty_cache_interval = "1"
        serialize_mps = "1"
    else:
        cpu_threads = "2"
        mps_empty_cache_interval = "4"
        serialize_mps = "1"

    return {
        "OMP_NUM_THREADS": cpu_threads,
        "MKL_NUM_THREADS": cpu_threads,
        "VECLIB_MAXIMUM_THREADS": cpu_threads,
        "NUMEXPR_NUM_THREADS": cpu_threads,
        "LADA_MPS_EMPTY_CACHE_INTERVAL": mps_empty_cache_interval,
        "LADA_SERIALIZE_MPS": serialize_mps,
    }


def child_environment(source_root: str | None, request: dict[str, object] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    memory_mode = str((request or {}).get("memoryMode", "Auto (Unified Memory)"))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    environment.setdefault("LADA_RETRY_SERIALIZED_MPS_ON_MPS_CRASH", "1")
    environment.setdefault("LADA_RETRY_REDUCED_MPS_ON_MPS_CRASH", "1")
    environment.setdefault("LADA_RETRY_CPU_ON_MPS_CRASH", "0")
    environment.setdefault("LADA_PROGRESS_CALLBACK_INTERVAL", "1")
    environment.setdefault("LADA_PROGRESS_CALLBACK_MIN_DELTA", "0.001")
    environment.setdefault("LADA_DIAGNOSTIC_WINDOW_SECONDS", "60")
    environment.setdefault("LADA_DIAGNOSTIC_CLIP_INTERVAL", "10")
    environment.setdefault("LADA_DIAGNOSTIC_SLOW_CLIP_SECONDS", "15")
    environment.setdefault("LADA_WRITE_IN_PROGRESS_OUTPUT", "1")
    for key, value in worker_performance_defaults(memory_mode).items():
        environment.setdefault(key, value)
    if memory_mode == "Long Video":
        environment.setdefault("LADA_MAX_DETECTIONS_PER_FRAME", "2")
        environment.setdefault("LADA_MIN_DETECTION_CONFIDENCE", "0.25")
    if environment.get("LADA_EXPERIMENTAL_RAW_METAL") == "1":
        environment.setdefault("PYTORCH_MPS_PREFER_METAL", "1")
    else:
        environment.pop("PYTORCH_MPS_PREFER_METAL", None)
    if site_packages := os.environ.get("LADA_SITE_PACKAGES"):
        python_paths = [site_packages]
        if source_root:
            python_paths.append(source_root)
        existing = environment.get("PYTHONPATH")
        if existing:
            python_paths.append(existing)
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def probe() -> int:
    source_root = os.environ.get("LADA_SOURCE_ROOT")
    bundled_python = os.environ.get("LADA_PYTHON")
    if bundled_python and pathlib.Path(bundled_python).exists():
        python = bundled_python
    elif source_root:
        candidate = pathlib.Path(source_root) / ".venv" / "bin" / "python"
        python = str(candidate) if candidate.exists() else None
    else:
        python = None

    if not python:
        emit("readiness", ready=False, message="Python runtime missing")
        return 1

    code = """
import json
import torch
from lada import ModelFiles
detection = {model.name for model in ModelFiles.get_detection_models()}
restoration = {model.name for model in ModelFiles.get_restoration_models()}
mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()
mps_tensor_device = None
if mps_available:
    mps_tensor_device = str(torch.ones(1, device=torch.device("mps")).device)
ready = (
    mps_available
    and mps_tensor_device == "mps:0"
    and "v4-fast" in detection
    and "v4-accurate" in detection
    and "basicvsrpp-v1.2" in restoration
)
print(json.dumps({
    "ready": ready,
    "mps": mps_available,
    "mps_built": mps_built,
    "mps_tensor_device": mps_tensor_device,
    "detection": sorted(detection),
    "restoration": sorted(restoration),
}))
"""
    result = subprocess.run(
        [python, "-c", code],
        cwd=source_root,
        env=child_environment(source_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        emit(
            "readiness",
            ready=False,
            message=result.stderr.strip() or "Runtime probe failed",
        )
        return result.returncode

    details = json.loads(result.stdout)
    if details["ready"]:
        emit("readiness", ready=True, message="MPS tensor verified · Fast + Accurate models")
        return 0
    missing = []
    if not details["mps"]:
        missing.append("Metal unavailable")
    elif details.get("mps_tensor_device") != "mps:0":
        missing.append("Metal tensor check failed")
    if "v4-fast" not in details["detection"] or "v4-accurate" not in details["detection"]:
        missing.append("detection models missing")
    if "basicvsrpp-v1.2" not in details["restoration"]:
        missing.append("restoration model missing")
    emit("readiness", ready=False, message=", ".join(missing))
    return 1


def simulate(request: dict[str, object]) -> None:
    # Development fallback: exercises native queue, progress, cancellation,
    # and state handling before the multi-gigabyte ML runtime is bundled.
    steps = 80
    started = time.monotonic()
    for step in range(steps + 1):
        progress = step / steps
        elapsed = max(time.monotonic() - started, 0.001)
        remaining = (elapsed / progress - elapsed) if progress else None
        emit("progress", progress=progress, remainingSeconds=remaining)
        time.sleep(0.055)
    emit("completed", output=request["output"], simulated=True)


def control_process(
    process: subprocess.Popen[str],
    cancel_requested: threading.Event,
) -> None:
    """Forward native-app controls to the entire restoration process group."""
    for line in sys.stdin:
        try:
            command = json.loads(line).get("command")
        except (json.JSONDecodeError, AttributeError):
            continue

        if process.poll() is not None:
            return
        try:
            process_group = os.getpgid(process.pid)
            if command == "pause":
                os.killpg(process_group, signal.SIGSTOP)
                emit("paused")
            elif command == "resume":
                os.killpg(process_group, signal.SIGCONT)
                emit("resumed")
            elif command == "cancel":
                cancel_requested.set()
                os.killpg(process_group, signal.SIGINT)
                emit("cancelling")
                return
        except ProcessLookupError:
            return


def stop_process_group(process: subprocess.Popen[str], signal_number: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal_number)
    except ProcessLookupError:
        pass


def monitor_parent(process: subprocess.Popen[str], parent_pid: int) -> None:
    """Stop GPU work if the native parent app exits or is force-quit."""
    while process.poll() is None:
        if os.getppid() != parent_pid:
            emit(
                "log",
                message=f"Native parent exited ({parent_pid} -> {os.getppid()}); stopping restoration.",
            )
            stop_process_group(process, signal.SIGTERM)
            return
        time.sleep(0.5)


def terminate_active_process(signal_number: int, _frame: object) -> None:
    if ACTIVE_PROCESS is not None:
        stop_process_group(ACTIVE_PROCESS, signal.SIGTERM)
    raise SystemExit(128 + signal_number)


def build_lada_command(
    request: dict[str, object],
    lada_command: list[str],
    device: str,
    max_clip_length: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    clip_length = max_clip_length
    if clip_length is None:
        clip_length = int(request.get("maxClipLength", 150))
    command = [
        *lada_command,
        "--input",
        str(request["input"]),
        "--output",
        str(request["output"]),
        "--device",
        device,
        "--mosaic-detection-model",
        str(request.get("detectionModel", "v4-fast")),
        "--max-clip-length",
        str(clip_length),
        "--encoding-preset",
        str(request.get("encodingPreset", "hevc-apple-gpu-balanced")),
        "--progress-json",
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def run_lada_attempt(
    request: dict[str, object],
    lada_command: list[str],
    device: str,
    max_clip_length: int | None = None,
    environment_overrides: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    pass_name: str | None = None,
) -> int:
    global ACTIVE_PROCESS
    source_root = os.environ.get("LADA_SOURCE_ROOT")
    command = build_lada_command(
        request,
        lada_command,
        device,
        max_clip_length=max_clip_length,
        extra_args=extra_args,
    )
    environment = child_environment(source_root, request)
    if environment_overrides:
        environment.update(environment_overrides)
    tuning_keys = (
        "LADA_SERIALIZE_MPS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LADA_MPS_EMPTY_CACHE_INTERVAL",
        "LADA_MAX_DETECTIONS_PER_FRAME",
        "LADA_MIN_DETECTION_CONFIDENCE",
    )
    tuning = {key: environment[key] for key in tuning_keys if key in environment}

    emit(
        "started",
        command=command,
        device=device,
        restorationPass=pass_name,
        maxClipLength=max_clip_length if max_clip_length is not None else int(request.get("maxClipLength", 150)),
        message=f"memoryMode={request.get('memoryMode', 'Auto (Unified Memory)')} · workerTuning={tuning}",
    )
    process = subprocess.Popen(
        command,
        cwd=source_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    ACTIVE_PROCESS = process
    cancel_requested = threading.Event()
    control_thread = threading.Thread(
        target=control_process,
        args=(process, cancel_requested),
        daemon=True,
    )
    control_thread.start()
    parent_monitor = threading.Thread(
        target=monitor_parent,
        args=(process, os.getppid()),
        daemon=True,
    )
    parent_monitor.start()

    # Lada currently renders a terminal progress bar rather than structured
    # events. Keep forwarding logs; the next engine task adds JSON progress
    # directly in the upstream processing loop.
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            emit("log", message=stripped)
            continue
        if event.get("type") == "progress":
            emit(
                "progress",
                progress=event.get("progress", 0),
                remainingSeconds=event.get("remainingSeconds"),
                framesProcessed=event.get("frames_processed"),
                totalFrames=event.get("total_frames"),
                meanFrameSeconds=event.get("mean_frame_seconds"),
                restorationPass=pass_name,
            )
        elif event.get("type") == "diagnostic":
            payload = dict(event)
            payload.pop("type", None)
            if pass_name:
                payload["restorationPass"] = pass_name
            emit("diagnostic", **payload)
        else:
            emit("log", message=stripped)

    return_code = process.wait()
    ACTIVE_PROCESS = None
    if cancel_requested.is_set() or return_code in (-signal.SIGINT, -signal.SIGTERM, 130):
        emit("cancelled")
        return 130
    return return_code


def should_use_two_pass(request: dict[str, object]) -> bool:
    if str(request.get("twoPassRestoration", "")).lower() in {"1", "true", "yes"}:
        return True
    memory_mode = str(request.get("memoryMode", ""))
    if "two-pass" in memory_mode.lower() or "two pass" in memory_mode.lower():
        return True
    return os.environ.get("LADA_TWO_PASS_RESTORATION", "0") == "1"


def run_lada_two_pass_attempt(
    request: dict[str, object],
    lada_command: list[str],
    device: str,
    max_clip_length: int,
    environment_overrides: dict[str, str] | None = None,
) -> int:
    cache_root = pathlib.Path(os.environ.get("LADA_TWO_PASS_CACHE_ROOT", tempfile.gettempdir()))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = pathlib.Path(tempfile.mkdtemp(prefix="lada_detection_cache_", dir=str(cache_root)))
    keep_cache = os.environ.get("LADA_KEEP_TWO_PASS_CACHE", "0") == "1"
    emit("log", message=f"Two-pass restoration enabled · cache={cache_dir}")
    try:
        detect_code = run_lada_attempt(
            request,
            lada_command,
            device,
            max_clip_length=max_clip_length,
            environment_overrides=environment_overrides,
            extra_args=["--cache-detections-to", str(cache_dir)],
            pass_name="detect",
        )
        if detect_code != 0:
            return detect_code
        return run_lada_attempt(
            request,
            lada_command,
            device,
            max_clip_length=max_clip_length,
            environment_overrides=environment_overrides,
            extra_args=["--restore-from-cache", str(cache_dir)],
            pass_name="restore",
        )
    finally:
        if keep_cache:
            emit("log", message=f"Keeping two-pass detection cache for debugging: {cache_dir}")
        else:
            shutil.rmtree(cache_dir, ignore_errors=True)


def run_configured_lada_attempt(
    request: dict[str, object],
    lada_command: list[str],
    device: str,
    max_clip_length: int,
    two_pass: bool,
    environment_overrides: dict[str, str] | None = None,
) -> int:
    if two_pass:
        return run_lada_two_pass_attempt(
            request,
            lada_command,
            device,
            max_clip_length=max_clip_length,
            environment_overrides=environment_overrides,
        )
    return run_lada_attempt(
        request,
        lada_command,
        device,
        max_clip_length=max_clip_length,
        environment_overrides=environment_overrides,
    )


def run_lada(request: dict[str, object], lada_command: list[str]) -> None:
    requested_device = str(request.get("device", "mps"))
    requested_clip_length = int(request.get("maxClipLength", 150))
    fallback_environment_overrides: dict[str, str] | None = None
    two_pass = should_use_two_pass(request)
    return_code = run_configured_lada_attempt(
        request,
        lada_command,
        requested_device,
        max_clip_length=requested_clip_length,
        two_pass=two_pass,
    )
    if return_code == 0:
        emit("progress", progress=1.0, remainingSeconds=0)
        emit("completed", output=request["output"], simulated=False)
        return
    if return_code == 130:
        return

    native_mps_crash = return_code in (-signal.SIGABRT, -signal.SIGSEGV)
    initial_environment = child_environment(os.environ.get("LADA_SOURCE_ROOT"), request)
    retry_serialized_mps = (
        native_mps_crash
        and requested_device == "mps"
        and initial_environment.get("LADA_SERIALIZE_MPS") == "0"
        and initial_environment.get("LADA_RETRY_SERIALIZED_MPS_ON_MPS_CRASH", "1") == "1"
    )
    if retry_serialized_mps:
        emit(
            "log",
            message=(
                "Apple GPU runtime aborted while parallel MPS work was enabled. "
                "Retrying on Apple GPU with serialized MPS enabled."
            ),
        )
        fallback_environment_overrides = {"LADA_SERIALIZE_MPS": "1"}
        try:
            pathlib.Path(str(request["output"])).unlink(missing_ok=True)
        except OSError:
            pass
        return_code = run_configured_lada_attempt(
            request,
            lada_command,
            requested_device,
            max_clip_length=requested_clip_length,
            two_pass=two_pass,
            environment_overrides=fallback_environment_overrides,
        )
        if return_code == 0:
            emit("progress", progress=1.0, remainingSeconds=0)
            emit(
                "completed",
                output=request["output"],
                simulated=False,
                fallbackDevice="mps",
                serializedMPS=True,
            )
            return
        if return_code == 130:
            return
        native_mps_crash = return_code in (-signal.SIGABRT, -signal.SIGSEGV)

    retry_reduced_mps = (
        native_mps_crash
        and requested_device == "mps"
        and os.environ.get("LADA_RETRY_REDUCED_MPS_ON_MPS_CRASH", "1") == "1"
    )
    if retry_reduced_mps:
        tried_clip_lengths = {requested_clip_length}
        reduced_clip_lengths = [
            clip_length for clip_length in (90, 75, 60, 45, 30)
            if clip_length < requested_clip_length and clip_length not in tried_clip_lengths
        ]
        for clip_length in reduced_clip_lengths:
            emit(
                "log",
                message=(
                    "Apple GPU runtime aborted inside MPSGraph. "
                    f"Retrying on Apple GPU with smaller max clip length ({clip_length})."
                ),
            )
            try:
                pathlib.Path(str(request["output"])).unlink(missing_ok=True)
            except OSError:
                pass
            return_code = run_configured_lada_attempt(
                request,
                lada_command,
                "mps",
                max_clip_length=clip_length,
                two_pass=two_pass,
                environment_overrides=fallback_environment_overrides,
            )
            if return_code == 0:
                emit(
                    "progress",
                    progress=1.0,
                    remainingSeconds=0,
                )
                emit(
                    "completed",
                    output=request["output"],
                    simulated=False,
                    fallbackDevice="mps",
                    maxClipLength=clip_length,
                )
                return
            if return_code == 130:
                return
            if return_code not in (-signal.SIGABRT, -signal.SIGSEGV):
                break

    retry_cpu = (
        native_mps_crash
        and requested_device == "mps"
        and os.environ.get("LADA_RETRY_CPU_ON_MPS_CRASH", "0") == "1"
    )
    if retry_cpu:
        emit(
            "log",
            message=(
                "Apple GPU runtime aborted inside MPSGraph. "
                "Retrying this video on CPU so the result can still be produced."
            ),
        )
        try:
            pathlib.Path(str(request["output"])).unlink(missing_ok=True)
        except OSError:
            pass
        return_code = run_configured_lada_attempt(
            request,
            lada_command,
            "cpu",
            max_clip_length=requested_clip_length,
            two_pass=two_pass,
        )
        if return_code == 0:
            emit("progress", progress=1.0, remainingSeconds=0)
            emit("completed", output=request["output"], simulated=False, fallbackDevice="cpu")
            return
        if return_code == 130:
            return

    if native_mps_crash:
        raise RuntimeError(
            "Lada's ML runtime crashed inside native Apple MPSGraph code. "
            "The packaged app serialized MPS work and attempted the configured fallback, "
            f"but lada-cli exited with status {return_code}."
        )
    raise RuntimeError(f"lada-cli exited with status {return_code}")


def main() -> int:
    apply_bundled_runtime_defaults()

    if "--probe" in sys.argv:
        return probe()

    request_line = sys.stdin.readline()
    if not request_line:
        emit("error", message="No worker request was provided.")
        return 2

    try:
        request = json.loads(request_line)
        lada_command = find_lada_command()
        if lada_command:
            run_lada(request, lada_command)
        elif request.get("simulateWhenUnavailable", False):
            simulate(request)
        else:
            raise RuntimeError(
                "lada-cli was not found. Install the Lada Python environment "
                "or use the packaged runtime."
            )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001
        emit("error", message=str(error))
        return 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate_active_process)
    signal.signal(signal.SIGINT, terminate_active_process)
    raise SystemExit(main())
