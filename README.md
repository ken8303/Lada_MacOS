# Lada for Apple Silicon

This workspace contains the first native macOS development slice for
[Lada](https://github.com/ladaapp/lada).

## What works now

- Native SwiftUI three-column queue interface.
- Multi-video import using the macOS file picker.
- AVFoundation metadata, thumbnail and preview loading.
- Editable restoration, memory and output settings.
- Queue start, pause, cancel, retry and completion states.
- JSON-lines worker boundary prepared for Lada on PyTorch MPS.
- Development fallback simulation when the Python runtime is not installed.
- Apple-Silicon-only build configuration.

## Run the development build

```sh
swift run LadaMac
```

To create an ad-hoc-signed development application bundle:

```sh
zsh packaging/macos/package.sh
open outputs/Lada.app
```

The app automatically uses `vendor/lada/.venv/bin/lada-cli` when that
environment exists. The current development workspace includes an isolated
Python 3.12 environment, PyTorch MPS support, the Fast and Accurate detection
models, and the BasicVSR++ v1.2 restoration model. If that runtime is absent,
the worker simulates progress so the native workflow can still be developed.

## Repository layout

- `Sources/LadaMac`: native macOS application.
- `Sources/LadaMac/Resources/lada_worker.py`: engine process adapter.
- `vendor/lada`: upstream Lada source at the inspected revision.
- `work/focus-queue-reference.png`: selected visual target.

## Engine status

- Real Lada export through MPS and Apple VideoToolbox is verified.
- Worker progress is emitted as structured JSON per processed frame.
- Pause, resume, and cancel commands control the full restoration process
  group and are covered by real MPS lifecycle tests.
- Parent-process monitoring prevents GPU restoration jobs from becoming
  orphaned if the native app is force-quit.
- Experimental two-pass restoration can be enabled with
  `LADA_TWO_PASS_RESTORATION=1`; this runs detection first into a temporary
  clip cache, then restores from that cache to avoid detection/restoration
  contention under the serialized Apple MPS lock.
- The app probes the embedded Python runtime, MPS device, and required models
  before enabling queue processing.
- PyAV supplies a development fallback for video metadata when `ffprobe`
  is not installed.
- Compatible audio such as AAC is preserved using an in-bundle PyAV remux,
  without requiring an external FFmpeg installation.
- `packaging/macos/package.sh` produces a self-contained arm64 app and ZIP
  with Python, PyTorch, Lada, models, and licenses bundled.
