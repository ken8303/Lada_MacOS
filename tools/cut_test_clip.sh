#!/usr/bin/env bash
set -euo pipefail

INPUT=""
OUTPUT=""
START_FRAME=""
END_FRAME=""
FPS=""
PYTHON_BIN=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'USAGE'
Cut a short dense-segment test clip by frame range.

Usage:
  tools/cut_test_clip.sh \
    --input source.mp4 \
    --start-frame 11150 \
    --end-frame 12500 \
    --output dense_test_clip.mp4

Optional:
  --fps 29.97
  --python vendor/lada/.venv/bin/python3

If ffmpeg is unavailable, the script falls back to PyAV from the vendored
Lada development runtime. If --fps is omitted, the script asks ffprobe first
and then PyAV for the source video frame rate.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --start-frame) START_FRAME="$2"; shift 2 ;;
    --end-frame) END_FRAME="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${INPUT:?--input is required}"
: "${OUTPUT:?--output is required}"
: "${START_FRAME:?--start-frame is required}"
: "${END_FRAME:?--end-frame is required}"

if [[ ! -f "$INPUT" ]]; then
  mapfile -t INPUT_MATCHES < <(compgen -G "$INPUT" || true)
  if [[ "${#INPUT_MATCHES[@]}" -eq 1 ]]; then
    INPUT="${INPUT_MATCHES[0]}"
  elif [[ "${#INPUT_MATCHES[@]}" -gt 1 ]]; then
    echo "Input pattern matched more than one file. Please pass the exact video path:" >&2
    printf '  %s\n' "${INPUT_MATCHES[@]}" >&2
    exit 2
  else
    echo "Input video not found: $INPUT" >&2
    exit 2
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/vendor/lada/.venv/bin/python3" ]]; then
    PYTHON_BIN="$REPO_ROOT/vendor/lada/.venv/bin/python3"
  elif [[ -x "$REPO_ROOT/outputs/Lada.app/Contents/Resources/runtime/python/bin/python3" ]]; then
    PYTHON_BIN="$REPO_ROOT/outputs/Lada.app/Contents/Resources/runtime/python/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "$FPS" ]]; then
  if command -v ffprobe >/dev/null 2>&1; then
    FPS="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT")"
  else
    FPS="$("$PYTHON_BIN" - "$INPUT" <<'PY'
from __future__ import annotations
from fractions import Fraction
import sys

try:
    import av
except Exception as error:
    raise SystemExit(f"PyAV is required when ffprobe is unavailable: {error}")

with av.open(sys.argv[1]) as container:
    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        raise SystemExit("No video stream found")
    rate = stream.average_rate or stream.base_rate
    if rate is None:
        raise SystemExit("Could not detect frame rate. Pass --fps manually.")
    print(str(Fraction(rate)))
PY
)"
  fi
fi

read -r START_SECONDS DURATION_SECONDS < <(
  "$PYTHON_BIN" - "$START_FRAME" "$END_FRAME" "$FPS" <<'PY'
from __future__ import annotations
from fractions import Fraction
import sys

start_frame = int(sys.argv[1])
end_frame = int(sys.argv[2])
fps = float(Fraction(sys.argv[3]))
if end_frame <= start_frame:
    raise SystemExit("--end-frame must be greater than --start-frame")
start = start_frame / fps
duration = (end_frame - start_frame) / fps
print(f"{start:.6f} {duration:.6f}")
PY
)

mkdir -p "$(dirname "$OUTPUT")"

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg \
    -hide_banner \
    -y \
    -ss "$START_SECONDS" \
    -i "$INPUT" \
    -t "$DURATION_SECONDS" \
    -map 0:v:0 \
    -an \
    -c:v libx264 \
    -preset veryfast \
    -crf 18 \
    "$OUTPUT"
else
  "$PYTHON_BIN" "$SCRIPT_DIR/cut_test_clip_pyav.py" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --start-frame "$START_FRAME" \
    --end-frame "$END_FRAME"
fi

echo "Wrote dense test clip: $OUTPUT"
