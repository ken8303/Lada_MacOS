#!/usr/bin/env bash
set -euo pipefail

INPUT=""
OUTPUT=""
START_FRAME=""
END_FRAME=""
FPS=""

usage() {
  sed -n '2,120p' "$0" | sed 's/^# \{0,1\}//'
}

# Cut a short dense-segment test clip by frame range.
#
# Usage:
#   tools/cut_test_clip.sh \
#     --input source.mp4 \
#     --start-frame 11150 \
#     --end-frame 12500 \
#     --output dense_test_clip.mp4
#
# Optional:
#   --fps 29.97
#
# If --fps is omitted, the script asks ffprobe for the source video frame rate.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --start-frame) START_FRAME="$2"; shift 2 ;;
    --end-frame) END_FRAME="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${INPUT:?--input is required}"
: "${OUTPUT:?--output is required}"
: "${START_FRAME:?--start-frame is required}"
: "${END_FRAME:?--end-frame is required}"

if [[ ! -f "$INPUT" ]]; then
  echo "Input video not found: $INPUT" >&2
  exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required for cutting the test clip." >&2
  exit 2
fi

if [[ -z "$FPS" ]]; then
  if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffprobe not found. Pass --fps manually, for example --fps 29.97" >&2
    exit 2
  fi
  FPS="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT")"
fi

read -r START_SECONDS DURATION_SECONDS < <(
  python3 - "$START_FRAME" "$END_FRAME" "$FPS" <<'PY'
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

echo "Wrote dense test clip: $OUTPUT"
