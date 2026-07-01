#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT=""
WORKER_SCRIPT=""
PYTHON_BIN=""
OUTPUT_DIR="$REPO_ROOT/clip_length_ab_results"
MEMORY_MODE="Long Video"
DETECTION_MODEL="v4-fast"
ENCODING_PRESET="hevc-apple-gpu-balanced"
DEVICE="mps"
OLD_CLIP_LENGTH=45
NEW_CLIP_LENGTH=75
OLD_CACHE_INTERVAL=3
NEW_CACHE_INTERVAL=8
DIAGNOSTIC_WINDOW_SECONDS=10
DIAGNOSTIC_CLIP_INTERVAL=1

usage() {
  sed -n '2,120p' "$0" | sed 's/^# \{0,1\}//'
}

# Validate Long Video dense-scene tuning by running the same clip twice:
# old settings: maxClipLength=45, LADA_MPS_EMPTY_CACHE_INTERVAL=3
# new settings: maxClipLength=75, LADA_MPS_EMPTY_CACHE_INTERVAL=8
#
# Usage:
#   tools/run_clip_length_ab_test.sh --input dense_test_clip.mp4
#
# Optional:
#   --worker-script path/to/lada_worker.py
#   --python path/to/python
#   --output-dir ./clip_length_ab_results
#   --memory-mode "Long Video"
#   --old-clip-length 45
#   --new-clip-length 75
#   --old-cache-interval 3
#   --new-cache-interval 8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --worker-script) WORKER_SCRIPT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --memory-mode) MEMORY_MODE="$2"; shift 2 ;;
    --detection-model) DETECTION_MODEL="$2"; shift 2 ;;
    --encoding-preset) ENCODING_PRESET="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --old-clip-length) OLD_CLIP_LENGTH="$2"; shift 2 ;;
    --new-clip-length) NEW_CLIP_LENGTH="$2"; shift 2 ;;
    --old-cache-interval) OLD_CACHE_INTERVAL="$2"; shift 2 ;;
    --new-cache-interval) NEW_CACHE_INTERVAL="$2"; shift 2 ;;
    --diagnostic-window-seconds) DIAGNOSTIC_WINDOW_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${INPUT:?--input is required. Use tools/cut_test_clip.sh first for the dense segment.}"

if [[ -z "$WORKER_SCRIPT" ]]; then
  if [[ -f "$REPO_ROOT/outputs/Lada.app/Contents/Resources/lada_worker.py" ]]; then
    WORKER_SCRIPT="$REPO_ROOT/outputs/Lada.app/Contents/Resources/lada_worker.py"
  else
    WORKER_SCRIPT="$REPO_ROOT/Sources/LadaMac/Resources/lada_worker.py"
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ "$WORKER_SCRIPT" == "$REPO_ROOT/outputs/Lada.app/Contents/Resources/lada_worker.py" ]] && [[ -x "$REPO_ROOT/outputs/Lada.app/Contents/Resources/runtime/python/bin/python3" ]]; then
    PYTHON_BIN="$REPO_ROOT/outputs/Lada.app/Contents/Resources/runtime/python/bin/python3"
  elif [[ -x "$REPO_ROOT/vendor/lada/.venv/bin/python3" ]]; then
    PYTHON_BIN="$REPO_ROOT/vendor/lada/.venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -f "$WORKER_SCRIPT" ]]; then
  echo "Worker script not found: $WORKER_SCRIPT" >&2
  exit 2
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Input clip not found: $INPUT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
INPUT_ABS="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"

echo "Worker: $WORKER_SCRIPT"
echo "Python: $PYTHON_BIN"
echo "Input:  $INPUT_ABS"
echo "Output: $OUTPUT_DIR"
echo

run_variant() {
  local variant_name="$1"
  local clip_length="$2"
  local cache_interval="$3"
  local output_video="$OUTPUT_DIR/${variant_name}.mp4"
  local raw_log="$OUTPUT_DIR/${variant_name}.raw.jsonl"

  echo "=== Running variant: $variant_name (maxClipLength=$clip_length, cacheInterval=$cache_interval) ==="

  local request_json
  request_json="$("$PYTHON_BIN" - "$INPUT_ABS" "$output_video" "$DEVICE" "$MEMORY_MODE" "$clip_length" "$DETECTION_MODEL" "$ENCODING_PRESET" <<'PY'
import json
import sys

input_path, output_path, device, memory_mode, max_clip_length, detection_model, encoding_preset = sys.argv[1:8]
print(json.dumps({
    "input": input_path,
    "output": output_path,
    "device": device,
    "memoryMode": memory_mode,
    "maxClipLength": int(max_clip_length),
    "detectionModel": detection_model,
    "encodingPreset": encoding_preset,
    "simulateWhenUnavailable": False,
}))
PY
)"

  local start_ts end_ts
  start_ts="$(date +%s)"
  echo "$request_json" | env \
    LADA_SERIALIZE_MPS="1" \
    LADA_MPS_EMPTY_CACHE_INTERVAL="$cache_interval" \
    LADA_DIAGNOSTIC_WINDOW_SECONDS="$DIAGNOSTIC_WINDOW_SECONDS" \
    LADA_DIAGNOSTIC_CLIP_INTERVAL="$DIAGNOSTIC_CLIP_INTERVAL" \
    "$PYTHON_BIN" "$WORKER_SCRIPT" > "$raw_log" 2>&1
  local exit_code=$?
  end_ts="$(date +%s)"

  echo "exit code: $exit_code"
  echo "wall time: $((end_ts - start_ts))s"
  echo "log:       $raw_log"
  if [[ "$exit_code" -ne 0 ]]; then
    echo "note: non-zero exit. The comparison step will still inspect the log for crash/memory signals."
  fi
  echo
}

run_variant "old_settings" "$OLD_CLIP_LENGTH" "$OLD_CACHE_INTERVAL"
run_variant "new_settings" "$NEW_CLIP_LENGTH" "$NEW_CACHE_INTERVAL"

echo "=== Comparing runs ==="
COMPARE_SCRIPT="$SCRIPT_DIR/compare_clip_length_logs.py"
if [[ -f "$COMPARE_SCRIPT" ]]; then
  "$PYTHON_BIN" "$COMPARE_SCRIPT" "$OUTPUT_DIR/old_settings.raw.jsonl" "$OUTPUT_DIR/new_settings.raw.jsonl"
else
  echo "Comparison script not found: $COMPARE_SCRIPT" >&2
  exit 2
fi
