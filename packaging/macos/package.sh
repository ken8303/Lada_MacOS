#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$ROOT/.build"
PRODUCT_DIR="$BUILD_DIR/arm64-apple-macosx/release"
OUTPUT_DIR="$ROOT/outputs"
STAGING_DIR="/private/tmp/lada-macos-package"
APP="$STAGING_DIR/Lada.app"
OUTPUT_APP="$OUTPUT_DIR/Lada.app"
OUTPUT_ZIP="$OUTPUT_DIR/Lada-Apple-Silicon.zip"
NATIVE_MODELS_SOURCE="$ROOT/native-models"
NATIVE_MODELS_MANIFEST="$OUTPUT_DIR/NATIVE-MODELS.txt"

SITE_PACKAGE_EXCLUDES=(
  "tests"
  "test"
  "yapftests"
  "polars"
  "_polars_runtime_32"
  "polars-1.37.1.dist-info"
  "polars_runtime_32-1.37.1.dist-info"
  "torch/_export/db"
  "torch/testing/_internal/opinfo"
  "torchgen/packaged"
  "torch/include"
  "torch/share"
  "torch/utils/benchmark"
  "torch/utils/tensorboard"
  "torch/_vendor/quack"
  "torch/backends/xnnpack"
  "attrs"
  "attrs-26.1.0.dist-info"
  "cattrs"
  "cattrs-26.1.0.dist-info"
  "colorama"
  "colorama-0.4.6.dist-info"
  "coremltools"
  "coremltools-9.0.dist-info"
  "flatbuffers"
  "flatbuffers-25.12.19.dist-info"
  "ml_dtypes"
  "ml_dtypes-0.5.4.dist-info"
  "onnx-1.22.0.dist-info"
  "onnxruntime"
  "onnxruntime-1.27.0.dist-info"
  "onnxslim"
  "onnxslim-0.1.94.dist-info"
  "protobuf-7.35.1.dist-info"
  "pyaml"
  "pyaml-26.2.1.dist-info"
)

copy_tree() {
  local source="$1"
  local destination="$2"
  /usr/bin/python3 "$ROOT/packaging/macos/copy_tree.py" "$source" "$destination"
}

copy_python_runtime() {
  local source="$1"
  local destination="$2"
  /usr/bin/python3 "$ROOT/packaging/macos/copy_tree.py" "$source" "$destination" \
    "lib/python3.12/idlelib" \
    "lib/python3.12/tkinter" \
    "lib/python3.12/turtledemo" \
    "lib/python3.12/test" \
    "lib/python3.12/ensurepip" \
    "lib/python3.12/site-packages/pip" \
    "lib/python3.12/site-packages/pip-25.1.1.dist-info" \
    "lib/tcl9" \
    "lib/tcl9.0" \
    "lib/tk9.0" \
    "lib/itcl4.3.5" \
    "lib/thread3.0.4"
}

copy_site_packages() {
  local source="$1"
  local destination="$2"
  /usr/bin/python3 "$ROOT/packaging/macos/copy_tree.py" \
    "$source" \
    "$destination" \
    "${SITE_PACKAGE_EXCLUDES[@]}"
}

copy_native_models() {
  local source="$1"
  local destination="$2"
  local manifest="/private/tmp/lada-native-models-to-copy.txt"
  local copied=0
  mkdir -p "$destination"

  if [[ -d "$source" ]]; then
    /usr/bin/find "$source" -maxdepth 1 \( -name "*.mlmodelc" -o -name "*.aimodel" \) -print | sort > "$manifest"
    while IFS= read -r model; do
      [[ -n "$model" ]] || continue
      if [[ -d "$model" ]]; then
        copy_tree "$model" "$destination/$(basename "$model")"
      else
        cp "$model" "$destination/$(basename "$model")"
      fi
      copied=$((copied + 1))
    done < "$manifest"
  fi

  if [[ "$copied" == "0" ]]; then
    echo "No native Core ML/Core AI model assets found in $source"
  else
    echo "Copied $copied native Core ML/Core AI model asset(s)"
  fi
}

hydrate_included_site_packages() {
  local source="$1"
  local dataless_list="/private/tmp/lada-site-packages-dataless.txt"
  local count

  for attempt in {1..12}; do
    /usr/bin/python3 "$ROOT/packaging/macos/copy_tree.py" \
      --list-dataless \
      "$source" \
      "${SITE_PACKAGE_EXCLUDES[@]}" > "$dataless_list"
    count="$(wc -l < "$dataless_list" | tr -d ' ')"
    if [[ "$count" == "0" ]]; then
      return 0
    fi

    echo "Hydrating $count packaged runtime files (attempt $attempt/12)…"
    xargs -n 50 brctl download < "$dataless_list" || true
    sleep 5
  done

  /usr/bin/python3 "$ROOT/packaging/macos/copy_tree.py" \
    --list-dataless \
    "$source" \
    "${SITE_PACKAGE_EXCLUDES[@]}" > "$dataless_list"
  count="$(wc -l < "$dataless_list" | tr -d ' ')"
  if [[ "$count" != "0" ]]; then
    echo "Unable to hydrate $count packaged runtime files:" >&2
    sed -n '1,40p' "$dataless_list" >&2
    return 1
  fi
}

validate_staged_app() {
  local resources="$APP/Contents/Resources"
  local python="$resources/runtime/python/bin/python3.12"
  local probe_output

  codesign --verify --deep --strict --verbose=2 "$APP"

  find "$resources" -maxdepth 1 -type d -name "*.mlmodelc" -print | sort

  if ! probe_output="$(
    env \
      LADA_PYTHON="$python" \
      LADA_SITE_PACKAGES="$resources/runtime/site-packages" \
      LADA_SOURCE_ROOT="$resources/lada" \
      LADA_MODEL_DIR="$resources/lada/model_weights" \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTORCH_ENABLE_MPS_FALLBACK=1 \
      "$python" "$resources/lada_worker.py" --probe
  )"; then
    echo "$probe_output" >&2
    return 1
  fi
  echo "$probe_output"

  if ! probe_output="$(
    env \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTORCH_ENABLE_MPS_FALLBACK=1 \
      "$python" "$resources/lada_worker.py" --probe
  )"; then
    echo "$probe_output" >&2
    return 1
  fi
  echo "$probe_output"

  env \
    LADA_PYTHON="$python" \
    LADA_SITE_PACKAGES="$resources/runtime/site-packages" \
    LADA_SOURCE_ROOT="$resources/lada" \
    LADA_MODEL_DIR="$resources/lada/model_weights" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    PYTHONPATH="$resources/runtime/site-packages:$resources/lada" \
    "$python" - <<'PY'
import torch
import torchvision
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from ultralytics.models.sam.sam3.geometry_encoders import Prompt
from lada.restorationpipeline import load_models
assert torch.backends.mps.is_available(), "MPS backend is unavailable"
assert str(torch.ones(1, device="mps").device) == "mps:0", "MPS tensor allocation failed"
print("runtime imports ok")
PY
}

write_release_notes() {
  local sha
  local native_model_count
  sha="$(shasum -a 256 "$OUTPUT_ZIP" | awk '{print $1}')"
  find "$OUTPUT_APP/Contents/Resources" -maxdepth 1 \( -name "*.mlmodelc" -o -name "*.aimodel" \) -print | sort > "$NATIVE_MODELS_MANIFEST"
  native_model_count="$(wc -l < "$NATIVE_MODELS_MANIFEST" | tr -d ' ')"
  printf '%s\n' "$sha  Lada-Apple-Silicon.zip" > "$OUTPUT_DIR/SHA256.txt"
  {
    printf '%s\n' "# Lada Apple Silicon Build Notes"
    printf '%s\n' ""
    printf '%s\n' "- Build date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf '%s\n' "- Target: Apple Silicon macOS"
    printf '%s\n' "- Swift build: release optimized"
    printf '%s\n' "- Signature: ad-hoc signed"
    printf '%s\n' "- Validation: staged app signature check, worker readiness probe, MPS tensor check, and Torch/Lada import checks"
    printf '%s\n' "- Runtime: bundled Python, PyTorch, Lada source, and model weights"
    printf '%s\n' "- Stability: serializes Apple MPS model execution, defaults to safer 45/60/90 clip lengths, retries smaller MPS clips if MPSGraph aborts, and avoids silent CPU fallback unless explicitly enabled"
    printf '%s\n' "- Performance: Memory Mode now controls MPS clip length and MPS cache flushing frequency; Performance mode uses larger clips and less frequent cache flushes"
    printf '%s\n' "- Defaults: new queue jobs now start with Quality Balanced and Memory Mode Long Video for steadier sustained throughput on dense or multi-hour videos"
    printf '%s\n' "- Encoding: Quality picker now selects Apple VideoToolbox fast/balanced/HQ presets instead of using one fixed preset"
    printf '%s\n' "- Live export output: while a job is processing, the worker writes a visible .in-progress.mp4 video beside the final output, then replaces it with the completed audio+video result at finish"
    printf '%s\n' "- Queue UI: stabilizes worker progress updates so clip-level resets cannot send progress backwards or inflate ETA to unrealistic values"
    printf '%s\n' "- Queue diagnostics: optional per-job progress debug logs record raw/stable progress, ETA, 60-second heartbeat events, and worker lifecycle/log events for long-video troubleshooting"
    printf '%s\n' "- Worker diagnostics: debug logs now include export-window, detect-window, clip-create, clip-restore, and clip-restore-window metrics to identify whether slowdown is detection, restoration, encoding, or dense mosaic sections"
    printf '%s\n' "- Queue performance: throttles high-frequency worker progress, UI updates, and debug-log writes to reduce overhead during long videos"
    printf '%s\n' "- Long Video dense sections: caps low-confidence/high-volume per-frame detections in Long Video mode to reduce tiny-clip explosions, restoration wait time, and runaway ETA on dense mosaic scenes"
    printf '%s\n' "- Queue ETA: hides stale ETA after 5 minutes without progress movement and shows bounded realistic ETA instead of over-optimistic best-speed estimates"
    printf '%s\n' "- Native rewrite: includes RestorationEngine seam and NativeMetalEngine AVFoundation transcode smoke path"
    printf '%s\n' "- Native detection seam: includes detector-shaped confidence, bounding-box, mask-metadata output and a region-provider adapter for future Core ML/YOLO mosaic regions"
    printf '%s\n' "- Native Core ML: loads/caches LadaMosaicDetector.mlmodelc when present, builds letterboxed model input tensors, and parses YOLO-style model outputs"
    printf '%s\n' "- Native Core AI: detects macOS 27 Core AI runtime, reports compute-unit readiness, inspects .aimodel asset metadata/functions when bundled, includes a Core AI engine seam, and is ready to package detector/restorer assets once converted"
    printf '%s\n' "- Native model packaging: copies compiled .mlmodelc bundles and .aimodel assets from native-models/ into app Resources; bundled count: $native_model_count"
    printf '%s\n' "- Native model conversion: includes YOLO-to-Core-ML contract, export helper, and validated ONNX detector intermediate"
    printf '%s\n' "- Native validation: includes Python YOLO reference capture tooling plus Swift fixture decoding and IoU/confidence comparison tests"
    printf '%s\n' "- Native YOLO post-processing: parses exported detector tensor shapes into thresholded, NMS-filtered NativeMosaicDetection values"
    printf '%s\n' "- Native engine: hidden LADA_NATIVE_PROCESSED_FRAMES=1 path uses Core ML detector-backed regions when bundled, then runs crop/resize/injectable-restorer/composite Metal scaffold before writing"
    printf '%s\n' "- Native restoration: includes single-frame and temporal Core ML restorer seams plus an explicit BTCHW LadaMosaicRestorer contract; direct BasicVSR++ conversion is still blocked by unsupported resize/deformable-convolution ops"
    printf '%s\n' "- Native video: uses modern async AVFoundation track metadata loading with warning-clean Swift test/build coverage"
    printf '%s\n' "- Native frame bridge: copies decoded AVFoundation BGRA pixel buffers into Metal and converts processed BGRA frames back to CVPixelBuffer for writing"
    printf '%s\n' "- Native Metal: includes BGRA mask-blend, nearest-resize, crop, and region-composite compute kernels with Swift test coverage"
    printf '%s\n' ""
    printf '%s\n' "SHA-256:"
    printf '%s\n' ""
    printf '%s\n' "\`\`\`"
    printf '%s\n' "$sha  Lada-Apple-Silicon.zip"
    printf '%s\n' "\`\`\`"
  } > "$OUTPUT_DIR/BUILD-NOTES.md"
}

env \
  CLANG_MODULE_CACHE_PATH=/private/tmp/lada-clang-cache \
  SWIFTPM_MODULECACHE_OVERRIDE=/private/tmp/lada-swift-cache \
  swift build \
  -c release \
  --package-path "$ROOT" \
  --scratch-path "$BUILD_DIR" \
  -Xswiftc -gnone

rm -rf "$STAGING_DIR"
rm -rf "$OUTPUT_APP"
rm -f "$OUTPUT_ZIP"
mkdir -p "$APP/Contents/MacOS"
mkdir -p "$APP/Contents/Resources"
cp "$PRODUCT_DIR/LadaMac" "$APP/Contents/MacOS/LadaMac"
cp "$ROOT/packaging/macos/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/Sources/LadaMac/Resources/lada_worker.py" "$APP/Contents/Resources/lada_worker.py"

RUNTIME="$APP/Contents/Resources/runtime"
LADA_RESOURCES="$APP/Contents/Resources/lada"
mkdir -p "$RUNTIME"
mkdir -p "$LADA_RESOURCES"

hydrate_included_site_packages "$ROOT/vendor/lada/.venv/lib/python3.12/site-packages"
copy_python_runtime "$ROOT/.runtime/python/cpython-3.12.13-macos-aarch64-none" "$RUNTIME/python"
copy_site_packages "$ROOT/vendor/lada/.venv/lib/python3.12/site-packages" "$RUNTIME/site-packages"
copy_tree "$ROOT/vendor/lada/lada" "$LADA_RESOURCES/lada"
copy_tree "$ROOT/vendor/lada/configs" "$LADA_RESOURCES/configs"
copy_tree "$ROOT/vendor/lada/model_weights" "$LADA_RESOURCES/model_weights"
cp "$ROOT/vendor/lada/LICENSE.md" "$LADA_RESOURCES/LICENSE.md"
cp -R "$ROOT/vendor/lada/LICENSES" "$LADA_RESOURCES/LICENSES"
copy_native_models "$NATIVE_MODELS_SOURCE" "$APP/Contents/Resources"

xattr -cr "$APP"
codesign --force --deep --sign - "$APP"
validate_staged_app

ditto --norsrc "$APP" "$OUTPUT_APP"
ditto -c -k --keepParent --norsrc "$APP" "$OUTPUT_ZIP"
write_release_notes
echo "$OUTPUT_APP"
echo "$OUTPUT_ZIP"
