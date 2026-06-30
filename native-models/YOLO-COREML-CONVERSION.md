# YOLO mosaic detector → Core ML / Core AI conversion contract

This folder is the drop-in location for future native Core ML model bundles and
Core AI model assets.

## Source weights

Current PyTorch/Ultralytics detector candidates:

- Fast: `vendor/lada/model_weights/lada_mosaic_detection_model_v4_fast.pt`
- Accurate: `vendor/lada/model_weights/lada_mosaic_detection_model_v4_accurate.pt`

The first native detector target should use the fast model unless accuracy
regresses badly, because it is the safer default for on-device video processing.

## Packaged model name

The Swift app looks for:

```text
native-models/LadaMosaicDetector.mlmodelc
native-models/LadaMosaicDetector.aimodel
```

During packaging, these native model assets are copied to:

```text
Lada.app/Contents/Resources/LadaMosaicDetector.mlmodelc
Lada.app/Contents/Resources/LadaMosaicDetector.aimodel
```

## Initial conversion target

- Input image size: `640 × 640`
- Stride: `32`
- Input layout before model: RGB, normalized to `0...1`
- Current Python preprocessing: Ultralytics letterbox with `auto=True`
- Current Python confidence default: `0.25` inside the model wrapper
- Current Python evaluation confidence default: `0.4`
- Current Python IoU default: `0.7`

## Swift output contract

Converted detector output must eventually map into:

```swift
NativeMosaicDetection(
    confidence: Float,
    boundingBox: NativeRestorationRegion,
    mask: NativeDetectionMaskMetadata?
)
```

Required post-processing responsibilities:

1. Run model inference.
2. Decode YOLO boxes and mask prototypes.
3. Apply confidence filtering.
4. Apply non-maximum suppression.
5. Scale coordinates from model/letterbox space back to source-frame pixels.
6. Emit one `NativeMosaicDetection` per valid mosaic region.

Current Swift scaffolding already handles:

- letterboxed RGB input tensor creation,
- cached compiled Core ML model loading,
- `output0`/`output1` tensor extraction,
- YOLO output shape parsing,
- letterbox-aware source-frame coordinate mapping,
- confidence threshold filtering,
- frame-bound clipping,
- simple IoU-based suppression,
- one-or-many restoration regions,
- crop → model-size resize → placeholder restore → resize-back → composite.

## Validation checklist

Before replacing the Python detector:

1. Pick 10 fixed validation frames.
2. Run the current Python detector reference capture:

   ```text
   PYTHONPATH=vendor/lada vendor/lada/.venv/bin/python tools/native-models/capture_yolo_reference.py <input-video-or-image> --frames 0,30,60 --device cpu
   ```

3. Save:
   - boxes,
   - confidence,
   - mask dimensions,
   - frame dimensions.
4. Run the Core ML detector on the same frames.
5. Compare:
   - detection count,
   - box IoU,
   - confidence rank ordering,
   - mask non-empty area.
6. Only then make the native detector the default production detector path.

Swift already includes the comparison harness for boxes and confidence:
`NativeDetectorReferenceComparator` matches detections by IoU, reports
confidence drift, and flags missing or extra detections.

Current baseline fixture:

```text
native-models/reference-detections/smoke-input-yolo-reference.json
```

## Packaging

Drop a compiled `.mlmodelc` bundle into `native-models/` and run:

```text
sh packaging/macos/package.sh
```

For Core AI on macOS 27, drop `LadaMosaicDetector.aimodel` into
`native-models/` and run the same packaging command. The app will inspect the
asset metadata/functions in Settings and mark invalid assets as needing
attention instead of reporting false readiness.

The build writes:

```text
outputs/NATIVE-MODELS.txt
```

If no native models are present, the app remains safe: the Core ML detector
reports missing availability and emits no detections.

## Current export status

As of 2026-06-27:

- Core ML export works with the opt-in helper workaround:

  ```text
  PYTHONPATH=vendor/lada vendor/lada/.venv/bin/python tools/native-models/export_yolo_to_coreml.py --format coreml --run-export --patch-coremltools-scalar-casts --output native-models/LadaMosaicDetector.mlmodelc
  ```

- Current Core ML artifact: `native-models/LadaMosaicDetector.mlmodelc`
- Core ML input: `image`, 640 × 640 RGB `ImageType`
- Core ML outputs:
  - `var_1324`, shape `1 × 38 × 8400`
  - `var_1362`, shape `1 × 32 × 160 × 160`
- Swift loads and executes the compiled model in a smoke test.
- Swift compares the compiled model against the Python smoke-frame fixture and
  passes the IoU/confidence calibration threshold.
- ONNX intermediate export works.
- Current ONNX artifact: `native-models/LadaMosaicDetector.onnx`
- ONNX input: `images`, shape `1 × 3 × 640 × 640`
- ONNX outputs:
  - `output0`, shape `1 × 38 × 8400`
  - `output1`, shape `1 × 32 × 160 × 160`
- ONNX opset: `20`

The remaining path is:

1. Keep the ONNX artifact as a validated intermediate/debug reference.
2. Expand Core ML detector comparison against Python references on real
   validation clips.
3. Convert the detector to Core AI once Apple's macOS 27 `coreai-torch`
   converter package is installed:

   ```text
   PYTHONPATH=vendor/lada vendor/lada/.venv/bin/python tools/native-models/export_yolo_to_coreai.py --run-export --output native-models/LadaMosaicDetector.aimodel
   ```

   Current helper status on this machine: dry-run works, but the Python
   converter package is not installed yet. The macOS 27 SDK/runtime frameworks
   are present, so this is a converter-tooling blocker rather than an app
   packaging/runtime blocker.
4. Once parity is acceptable, wire `NativeCoreMLMosaicDetector` into the native
   processed-frame path as the detector provider.
