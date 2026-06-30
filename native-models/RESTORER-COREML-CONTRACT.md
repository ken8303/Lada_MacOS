# LadaMosaicRestorer Core ML Contract

Status: proposed native contract for the future `LadaMosaicRestorer.mlmodelc`.

The current shipping restoration path still uses Python/PyTorch BasicVSR++.
The native app now has a temporal restorer seam, but the real Core ML restorer
bundle is not present yet.

## Source model

- Lada model name: `basicvsrpp-v1.2`
- Source weights: `vendor/lada/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth`
- Python loader: `vendor/lada/lada/models/basicvsrpp/inference.py`
- Python runtime wrapper: `vendor/lada/lada/restorationpipeline/basicvsrpp_mosaic_restorer.py`
- Architecture: `BasicVSRPlusPlusGanNet`
- Mid channels: `64`
- Residual blocks: `15`
- Training crop size used by the config: `256`
- Typical training clip length used by the config: `16`

## Proposed Core ML feature contract

Expected compiled bundle name:

```text
LadaMosaicRestorer.mlmodelc
```

Input:

```text
frames
```

Shape:

```text
1 × 16 × 3 × 256 × 256
```

Layout:

```text
B × T × C × H × W
```

Color order:

```text
BGR
```

Value range:

```text
float32, normalized 0...1
```

Output:

```text
restored_frames
```

Output shape:

```text
1 × 16 × 3 × 256 × 256
```

The output should preserve clip length and spatial size. Swift then converts the
output back into BGRA/BGR crop pixels, resizes the crop back to the detected
source mosaic region, and composites it with the native Metal mask pipeline.

## Why sequence input matters

BasicVSR++ is not an image-to-image model. It estimates temporal features and
optical flow across adjacent frames. Running it one frame at a time would remove
the main reason this model exists and would likely reduce quality. The native
contract therefore uses `NativeTemporalRegionRestorer` rather than only the
older single-frame `NativeRegionRestorer`.

## Probe result

Short Core ML probe command:

```text
vendor/lada/.venv/bin/python tools/native-models/export_basicvsrpp_to_coreml.py --format coreml --clip-length 2 --spatial-size 256 --output /private/tmp/LadaMosaicRestorer-probe.mlmodel --patch-coremltools-scalar-casts --run-export
```

Result: conversion reaches the Torch graph after applying the same
`coremltools` scalar/list workaround used by the YOLO detector, then fails at:

```text
PyTorch convert function for op 'upsample_bicubic2d' not implemented.
```

Follow-up Core ML probe command:

```text
vendor/lada/.venv/bin/python tools/native-models/export_basicvsrpp_to_coreml.py --format coreml --clip-length 2 --spatial-size 256 --output /private/tmp/LadaMosaicRestorer-nearest-probe.mlmodel --patch-coremltools-scalar-casts --replace-bicubic-downsample nearest --run-export
```

Result: replacing the internal bicubic downsample with nearest-neighbor during
export tracing gets past the resize blocker. A narrow `new_zeros` converter
patch then gets past Core ML's scalar shape concat issue. The next blocker is
the expected hard one:

```text
PyTorch convert function for op 'torchvision::deform_conv2d' not implemented.
```

Short ONNX probe command:

```text
vendor/lada/.venv/bin/python tools/native-models/export_basicvsrpp_to_coreml.py --format onnx --clip-length 2 --spatial-size 256 --onnx-output /private/tmp/LadaMosaicRestorer-probe.onnx --run-export
```

Result: the current Torch ONNX exporter requires the optional `onnxscript`
package before it can reach the graph.

## Known conversion risk

The confirmed Core ML blockers are:

1. `upsample_bicubic2d`
2. scalar shape handling around `new_zeros`
3. `torchvision::deform_conv2d`

The export helper can probe around the first two, but deformable convolution is
the meaningful architecture blocker. The likely paths are:

1. replace the deformable-convolution alignment block with a Core ML-friendly
   approximation and retrain/fine-tune;
2. split restoration into custom Metal kernels plus smaller Core ML blocks;
3. keep restoration in PyTorch while only detector/video IO move native.

## Simplified-alignment probe

Probe command:

```text
vendor/lada/.venv/bin/python tools/native-models/export_basicvsrpp_to_coreml.py --format coreml --clip-length 2 --spatial-size 256 --output /private/tmp/LadaMosaicRestorer-standardconv-probe.mlpackage --patch-coremltools-scalar-casts --replace-bicubic-downsample nearest --replace-deform-conv standard-conv --run-export
```

Result: succeeds and saves:

```text
/private/tmp/LadaMosaicRestorer-standardconv-probe.mlpackage
```

The package also compiles with Xcode's Core ML compiler to:

```text
/private/tmp/LadaMosaicRestorer-standardconv-probe.mlmodelc
```

This model is a probe, not a quality candidate: it replaces deformable
convolution with ordinary convolution and ignores offsets/masks. The useful
finding is architectural: once deformable alignment is removed or replaced, the
rest of the 2-frame BasicVSR++-style graph can convert and compile as Core ML.

Inspection command:

```text
vendor/lada/.venv/bin/python tools/native-models/inspect_coreml_restorer.py /private/tmp/LadaMosaicRestorer-standardconv-probe.mlpackage --run-zero-prediction
```

Result: the probe declares `frames` input shape `1×2×3×256×256`, declares
`restored_frames` output shape `1×2×3×256×256`, and runs a zero-input
prediction successfully. The package spec stores the output as FLOAT16, while
Core ML prediction returns a float32 NumPy array.

Comparison command:

```text
vendor/lada/.venv/bin/python tools/native-models/compare_restorer_probe.py /private/tmp/LadaMosaicRestorer-standardconv-probe.mlpackage --clip-length 2 --spatial-size 256
```

Result against Python BasicVSR++ on the same deterministic synthetic clip:

```text
mean absolute error: 10.79
max absolute error: 185
RMSE: 18.38
PSNR: 22.84 dB
```

This is too much drift to treat the standard-conv probe as production quality.
It is still useful because it quantifies the quality cost of simply dropping
deformable alignment.

Variant matrix:

- nearest downsample + standard convolution: MAE `10.79`, PSNR `22.84 dB`
- bilinear downsample + standard convolution: MAE `10.79`, PSNR `22.84 dB`
- bilinear downsample + mask-modulated standard convolution: MAE `1.08`, PSNR
  `41.28 dB`

On this synthetic probe, nearest and bilinear replacements produce identical
metrics for plain standard convolution. Mask-modulated standard convolution is
dramatically closer to Python BasicVSR++, suggesting that the learned
deformable-conv mask is useful even if offsets are ignored. This is still not a
validated production model, but it is the strongest Core ML-friendly alignment
candidate so far.

Smoke-video comparison on `work/smoke-input.mp4` is stronger:

- bilinear + standard convolution: MAE `2.29`, PSNR `35.02 dB`
- bilinear + mask-modulated standard convolution: MAE `0.38`, PSNR `51.86 dB`

This keeps the masked variant as the best current native-restorer candidate.

## Validation criteria before enabling by default

- The compiled model loads from app Resources as `LadaMosaicRestorer.mlmodelc`.
- Output shape exactly matches input shape.
- A fixed reference clip matches Python BasicVSR++ within an agreed pixel-error
  tolerance.
- Several real clips pass visual inspection for temporal shimmer/flicker.
- Memory use stays bounded on Apple Silicon during long exports.
- The stable default remains Python/PyTorch until the native restorer passes the
  reference suite.
