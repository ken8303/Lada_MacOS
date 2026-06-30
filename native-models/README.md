# Native Core ML model drop-in folder

Place compiled Core ML model bundles and Core AI model assets here before
packaging.

Expected names:

- `LadaMosaicDetector.mlmodelc`
- `LadaMosaicRestorer.mlmodelc`
- `LadaMosaicDetector.aimodel`
- `LadaMosaicRestorer.aimodel`

The packaging script copies any `*.mlmodelc` directory or `*.aimodel` asset
from this folder into `Lada.app/Contents/Resources/` and writes a native model
manifest into the build notes.

The app currently has a native detector bridge for `LadaMosaicDetector.mlmodelc`:
it builds letterboxed Core ML inputs, caches the loaded model, reads either
`output0`/`output1` or the exported Ultralytics `var_1324`/`var_1362` outputs,
and maps YOLO-style detections back to source-frame coordinates.

`LadaMosaicDetector.mlmodelc` is now present and passes the smoke-frame
calibration test against the Python reference fixture. The remaining detector
work is expanding calibration to real validation clips before making the native
detector the default production path.

`LadaMosaicRestorer.mlmodelc` is not present yet. The app has a native restorer
loader/fallback scaffold and a temporal clip-level restorer seam. The current
native processed-frame path still uses a safe placeholder, but the future real
BasicVSR++ replacement should plug into `NativeTemporalRegionRestorer` so it can
consume and return a sequence of frames instead of treating each frame as an
unrelated still image.

See `RESTORER-COREML-CONTRACT.md` for the proposed `frames` /
`restored_frames` `BTCHW` Core ML contract and the known conversion risk around
BasicVSR++ deformable convolution.

Useful restorer tooling:

- `tools/native-models/export_yolo_to_coreai.py` — dry-run or run the future
  YOLO detector export to `LadaMosaicDetector.aimodel` once Apple's Core AI
  Torch converter package is installed for macOS 27.
- `tools/native-models/export_basicvsrpp_to_coreml.py` — dry-run or probe
  BasicVSR++ Core ML export.
- `tools/native-models/inspect_coreml_restorer.py` — inspect a restorer
  `.mlpackage` and optionally run a zero-input prediction smoke test.
- `tools/native-models/compare_restorer_probe.py` — compare a Core ML restorer
  probe against the real Python BasicVSR++ output on the same deterministic
  synthetic clip.
