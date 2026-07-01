# Clip-length / cache-interval A/B validation

This validates the Long Video tuning from commit `f004d84`:

- old settings: `maxClipLength=45`, `LADA_MPS_EMPTY_CACHE_INTERVAL=3`
- new settings: `maxClipLength=75`, `LADA_MPS_EMPTY_CACHE_INTERVAL=8`

Both variants keep `LADA_SERIALIZE_MPS=1`, because real HUNTC-619 testing already showed that disabling MPS serialization can crash PyTorch MPS.

## 1. Cut a dense test clip

Use the known dense HUNTC-619 segment around frames `11150–12500`:

```sh
tools/cut_test_clip.sh \
  --input /path/to/original_source_video.mp4 \
  --start-frame 11150 \
  --end-frame 12500 \
  --output /tmp/lada_dense_test_clip.mp4
```

If `ffprobe` cannot read the frame rate automatically, pass `--fps 29.97`.

## 2. Run old vs new settings

```sh
tools/run_clip_length_ab_test.sh \
  --input /tmp/lada_dense_test_clip.mp4 \
  --output-dir /tmp/lada_clip_length_ab
```

The script auto-detects the best worker/runtime in this order:

1. `outputs/Lada.app/Contents/Resources/lada_worker.py` with bundled app Python
2. `Sources/LadaMac/Resources/lada_worker.py` with `vendor/lada/.venv/bin/python3`
3. `Sources/LadaMac/Resources/lada_worker.py` with system `python3`

You can override either path:

```sh
tools/run_clip_length_ab_test.sh \
  --input /tmp/lada_dense_test_clip.mp4 \
  --worker-script Sources/LadaMac/Resources/lada_worker.py \
  --python vendor/lada/.venv/bin/python3
```

## 3. Read the result

The run creates:

- `old_settings.raw.jsonl`
- `new_settings.raw.jsonl`
- `old_settings.mp4`
- `new_settings.mp4`

It then runs:

```sh
tools/compare_clip_length_logs.py \
  /tmp/lada_clip_length_ab/old_settings.raw.jsonl \
  /tmp/lada_clip_length_ab/new_settings.raw.jsonl
```

The comparison reports detection speed, restore speed, clip-create count, max observed clip length, and crash/memory signals.

Use the result like this:

- New settings crash or show MPS/OOM signals: treat `75` as risky and test a midpoint such as `60`.
- New settings improve throughput by more than about 10% without crash: keep the larger Long Video defaults.
- New settings do not improve much but also do not crash: clip length is probably not the main remaining bottleneck; the larger two-pass detection/restoration restructure becomes the next lever.
