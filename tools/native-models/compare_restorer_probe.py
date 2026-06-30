#!/usr/bin/env python3
"""Compare a Core ML restorer probe against Python BasicVSR++ on one tiny clip."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "vendor/lada/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coreml_model",
        type=Path,
        help="Path to a Core ML restorer .mlpackage.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to Python BasicVSR++ restoration weights.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=2,
        help="Clip length to compare. Keep this aligned with the probe model.",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=256,
        help="Square clip size to compare.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic synthetic input seed.",
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help="Optional video source. When provided, compare on decoded video frames instead of synthetic frames.",
    )
    return parser.parse_args()


def make_synthetic_bgr_clip(
    clip_length: int,
    spatial_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:spatial_size, 0:spatial_size]
    frames = []
    for frame_index in range(clip_length):
        base = np.stack(
            [
                (x + frame_index * 7) % 256,
                (y * 2 + frame_index * 11) % 256,
                ((x + y) // 2 + frame_index * 13) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        noise = rng.integers(0, 16, size=base.shape, dtype=np.uint8)
        frames.append(np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8))
    return np.stack(frames, axis=0)


def read_video_bgr_clip(
    input_video: Path,
    clip_length: int,
    spatial_size: int,
) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    frames = []
    try:
        while len(frames) < clip_length:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(
                frame,
                (spatial_size, spatial_size),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(frame)
    finally:
        capture.release()

    if len(frames) != clip_length:
        raise RuntimeError(
            f"Expected {clip_length} frames from {input_video}, got {len(frames)}"
        )
    return np.stack(frames, axis=0)


def run_python_basicvsr(
    clip_bgr_uint8: np.ndarray,
    weights: Path,
) -> np.ndarray:
    import torch
    from lada.models.basicvsrpp.inference import load_model
    from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer

    model = load_model(None, str(weights), torch.device("cpu"), fp16=False)
    restorer = BasicvsrppMosaicRestorer(model, torch.device("cpu"), fp16=False)
    frames = [
        torch.from_numpy(frame.copy())
        for frame in clip_bgr_uint8
    ]
    restored = restorer.restore(frames)
    return np.stack([frame.cpu().numpy() for frame in restored], axis=0)


def run_coreml_restorer(
    clip_bgr_uint8: np.ndarray,
    model_path: Path,
) -> np.ndarray:
    import coremltools as ct

    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    frames = clip_bgr_uint8.astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2))[np.newaxis, ...]
    prediction = model.predict({"frames": frames})
    output = prediction["restored_frames"]
    output = np.asarray(output, dtype=np.float32)
    output = np.clip(np.rint(output[0] * 255.0), 0, 255).astype(np.uint8)
    return np.transpose(output, (0, 2, 3, 1))


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    diff = candidate.astype(np.float32) - reference.astype(np.float32)
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    psnr = math.inf if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "mean_absolute_error": float(np.mean(abs_diff)),
        "max_absolute_error": float(np.max(abs_diff)),
        "root_mean_square_error": math.sqrt(mse),
        "psnr_db": psnr,
    }


def main() -> int:
    args = parse_args()
    if not args.coreml_model.exists():
        print(json.dumps({"coreml_model": str(args.coreml_model), "exists": False}, indent=2))
        return 2
    if not args.weights.exists():
        print(json.dumps({"weights": str(args.weights), "exists": False}, indent=2))
        return 2

    if args.input_video:
        input_clip = read_video_bgr_clip(
            input_video=args.input_video,
            clip_length=args.clip_length,
            spatial_size=args.spatial_size,
        )
        source: dict[str, object] = {
            "kind": "video",
            "path": str(args.input_video),
        }
    else:
        input_clip = make_synthetic_bgr_clip(
            clip_length=args.clip_length,
            spatial_size=args.spatial_size,
            seed=args.seed,
        )
        source = {
            "kind": "synthetic",
            "seed": args.seed,
        }
    reference = run_python_basicvsr(input_clip, args.weights)
    candidate = run_coreml_restorer(input_clip, args.coreml_model)
    report = {
        "coreml_model": str(args.coreml_model),
        "python_weights": str(args.weights),
        "clip_length": args.clip_length,
        "spatial_size": args.spatial_size,
        "source": source,
        "comparison": compare(reference, candidate),
        "interpretation": (
            "This compares a probe-only Core ML-friendly alignment model against "
            "the real Python BasicVSR++ model. Large error is expected; the "
            "goal is to quantify drift and keep future variants honest."
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
