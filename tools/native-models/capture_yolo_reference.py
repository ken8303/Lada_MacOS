#!/usr/bin/env python3
"""Capture Python/Ultralytics YOLO detector reference outputs for native validation.

The native Core ML detector should not become the production path until it can
match the current Python detector on fixed frames. This tool writes a compact
JSON fixture that records the Python detector settings and per-frame detections.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "vendor/lada/model_weights/lada_mosaic_detection_model_v4_fast.pt"
DEFAULT_OUTPUT = ROOT / "native-models/reference-detections/lada-yolo-reference.json"


@dataclass
class DetectionReference:
    cls: int
    confidence: float
    xyxy: list[float]
    mask_shape: list[int] | None
    mask_area: int | None


@dataclass
class FrameReference:
    frame_index: int
    width: int
    height: int
    detections: list[DetectionReference]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image or video path.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="YOLO detector .pt weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON fixture output path.",
    )
    parser.add_argument(
        "--frames",
        default="0",
        help="Comma-separated video frame indices to capture, for example 0,30,60.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for reference capture. Use cpu for deterministic fixtures.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO image size. Keep aligned with Swift native preprocessing.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold. Lada restoration pipeline currently uses 0.15.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--detect-face-mosaics",
        action="store_true",
        help="Match Lada's class filter for face mosaic mode by keeping class 0 only.",
    )
    return parser.parse_args()


def parse_frame_indices(value: str) -> list[int]:
    indices = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        index = int(part)
        if index < 0:
            raise ValueError("Frame indices must be non-negative")
        indices.append(index)
    return indices or [0]


def read_frames(path: Path, frame_indices: list[int]) -> list[tuple[int, Any]]:
    import cv2

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        return [(0, image)]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames: list[tuple[int, Any]] = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {frame_index} from {path}")
            frames.append((frame_index, frame))
    finally:
        capture.release()
    return frames


def detection_references(result: Any) -> list[DetectionReference]:
    detections: list[DetectionReference] = []
    boxes = result.boxes
    masks = result.masks
    for index, box in enumerate(boxes):
        xyxy = [float(value) for value in box.xyxy[0].tolist()]
        confidence = float(box.conf[0].item())
        cls = int(box.cls[0].item())
        mask_shape: list[int] | None = None
        mask_area: int | None = None
        if masks is not None and index < len(masks):
            mask_tensor = masks[index].data
            mask_shape = [int(value) for value in mask_tensor.shape]
            mask_area = int((mask_tensor > 0).sum().item())
        detections.append(
            DetectionReference(
                cls=cls,
                confidence=confidence,
                xyxy=xyxy,
                mask_shape=mask_shape,
                mask_area=mask_area,
            )
        )
    return detections


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(ROOT / "vendor/lada"))
    from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

    frame_indices = parse_frame_indices(args.frames)
    frames = read_frames(args.input, frame_indices)
    classes = [0] if args.detect_face_mosaics else None
    model = Yolo11SegmentationModel(
        str(args.weights),
        args.device,
        imgsz=args.imgsz,
        fp16=False,
        conf=args.conf,
        iou=args.iou,
        classes=classes,
    )

    frame_references: list[FrameReference] = []
    with torch.inference_mode():
        for frame_index, frame in frames:
            frame_tensor = torch.from_numpy(frame)
            batch = model.preprocess([frame_tensor])
            results = model.inference_and_postprocess(batch, [frame_tensor])
            height, width = frame.shape[:2]
            frame_references.append(
                FrameReference(
                    frame_index=frame_index,
                    width=width,
                    height=height,
                    detections=detection_references(results[0]),
                )
            )

    return {
        "schema_version": 1,
        "source": str(args.input),
        "weights": str(args.weights),
        "device": args.device,
        "imgsz": args.imgsz,
        "stride": model.stride,
        "conf": args.conf,
        "iou": args.iou,
        "classes": classes,
        "letterbox": {
            "kind": "ultralytics",
            "auto": True,
            "stride": model.stride,
        },
        "frames": [asdict(frame_reference) for frame_reference in frame_references],
    }


def main() -> int:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.input.exists():
        print(f"Missing input: {args.input}", file=sys.stderr)
        return 2
    if not args.weights.exists():
        print(f"Missing weights: {args.weights}", file=sys.stderr)
        return 2

    fixture = capture(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    total = sum(len(frame["detections"]) for frame in fixture["frames"])
    print(
        f"Wrote {len(fixture['frames'])} frame reference(s), "
        f"{total} detection(s): {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
