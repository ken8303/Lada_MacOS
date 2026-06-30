#!/usr/bin/env python3
"""Prepare or run YOLO mosaic detector export to a Core AI .aimodel asset.

Default mode is a dry-run so this script is safe on machines that have the
macOS 27 Core AI runtime but do not yet have Apple's Python converter package.

The intended final asset is:

    native-models/LadaMosaicDetector.aimodel

That asset will be copied into the app bundle by packaging/macos/package.sh and
inspected by NativeCoreAICapabilities at runtime.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "vendor/lada/model_weights/lada_mosaic_detection_model_v4_fast.pt"
DEFAULT_OUTPUT = ROOT / "native-models/LadaMosaicDetector.aimodel"
COREAI_MODULE_CANDIDATES = (
    "coreai.torch",
    "coreai_torch",
    "coreaitorch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to YOLO .pt detector weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .aimodel path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Detector export image size. Keep this aligned with Swift preprocessing.",
    )
    parser.add_argument(
        "--input-name",
        default="images",
        help="Core AI input name to request from the converter.",
    )
    parser.add_argument(
        "--output-names",
        default="output0,output1",
        help="Comma-separated Core AI output names to request from the converter.",
    )
    parser.add_argument(
        "--run-export",
        action="store_true",
        help="Actually run export instead of printing a dependency/conversion plan.",
    )
    return parser.parse_args()


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def locate_coreai_torch() -> dict[str, Any]:
    """Return converter module information without requiring it to be installed."""

    for module_name in COREAI_MODULE_CANDIDATES:
        if not module_available(module_name):
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # pragma: no cover - optional beta package
            return {
                "available": False,
                "module": module_name,
                "error": f"import failed: {error}",
            }

        converter = getattr(module, "TorchConverter", None)
        if converter is None and hasattr(module, "converter"):
            converter = getattr(module.converter, "TorchConverter", None)
        if converter is None:
            return {
                "available": True,
                "module": module_name,
                "converter": None,
                "error": "module found, but TorchConverter was not exposed",
            }

        return {
            "available": True,
            "module": module_name,
            "converter": "TorchConverter",
            "signature": str(inspect.signature(converter)),
        }

    return {
        "available": False,
        "module": None,
        "converter": None,
        "error": "Core AI Python converter package not installed",
    }


def dry_run(args: argparse.Namespace) -> int:
    converter = locate_coreai_torch()
    plan = {
        "weights": str(args.weights),
        "weights_exists": args.weights.exists(),
        "output": str(args.output),
        "imgsz": args.imgsz,
        "input_name": args.input_name,
        "output_names": output_names(args),
        "python": sys.executable,
        "dependencies": {
            "torch": module_available("torch"),
            "ultralytics": module_available("ultralytics"),
            "coreai_torch_candidates": {
                name: module_available(name) for name in COREAI_MODULE_CANDIDATES
            },
        },
        "coreai_converter": converter,
        "next_command": (
            f"PYTHONPATH=vendor/lada {sys.executable} {Path(__file__).as_posix()} "
            f"--weights {args.weights} --output {args.output} --imgsz {args.imgsz} --run-export"
        ),
    }
    print(json.dumps(plan, indent=2))
    return 0 if args.weights.exists() else 2


def output_names(args: argparse.Namespace) -> list[str]:
    return [
        name.strip()
        for name in args.output_names.split(",")
        if name.strip()
    ]


def export_program(args: argparse.Namespace):
    import torch
    from ultralytics import YOLO

    yolo = YOLO(str(args.weights))
    model = yolo.model.eval()
    example = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)

    with torch.no_grad():
        return torch.export.export(model, (example,))


def run_export(args: argparse.Namespace) -> int:
    if not args.weights.exists():
        print(f"Missing weights: {args.weights}", file=sys.stderr)
        return 2

    missing = [
        name
        for name in ("torch", "ultralytics")
        if not module_available(name)
    ]
    if missing:
        print(f"Missing Python dependencies: {', '.join(missing)}", file=sys.stderr)
        return 3

    converter_info = locate_coreai_torch()
    if not converter_info.get("available") or converter_info.get("converter") != "TorchConverter":
        print(
            "Core AI converter unavailable. Install Apple's Core AI Torch converter "
            "package for this macOS 27 beta, then rerun this command.",
            file=sys.stderr,
        )
        print(json.dumps(converter_info, indent=2), file=sys.stderr)
        return 4

    module = importlib.import_module(str(converter_info["module"]))
    converter_type = getattr(module, "TorchConverter", None)
    if converter_type is None and hasattr(module, "converter"):
        converter_type = getattr(module.converter, "TorchConverter", None)
    if converter_type is None:
        print(json.dumps(converter_info, indent=2), file=sys.stderr)
        return 4

    exported_program = export_program(args)
    output = args.output.expanduser().resolve()
    temporary_output = output.with_suffix(output.suffix + ".tmp")

    if temporary_output.exists():
        if temporary_output.is_dir():
            shutil.rmtree(temporary_output)
        else:
            temporary_output.unlink()

    try:
        converter = converter_type(
            exported_program,
            input_names=[args.input_name],
            output_names=output_names(args),
        )
        save_result = converter.save(str(temporary_output))
    except TypeError as error:
        print(
            "Core AI TorchConverter API did not match the documented constructor "
            "shape used by this helper. Converter signature:",
            file=sys.stderr,
        )
        print(converter_info.get("signature"), file=sys.stderr)
        print(f"Error: {error}", file=sys.stderr)
        return 5

    if save_result is not None:
        print(f"Core AI converter returned: {save_result}")

    if not temporary_output.exists():
        print(f"Core AI export did not create expected output: {temporary_output}", file=sys.stderr)
        return 6

    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    temporary_output.rename(output)
    print(f"Exported Core AI detector asset to {output}")
    return 0


def main() -> int:
    args = parse_args()
    args.weights = args.weights.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.run_export:
        return run_export(args)
    return dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
