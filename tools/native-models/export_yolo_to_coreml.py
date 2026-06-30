#!/usr/bin/env python3
"""Prepare or run YOLO mosaic detector export to Core ML or ONNX.

Default mode is a dry-run so this script is safe to execute before optional
conversion packages are installed. Use --run-export when the Python environment
has ultralytics and coremltools available.
"""

from __future__ import annotations

import argparse
import json
import numbers
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "vendor/lada/model_weights/lada_mosaic_detection_model_v4_fast.pt"
DEFAULT_COREML_OUTPUT = ROOT / "native-models/LadaMosaicDetector.mlmodelc"
DEFAULT_ONNX_OUTPUT = ROOT / "native-models/LadaMosaicDetector.onnx"


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
        default=None,
        help="Expected output path. Defaults to native-models/LadaMosaicDetector.mlmodelc for Core ML or .onnx for ONNX.",
    )
    parser.add_argument(
        "--format",
        choices=("coreml", "onnx"),
        default="coreml",
        help="Export format. ONNX is useful as an intermediate when Core ML conversion fails.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO export image size. Keep this aligned with Swift preprocessing.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset to request. TorchScript exporter currently supports up to opset 20.",
    )
    parser.add_argument(
        "--run-export",
        action="store_true",
        help="Actually run Ultralytics Core ML export instead of printing a plan.",
    )
    parser.add_argument(
        "--patch-coremltools-scalar-casts",
        action="store_true",
        help=(
            "Apply a narrow coremltools Torch frontend workaround for length-1 "
            "constant tensors passed to int/float/bool casts. This is useful "
            "with newer Torch graphs that coremltools 9.0 does not officially support."
        ),
    )
    return parser.parse_args()


def dry_run(args: argparse.Namespace) -> int:
    plan = {
        "weights": str(args.weights),
        "weights_exists": args.weights.exists(),
        "output": str(args.output),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "format": args.format,
        "expected_packaged_name": "LadaMosaicDetector.mlmodelc" if args.format == "coreml" else "LadaMosaicDetector.onnx",
        "next_command": (
            f"{sys.executable} {Path(__file__).as_posix()} "
            f"--weights {args.weights} --output {args.output} --imgsz {args.imgsz} --run-export"
        ),
        "coremltools_scalar_cast_patch": args.patch_coremltools_scalar_casts,
    }
    print(json.dumps(plan, indent=2))
    return 0 if args.weights.exists() else 2


def patch_coremltools_scalar_casts() -> None:
    """Patch coremltools 9.0 for Torch length-1 constant tensor casts.

    coremltools' Torch frontend accepts scalar or length-1 tensor inputs for
    cast ops, but its constant-folding branch calls `int(x.val)` directly. With
    recent Torch graphs, `x.val` can be a length-1 ndarray, which raises:

        TypeError: only 0-dimensional arrays can be converted to Python scalars

    This patch preserves the existing shape checks and only unwraps constants
    whose single value can be safely represented as a Python scalar.
    """

    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil.mil.var import ListVar, Var
    from coremltools.converters.mil.frontend.torch import ops

    def scalar_value(value):
        if isinstance(value, numbers.Number):
            return value
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError("input to cast must be a scalar or length 1 tensor")
        return array.reshape(()).item()

    def patched_cast(context, node, dtype, dtype_name):
        inputs = ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")

        if x.can_be_folded_to_const():
            value = scalar_value(x.val)
            if not isinstance(value, dtype):
                res = mb.const(val=dtype(value), name=node.name)
            else:
                res = x
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    def patched_int(context, node):
        patched_cast(context, node, int, "int32")

    def patched_float(context, node):
        patched_cast(context, node, float, "fp32")

    def patched_bool(context, node):
        patched_cast(context, node, bool, "bool")

    def patched_view(context, node):
        inputs = ops._get_inputs(context, node, expected=2)
        x = inputs[0]
        shape = inputs[1]

        if isinstance(shape, Var) and np.prod(shape.shape) == 0:
            assert np.prod(x.shape) <= 1, (
                "Reshape to empty shape works only for scalar and single-element tensor"
            )
            context.add(mb.identity(x=x, name=node.name))
            return

        if isinstance(shape, ListVar):
            length = mb.list_length(ls=shape)
            indices = mb.range_1d(start=0, end=length, step=1)
            shape = mb.list_gather(ls=shape, indices=indices)

        if isinstance(shape, list) and all(
            isinstance(dim, Var) and len(dim.shape) == 0 for dim in shape
        ):
            int_shape = []
            for index, size in enumerate(shape):
                int_size = size if size.dtype == types.int32 else mb.cast(x=size, dtype="int32")
                int_shape.append(
                    mb.expand_dims(
                        x=int_size,
                        axes=[0],
                        name=f"{node.name}_shape_dim_{index}",
                    )
                )
            shape = mb.concat(values=int_shape, axis=0, name=node.name + "_shape")
        elif isinstance(shape, list) and all(
            isinstance(dim, Var) for dim in shape
        ):
            int_shape = []
            for index, size in enumerate(shape):
                int_size = size if size.dtype == types.int32 else mb.cast(x=size, dtype="int32")
                if len(int_size.shape) == 0:
                    int_size = mb.expand_dims(
                        x=int_size,
                        axes=[0],
                        name=f"{node.name}_shape_dim_{index}",
                    )
                int_shape.append(int_size)
            shape = mb.concat(values=int_shape, axis=0, name=node.name + "_shape")

        shape = mb.cast(x=shape, dtype="int32")

        if types.is_complex(x.dtype):
            real, imag = (
                mb.reshape(x=value, shape=shape, name=node.name)
                for value in (mb.complex_real(data=x), mb.complex_imag(data=x))
            )
            view = mb.complex(real_data=real, imag_data=imag, name=node.name)
        else:
            view = mb.reshape(x=x, shape=shape, name=node.name)
        context.add(view)

    ops._cast = patched_cast
    ops._int = patched_int
    ops._float = patched_float
    ops._bool = patched_bool
    ops.view = patched_view
    ops._TORCH_OPS_REGISTRY.register_func(patched_int, ["int"], override=True)
    ops._TORCH_OPS_REGISTRY.register_func(patched_float, ["float"], override=True)
    ops._TORCH_OPS_REGISTRY.register_func(patched_bool, ["bool"], override=True)
    ops._TORCH_OPS_REGISTRY.register_func(
        patched_view,
        ["view", "view_copy", "_unsafe_view", "reshape"],
        override=True,
    )


def run_export(args: argparse.Namespace) -> int:
    if not args.weights.exists():
        print(f"Missing weights: {args.weights}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except Exception as error:  # pragma: no cover - optional environment path
        print(f"Could not import ultralytics: {error}", file=sys.stderr)
        return 3

    try:
        if args.format == "coreml":
            import coremltools  # noqa: F401
        else:
            import onnx  # noqa: F401
    except Exception as error:  # pragma: no cover - optional environment path
        print(f"Could not import optional {args.format} export dependency: {error}", file=sys.stderr)
        return 4

    if args.format == "coreml" and args.patch_coremltools_scalar_casts:
        patch_coremltools_scalar_casts()

    model = YOLO(str(args.weights))
    exported = Path(
        model.export(
            format=args.format,
            imgsz=args.imgsz,
            opset=args.opset,
            nms=False,
        )
    )
    if not exported.exists():
        print(f"Ultralytics export did not create expected output: {exported}", file=sys.stderr)
        return 5

    if args.format == "coreml" and args.output.suffix == ".mlmodelc":
        from coremltools.models.utils import compile_model

        if args.output.exists():
            if args.output.is_dir():
                shutil.rmtree(args.output)
            else:
                args.output.unlink()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        compiled = Path(compile_model(str(exported), str(args.output)))
        if not compiled.exists():
            print(f"Core ML compile did not create expected output: {compiled}", file=sys.stderr)
            return 6
        if exported != args.output and exported.exists():
            if exported.is_dir():
                shutil.rmtree(exported)
            else:
                exported.unlink()
        print(f"Compiled exported Core ML model to {args.output}")
        return 0

    if args.output.exists():
        if args.output.is_dir():
            shutil.rmtree(args.output)
        else:
            args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if exported.is_dir():
        shutil.copytree(exported, args.output)
    else:
        shutil.copy2(exported, args.output)
    if exported != args.output and exported.exists():
        if exported.is_dir():
            shutil.rmtree(exported)
        else:
            exported.unlink()
    print(f"Copied exported {args.format} model to {args.output}")
    return 0


def main() -> int:
    args = parse_args()
    args.weights = args.weights.expanduser().resolve()
    if args.output is None:
        args.output = DEFAULT_COREML_OUTPUT if args.format == "coreml" else DEFAULT_ONNX_OUTPUT
    args.output = args.output.expanduser().resolve()
    if args.run_export:
        return run_export(args)
    return dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
