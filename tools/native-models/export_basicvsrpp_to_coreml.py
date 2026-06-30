#!/usr/bin/env python3
"""Prepare or run BasicVSR++ mosaic restorer export to Core ML.

Default mode is a dry-run. The real conversion is intentionally opt-in because
BasicVSR++ uses deformable convolution, which is a likely Core ML conversion
blocker and can take a while to fail.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "vendor/lada/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"
DEFAULT_OUTPUT = ROOT / "native-models/LadaMosaicRestorer.mlmodelc"
DEFAULT_ONNX_OUTPUT = ROOT / "native-models/LadaMosaicRestorer.onnx"
DEFAULT_CLIP_LENGTH = 16
DEFAULT_SPATIAL_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to BasicVSR++ .pth restoration weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .mlmodelc path.",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=DEFAULT_ONNX_OUTPUT,
        help="Optional ONNX intermediate output path.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=DEFAULT_CLIP_LENGTH,
        help="Fixed temporal length for the exported contract.",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=DEFAULT_SPATIAL_SIZE,
        help="Fixed square crop size for the exported contract.",
    )
    parser.add_argument(
        "--format",
        choices=("coreml", "onnx"),
        default="coreml",
        help="Export target. ONNX is useful to locate unsupported ops first.",
    )
    parser.add_argument(
        "--run-export",
        action="store_true",
        help="Actually attempt export instead of printing the plan.",
    )
    parser.add_argument(
        "--patch-coremltools-scalar-casts",
        action="store_true",
        help="Apply the same coremltools Torch scalar/list workaround used by the detector export helper.",
    )
    parser.add_argument(
        "--replace-bicubic-downsample",
        choices=("none", "nearest", "bilinear"),
        default="none",
        help=(
            "Experimental export-only replacement for BasicVSR++'s internal "
            "bicubic 0.25 downsample. This is for probing Core ML graph support "
            "and does not change the shipping Python model."
        ),
    )
    parser.add_argument(
        "--replace-deform-conv",
        choices=("none", "standard-conv", "masked-standard-conv"),
        default="none",
        help=(
            "Experimental export-only replacement for torchvision deformable "
            "convolution. standard-conv ignores learned offsets/masks; "
            "masked-standard-conv ignores offsets but modulates the ordinary "
            "convolution by the learned mask average. Both are probe-only."
        ),
    )
    return parser.parse_args()


def contract(args: argparse.Namespace) -> dict[str, object]:
    shape = [
        1,
        args.clip_length,
        3,
        args.spatial_size,
        args.spatial_size,
    ]
    return {
        "model_name": "LadaMosaicRestorer",
        "source_model": "basicvsrpp-v1.2",
        "weights": str(args.weights),
        "weights_exists": args.weights.exists(),
        "format": args.format,
        "output": str(args.output if args.format == "coreml" else args.onnx_output),
        "input": {
            "name": "frames",
            "shape": shape,
            "layout": "BTCHW",
            "color_order": "BGR",
            "dtype": "float32",
            "value_range": "0...1",
        },
        "output_contract": {
            "name": "restored_frames",
            "shape": shape,
            "layout": "BTCHW",
            "color_order": "BGR",
            "dtype": "float32",
            "value_range": "0...1",
        },
        "known_conversion_risk": [
            "coremltools 9.0 does not implement PyTorch upsample_bicubic2d in this graph unless replaced during export probing",
            "torchvision.ops.deform_conv2d is confirmed unsupported in BasicVSR++ second-order alignment",
            "torch.onnx export with this Torch build requires the optional onnxscript package",
        ],
        "coremltools_scalar_cast_patch": args.patch_coremltools_scalar_casts,
        "replace_bicubic_downsample": args.replace_bicubic_downsample,
        "replace_deform_conv": args.replace_deform_conv,
    }


def coreml_save_output_path(path: Path) -> Path:
    if path.suffix == ".mlpackage":
        return path
    if path.suffix == ".mlmodelc":
        return path.with_suffix(".mlpackage")
    if path.suffix == ".mlmodel":
        return path.with_suffix(".mlpackage")
    return path


def dry_run(args: argparse.Namespace) -> int:
    plan = contract(args)
    plan["next_command"] = (
        f"{sys.executable} {Path(__file__).as_posix()} "
        f"--weights {args.weights} --output {args.output} "
        f"--clip-length {args.clip_length} --spatial-size {args.spatial_size} "
        f"--format {args.format} "
        f"--replace-bicubic-downsample {args.replace_bicubic_downsample} "
        f"--replace-deform-conv {args.replace_deform_conv} --run-export"
    )
    print(json.dumps(plan, indent=2))
    return 0 if args.weights.exists() else 2


@contextmanager
def patched_bicubic_downsample(replacement: str):
    if replacement == "none":
        yield
        return

    import torch.nn.functional as F

    original_interpolate = F.interpolate

    def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, **kwargs):
        if mode == "bicubic" and scale_factor == 0.25:
            if replacement == "nearest":
                return original_interpolate(
                    input,
                    size=size,
                    scale_factor=scale_factor,
                    mode="nearest",
                    **kwargs,
                )
            return original_interpolate(
                input,
                size=size,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
                **kwargs,
            )
        return original_interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            **kwargs,
        )

    F.interpolate = interpolate
    try:
        yield
    finally:
        F.interpolate = original_interpolate


@contextmanager
def patched_deform_conv(replacement: str):
    if replacement == "none":
        yield
        return

    import torch.nn.functional as F
    import torchvision.ops

    original_deform_conv2d = torchvision.ops.deform_conv2d

    def deform_conv2d(
        input,
        offset,
        weight,
        bias=None,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        mask=None,
    ):
        output = F.conv2d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if replacement == "masked-standard-conv" and mask is not None:
            output = output * mask.mean(dim=1, keepdim=True)
        return output

    torchvision.ops.deform_conv2d = deform_conv2d
    try:
        yield
    finally:
        torchvision.ops.deform_conv2d = original_deform_conv2d


@contextmanager
def patched_export_model_ops(args: argparse.Namespace):
    with patched_bicubic_downsample(args.replace_bicubic_downsample):
        with patched_deform_conv(args.replace_deform_conv):
            yield


def load_generator(weights: Path, device: str):
    import torch
    from lada.models.basicvsrpp.inference import load_model
    from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan

    model = load_model(None, str(weights), torch.device(device), fp16=False)
    if not isinstance(model, BasicVSRPlusPlusGan):
        raise TypeError(f"Expected BasicVSRPlusPlusGan, got {type(model)!r}")
    return model.generator.eval()


def patch_coremltools_restorer_ops() -> None:
    from export_yolo_to_coreml import patch_coremltools_scalar_casts

    patch_coremltools_scalar_casts()

    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil.frontend.torch import ops

    def _shape_list_to_rank_one(shape, node_name: str):
        if not isinstance(shape, list):
            if len(shape.shape) == 0:
                return mb.expand_dims(x=shape, axes=[0], name=node_name + "_shape")
            return shape

        dims = []
        for index, dim in enumerate(shape):
            if dim.dtype != types.int32:
                dim = mb.cast(x=dim, dtype="int32")
            if len(dim.shape) == 0:
                dim = mb.expand_dims(
                    x=dim,
                    axes=[0],
                    name=f"{node_name}_shape_dim_{index}",
                )
            dims.append(dim)
        return mb.concat(values=dims, axis=0, name=node_name + "_shape")

    def patched_new_zeros(context, node):
        inputs = ops._get_inputs(context, node)
        shape = _shape_list_to_rank_one(inputs[1], node.name)
        context.add(mb.fill(shape=shape, value=0.0, name=node.name))

    ops.new_zeros = patched_new_zeros
    ops._TORCH_OPS_REGISTRY.register_func(
        patched_new_zeros,
        ["new_zeros", "new_empty"],
        override=True,
    )


def export_onnx(args: argparse.Namespace) -> int:
    import torch

    generator = load_generator(args.weights, "cpu")
    example = torch.zeros(
        1,
        args.clip_length,
        3,
        args.spatial_size,
        args.spatial_size,
        dtype=torch.float32,
    )
    args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
    with patched_export_model_ops(args):
        torch.onnx.export(
            generator,
            example,
            args.onnx_output,
            input_names=["frames"],
            output_names=["restored_frames"],
            opset_version=20,
            dynamic_axes=None,
        )
    print(json.dumps({**contract(args), "exported": str(args.onnx_output)}, indent=2))
    return 0


def export_coreml(args: argparse.Namespace) -> int:
    import coremltools as ct
    import torch

    if args.patch_coremltools_scalar_casts:
        patch_coremltools_restorer_ops()

    generator = load_generator(args.weights, "cpu")
    example = torch.zeros(
        1,
        args.clip_length,
        3,
        args.spatial_size,
        args.spatial_size,
        dtype=torch.float32,
    )
    with patched_export_model_ops(args):
        traced = torch.jit.trace(generator, example, strict=False)
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="frames",
                shape=example.shape,
                dtype=example.numpy().dtype,
            )
        ],
        outputs=[ct.TensorType(name="restored_frames")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS14,
    )
    output = coreml_save_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(json.dumps({**contract(args), "exported": str(output)}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if not args.run_export:
        return dry_run(args)
    if not args.weights.exists():
        print(f"Missing weights: {args.weights}", file=sys.stderr)
        return 2
    if args.format == "onnx":
        return export_onnx(args)
    return export_coreml(args)


if __name__ == "__main__":
    raise SystemExit(main())
