#!/usr/bin/env python3
"""Inspect and optionally smoke-run a Core ML restorer model package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA_TYPES = {
    65552: "FLOAT16",
    65568: "FLOAT32",
    131104: "INT32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="Path to a Core ML .mlpackage restorer model.",
    )
    parser.add_argument(
        "--run-zero-prediction",
        action="store_true",
        help="Run a zero-input prediction and report output shape/range.",
    )
    parser.add_argument(
        "--compute-units",
        choices=("cpu", "cpu-and-neural-engine", "all"),
        default="cpu",
        help="Core ML compute units to request for the optional prediction.",
    )
    return parser.parse_args()


def compute_units(name: str):
    import coremltools as ct

    return {
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu-and-neural-engine": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }[name]


def multi_array_description(feature) -> dict[str, object]:
    feature_type = feature.type.WhichOneof("Type")
    description: dict[str, object] = {
        "name": feature.name,
        "type": feature_type,
    }
    if feature_type == "multiArrayType":
        array_type = feature.type.multiArrayType
        description["shape"] = list(array_type.shape)
        description["data_type"] = DATA_TYPES.get(
            array_type.dataType,
            str(array_type.dataType),
        )
    return description


def inspect_model(args: argparse.Namespace) -> dict[str, object]:
    import coremltools as ct

    model = ct.models.MLModel(
        str(args.model),
        compute_units=compute_units(args.compute_units),
    )
    spec = model.get_spec()
    report: dict[str, object] = {
        "model": str(args.model),
        "exists": args.model.exists(),
        "inputs": [multi_array_description(item) for item in spec.description.input],
        "outputs": [multi_array_description(item) for item in spec.description.output],
    }

    if args.run_zero_prediction:
        import numpy as np

        inputs = {}
        for item in report["inputs"]:
            if item["type"] != "multiArrayType":
                continue
            dtype = np.float32 if item.get("data_type") != "FLOAT16" else np.float16
            inputs[item["name"]] = np.zeros(item["shape"], dtype=dtype)
        prediction = model.predict(inputs)
        report["zero_prediction"] = {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min()),
                "max": float(value.max()),
            }
            for name, value in prediction.items()
        }

    return report


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(json.dumps({"model": str(args.model), "exists": False}, indent=2))
        return 2
    print(json.dumps(inspect_model(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
