#!/usr/bin/env python3
"""Validate ONNX Runtime numerical parity against committed PyTorch outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from preprocessing import reconstruct_physical


def read_f32(path: Path, shape: list[int]) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(shape)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-6)
    return {
        "maxAbs": float(np.max(np.abs(difference))),
        "meanAbs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maxRelative": float(np.max(np.abs(difference) / denominator)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, default=Path("public/data"))
    parser.add_argument("--report", type=Path, default=Path("reports/parity.json"))
    parser.add_argument("--max-abs", type=float, default=1e-3)
    args = parser.parse_args()
    fixture = json.loads((args.fixture_dir / "fixture.json").read_text())
    input_data = read_f32(
        args.fixture_dir / fixture["inputs"]["input"]["file"],
        fixture["inputs"]["input"]["shape"],
    )
    projection = read_f32(
        args.fixture_dir / fixture["inputs"]["projection"]["file"],
        fixture["inputs"]["projection"]["shape"],
    )
    expected = {
        "state": read_f32(
            args.fixture_dir / fixture["outputs"]["state"]["file"],
            fixture["outputs"]["state"]["shape"],
        ),
        "precip_occurrence": read_f32(
            args.fixture_dir / fixture["outputs"]["occurrence"]["file"],
            fixture["outputs"]["occurrence"]["shape"],
        ),
        "precip_intensity": read_f32(
            args.fixture_dir / fixture["outputs"]["intensity"]["file"],
            fixture["outputs"]["intensity"]["shape"],
        ),
    }
    session = ort.InferenceSession(
        str(args.model),
        providers=["CPUExecutionProvider"],
        sess_options=ort.SessionOptions(),
    )
    output_values = session.run(
        None, {"input_data": input_data, "projection": projection}
    )
    actual = dict(zip([item.name for item in session.get_outputs()], output_values))
    actual_physical = reconstruct_physical(
        actual["state"],
        actual["precip_occurrence"],
        actual["precip_intensity"],
    )
    expected_physical = read_f32(
        args.fixture_dir / fixture["outputs"]["physical"]["file"],
        fixture["outputs"]["physical"]["shape"],
    )
    report = {
        "model": str(args.model),
        "provider": session.get_providers()[0],
        "outputs": {
            name: metrics(actual[name], expected[name]) for name in expected
        },
        "physicalUnits": metrics(actual_physical, expected_physical),
        "wetMaskDisagreements": int(
            np.count_nonzero(
                (actual["precip_occurrence"] > 0)
                != (expected["precip_occurrence"] > 0)
            )
        ),
    }
    report["maxAbsAcrossOutputs"] = max(
        result["maxAbs"] for result in report["outputs"].values()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["maxAbsAcrossOutputs"] > args.max_abs:
        raise SystemExit(
            f"Parity failed: {report['maxAbsAcrossOutputs']:.6g} > {args.max_abs}"
        )


if __name__ == "__main__":
    main()
