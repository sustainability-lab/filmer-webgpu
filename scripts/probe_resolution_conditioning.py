#!/usr/bin/env python3
"""Probe FiLMeR's out-of-distribution resolution conditioning.

This is a sensitivity experiment, not resolution validation. Every run uses
the same d01 GFS/static input, geographic bounds, and 27 km WRF reference. Only
projection-vector element 12 is changed; the network output remains 99x99.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from build_validation_fixture import channel_metrics
from filmer_model import FiLMeRVariantB
from preprocessing import METADATA, normalized_projection, reconstruct_physical


def load_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} values; expected {expected}")
    return values.reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "public/data/validation/heldout-d01-20250101T0600-input.f32"
        ),
    )
    parser.add_argument(
        "--wrf-target",
        type=Path,
        default=Path(
            "public/data/validation/heldout-d01-20250101T0600-wrf-target.f32"
        ),
    )
    parser.add_argument(
        "--resolutions", type=float, nargs="+", default=[1, 3, 9, 27, 54]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/resolution-conditioning-probe.json"),
    )
    args = parser.parse_args()

    input_data = torch.from_numpy(
        load_f32(args.input, (1, 70, 127, 137)).copy()
    )
    target = load_f32(args.wrf_target, (6, 99, 99))
    model = FiLMeRVariantB().eval()
    model.load_state_dict(
        torch.load(args.state_dict, map_location="cpu", weights_only=True),
        strict=True,
    )
    valid_time = datetime(2025, 1, 1, 6)
    resolution_mean = METADATA["projection"]["mean"][12]
    resolution_std = METADATA["projection"]["std"][12]
    predictions: dict[float, np.ndarray] = {}
    cases: list[dict[str, object]] = []
    with torch.no_grad():
        for resolution in args.resolutions:
            projection = normalized_projection(1, valid_time)
            projection[12] = (resolution - resolution_mean) / resolution_std
            state, occurrence, intensity = model(
                input_data, projection.unsqueeze(0)
            )
            prediction = reconstruct_physical(
                state.numpy(), occurrence.numpy(), intensity.numpy()
            )[0]
            predictions[resolution] = prediction
            cases.append(
                {
                    "conditioningResolutionKm": resolution,
                    "normalizedResolutionCoordinate": float(projection[12]),
                    "outputShape": [6, 99, 99],
                    "metricsAgainst27KmWrfTarget": channel_metrics(
                        prediction, target
                    ),
                }
            )

    baseline = predictions[27.0]
    for case in cases:
        resolution = float(case["conditioningResolutionKm"])
        delta = predictions[resolution].astype(np.float64) - baseline.astype(
            np.float64
        )
        case["differenceFrom27KmPrediction"] = {
            "maeAllValues": float(np.abs(delta).mean()),
            "rmseAllValues": float(np.sqrt(np.square(delta).mean())),
            "maxAbsoluteAllValues": float(np.abs(delta).max()),
        }

    report = {
        "kind": "out-of-distribution-resolution-conditioning-sensitivity",
        "validatedResolutionKm": 27,
        "fixedInputShape": [1, 70, 127, 137],
        "fixedOutputShape": [6, 99, 99],
        "fixedDomain": "d01",
        "warning": (
            "Only the projection resolution scalar changes. Non-27 km cases "
            "are not forecasts at those resolutions and are compared with a "
            "27 km WRF target solely to expose model sensitivity."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
