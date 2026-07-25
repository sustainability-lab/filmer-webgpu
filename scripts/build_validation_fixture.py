#!/usr/bin/env python3
"""Build a held-out FiLMeR-vs-WRF browser verification case.

The inputs and WRF target come from the GMD submission's sample_test split.
The committed browser fixture contains physical-unit arrays only; the original
PyTorch containers remain outside the public repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from filmer_model import FiLMeRVariantB
from preprocessing import normalize_gfs, normalized_projection, reconstruct_physical

VARIABLES = ["T2", "U10", "V10", "Q2", "PSFC", "PRECIP"]
UNITS = ["K", "m s-1", "m s-1", "kg kg-1", "Pa", "mm"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, shape: list[int]) -> dict[str, object]:
    return {
        "file": path.name,
        "shape": shape,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_f32(path: Path, values: np.ndarray | torch.Tensor) -> None:
    array = (
        values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    path.write_bytes(np.asarray(array, dtype="<f4").tobytes())


def channel_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, (variable, unit) in enumerate(zip(VARIABLES, UNITS)):
        error = prediction[index].astype(np.float64) - target[index].astype(
            np.float64
        )
        rows.append(
            {
                "variable": variable,
                "unit": unit,
                "bias": float(error.mean()),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "targetMin": float(target[index].min()),
                "targetMax": float(target[index].max()),
                "predictionMin": float(prediction[index].min()),
                "predictionMax": float(prediction[index].max()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--gfs-previous", type=Path, required=True)
    parser.add_argument("--gfs-current", type=Path, required=True)
    parser.add_argument("--wrf-target", type=Path, required=True)
    parser.add_argument("--normalized-static", type=Path, required=True)
    parser.add_argument(
        "--static-source-sha256",
        default="090e9033d24d7c96f050f505d1a38ebba872d52fb82bcade841665c9a6ff0918",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("public/data/validation")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/heldout-wrf-d01-20250101T0600.json"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    previous = torch.load(
        args.gfs_previous, map_location="cpu", weights_only=True
    ).float()
    current = torch.load(
        args.gfs_current, map_location="cpu", weights_only=True
    ).float()
    static_values = np.fromfile(args.normalized_static, dtype="<f4")
    expected_static = 30 * 127 * 137
    if static_values.size != expected_static:
        raise ValueError(
            f"Expected {expected_static} normalized static values, "
            f"received {static_values.size}"
        )
    normalized_static = torch.from_numpy(
        static_values.reshape(30, 127, 137).copy()
    )

    input_data = torch.cat(
        [
            normalize_gfs(previous),
            normalize_gfs(current),
            normalized_static,
        ],
        dim=0,
    ).unsqueeze(0)
    valid_time = datetime(2025, 1, 1, 6)
    projection = normalized_projection(1, valid_time).unsqueeze(0)

    wrf_container = torch.load(
        args.wrf_target, map_location="cpu", weights_only=False
    )
    wrf_raw = wrf_container["target"].float()
    if tuple(wrf_raw.shape[:1]) != (7,):
        raise ValueError(
            f"Expected seven WRF channels, received {tuple(wrf_raw.shape)}"
        )
    wrf = F.interpolate(
        wrf_raw.unsqueeze(0),
        size=(99, 99),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    target = torch.cat(
        [
            wrf[[0, 1, 2, 5, 6]],
            wrf[3:4] + wrf[4:5],
        ],
        dim=0,
    )

    state_dict = torch.load(
        args.state_dict, map_location="cpu", weights_only=True
    )
    model = FiLMeRVariantB().eval()
    model.load_state_dict(state_dict, strict=True)
    with torch.no_grad():
        state, occurrence, intensity = model(input_data, projection)
    prediction = reconstruct_physical(
        state.numpy(), occurrence.numpy(), intensity.numpy()
    )[0]

    files = {
        "input": args.output_dir / "heldout-d01-20250101T0600-input.f32",
        "projection": (
            args.output_dir / "heldout-d01-20250101T0600-projection.f32"
        ),
        "target": (
            args.output_dir / "heldout-d01-20250101T0600-wrf-target.f32"
        ),
        "prediction": (
            args.output_dir
            / "heldout-d01-20250101T0600-python-prediction.f32"
        ),
    }
    write_f32(files["input"], input_data)
    write_f32(files["projection"], projection)
    write_f32(files["target"], target)
    write_f32(files["prediction"], prediction)

    metrics = channel_metrics(prediction, target.numpy())
    manifest = {
        "schemaVersion": 1,
        "kind": "held-out-wrf-reference-not-observations",
        "split": "sample_test",
        "domain": "d01",
        "domainId": 1,
        "resolutionKm": 27,
        "validTime": "2025-01-01T06:00:00Z",
        "previousGfsTime": "2025-01-01T00:00:00Z",
        "currentGfsTime": "2025-01-01T03:00:00Z",
        "gridShape": [99, 99],
        "variables": [
            {"name": name, "unit": unit}
            for name, unit in zip(VARIABLES, UNITS)
        ],
        "semantics": {
            "state": "Physical WRF target fields bilinearly resized to 99x99.",
            "precipitation": "WRF RAINC + RAINNC accumulated precipitation.",
            "scope": (
                "One held-out sample_test case. WRF is model supervision, "
                "not an observation or the full paper test set."
            ),
        },
        "sourceSha256": {
            "gfsPrevious": sha256(args.gfs_previous),
            "gfsCurrent": sha256(args.gfs_current),
            "wrfTarget": sha256(args.wrf_target),
            "normalizedStatic": sha256(args.normalized_static),
            "staticGeogrid": args.static_source_sha256,
            "stateDict": sha256(args.state_dict),
        },
        "artifacts": {
            "input": artifact(files["input"], [1, 70, 127, 137]),
            "projection": artifact(files["projection"], [1, 16]),
            "target": artifact(files["target"], [6, 99, 99]),
            "pythonPrediction": artifact(
                files["prediction"], [6, 99, 99]
            ),
        },
        "metrics": metrics,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "case": {
            "split": manifest["split"],
            "domain": manifest["domain"],
            "validTime": manifest["validTime"],
        },
        "checkpointSha256": (
            "1e648a0a017be62416d43942f13dab5f41bb7a25b49d59eece18a9a6afb1b557"
        ),
        "metrics": metrics,
        "fixtureManifestSha256": sha256(manifest_path),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
