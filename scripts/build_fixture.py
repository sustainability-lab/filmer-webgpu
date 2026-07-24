#!/usr/bin/env python3
"""Build a deterministic browser fixture from two public GFS tensor frames.

The static input is deliberately the normalized mean (all zeros), because the
domain-specific WPS geogrid file was not readable from the supplied account.
This fixture validates conversion and runtime parity; it is not a weather
forecast product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from filmer_model import FiLMeRVariantB
from preprocessing import normalize_gfs, normalized_projection, reconstruct_physical


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_f32(path: Path, array: np.ndarray) -> None:
    np.asarray(array, dtype="<f4").tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gfs-previous", type=Path, required=True)
    parser.add_argument("--gfs-current", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("public/data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    previous = torch.load(args.gfs_previous, map_location="cpu", weights_only=True)
    current = torch.load(args.gfs_current, map_location="cpu", weights_only=True)
    static_mean = torch.zeros(30, 127, 137, dtype=torch.float32)
    input_data = torch.cat(
        [normalize_gfs(previous), normalize_gfs(current), static_mean], dim=0
    ).unsqueeze(0)
    forecast_time = datetime(2025, 1, 1, 6)
    projection = normalized_projection(1, forecast_time).unsqueeze(0)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = FiLMeRVariantB().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.no_grad():
        state, occurrence, intensity = model(input_data, projection)
    physical = reconstruct_physical(
        state.numpy(), occurrence.numpy(), intensity.numpy()
    )

    files = {
        "input": args.output_dir / "fixture-input.f32",
        "projection": args.output_dir / "fixture-projection.f32",
        "state": args.output_dir / "fixture-state.f32",
        "occurrence": args.output_dir / "fixture-occurrence.f32",
        "intensity": args.output_dir / "fixture-intensity.f32",
        "physical": args.output_dir / "fixture-physical.f32",
    }
    write_f32(files["input"], input_data.numpy())
    write_f32(files["projection"], projection.numpy())
    write_f32(files["state"], state.numpy())
    write_f32(files["occurrence"], occurrence.numpy())
    write_f32(files["intensity"], intensity.numpy())
    write_f32(files["physical"], physical)

    manifest = {
        "kind": "numerical-parity-fixture-not-operational-forecast",
        "forecastTime": forecast_time.isoformat(),
        "domain": "d01",
        "staticInput": "normalized-zero/mean placeholder",
        "inputs": {
            "input": {"file": files["input"].name, "shape": [1, 70, 127, 137]},
            "projection": {
                "file": files["projection"].name,
                "shape": [1, 16],
            },
        },
        "outputs": {
            "state": {"file": files["state"].name, "shape": [1, 5, 99, 99]},
            "occurrence": {
                "file": files["occurrence"].name,
                "shape": [1, 1, 99, 99],
            },
            "intensity": {
                "file": files["intensity"].name,
                "shape": [1, 1, 99, 99],
            },
            "physical": {
                "file": files["physical"].name,
                "shape": [1, 6, 99, 99],
            },
        },
        "sha256": {name: sha256(path) for name, path in files.items()},
    }
    (args.output_dir / "fixture.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
