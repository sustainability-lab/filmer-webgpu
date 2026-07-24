#!/usr/bin/env python3
"""Package a pre-cropped GFS trajectory and WPS static grid for the browser.

The browser bundle is intentionally narrow: it does not download or crop
global GRIB files. It packages the exact 20-channel tensors and geo_em fields
expected by the FiLMeR training pipeline.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from preprocessing import (
    load_static_geogrid,
    normalize_static,
    resize_bilinear,
)


def find_frame(directory: Path, timestamp: datetime) -> Path:
    candidates = [
        directory / f"gfs_{timestamp:%Y-%m-%d_%H}.pt",
        directory / f"gdas_{timestamp:%Y-%m-%d_%H:%M:%S}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No pre-cropped GFS tensor for {timestamp.isoformat()} in {directory}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gfs-dir", type=Path, required=True)
    parser.add_argument("--static-nc", type=Path, required=True)
    parser.add_argument("--domain", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument(
        "--initialization",
        type=datetime.fromisoformat,
        required=True,
        help="Forecast initialization, e.g. 2025-01-01T00:00:00",
    )
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon <= 0 or args.horizon % 3:
        raise ValueError("Horizon must be a positive multiple of three hours")

    steps = args.horizon // 3
    frame_times = [
        args.initialization - timedelta(hours=3) + timedelta(hours=3 * index)
        for index in range(steps + 1)
    ]
    frame_paths = [find_frame(args.gfs_dir, timestamp) for timestamp in frame_times]
    frames = torch.stack(
        [
            torch.load(path, map_location="cpu", weights_only=True).float()
            for path in frame_paths
        ]
    )
    if tuple(frames.shape[1:]) != (20, 127, 137):
        raise ValueError(
            f"Expected [N,20,127,137] pre-cropped GFS; received {tuple(frames.shape)}"
        )

    # This follows the submitted loader: the static month is selected once from
    # the first GFS frame in the sequence.
    static = load_static_geogrid(args.static_nc, frame_times[0].month)
    static = resize_bilinear(normalize_static(static), (127, 137))
    with xr.open_dataset(args.static_nc) as dataset:
        latitude_values = dataset[
            "XLAT_M" if "XLAT_M" in dataset else "XLAT"
        ].values.copy()
        longitude_values = dataset[
            "XLONG_M" if "XLONG_M" in dataset else "XLONG"
        ].values.copy()
    if latitude_values.ndim == 3:
        latitude_values = latitude_values[0]
        longitude_values = longitude_values[0]
    latitude = resize_bilinear(
        torch.from_numpy(latitude_values).float().unsqueeze(0), (99, 99)
    ).squeeze(0)
    longitude = resize_bilinear(
        torch.from_numpy(longitude_values).float().unsqueeze(0), (99, 99)
    ).squeeze(0)
    output_times = [
        args.initialization + timedelta(hours=3 * (index + 1))
        for index in range(steps)
    ]
    manifest = {
        "schemaVersion": 1,
        "kind": "filmer-conditional-gfs-sequence",
        "domainId": args.domain,
        "initialization": args.initialization.isoformat(),
        "horizonHours": args.horizon,
        "cadenceHours": 3,
        "frameTimes": [timestamp.isoformat() for timestamp in frame_times],
        "outputTimes": [timestamp.isoformat() for timestamp in output_times],
        "gfs": {"file": "gfs.f32", "shape": list(frames.shape), "normalized": False},
        "static": {
            "file": "static-normalized.f32",
            "shape": list(static.shape),
            "normalized": True,
            "month": frame_times[0].month,
        },
        "outputGrid": {
            "latitudeFile": "grid-latitude.f32",
            "longitudeFile": "grid-longitude.f32",
            "shape": [99, 99],
        },
        "semantics": (
            "Each output at t+3h uses GFS(t-3h,t). Outputs are not fed back "
            "into the next step."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        bundle.writestr(
            "gfs.f32", np.asarray(frames.numpy(), dtype="<f4").tobytes()
        )
        bundle.writestr(
            "static-normalized.f32",
            np.asarray(static.numpy(), dtype="<f4").tobytes(),
        )
        bundle.writestr(
            "grid-latitude.f32",
            np.asarray(latitude.numpy(), dtype="<f4").tobytes(),
        )
        bundle.writestr(
            "grid-longitude.f32",
            np.asarray(longitude.numpy(), dtype="<f4").tobytes(),
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
