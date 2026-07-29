#!/usr/bin/env python3
"""Convert a FiLMeR browser-output ZIP to a CF-oriented NetCDF file."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr


def read_f32(bundle: zipfile.ZipFile, name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.frombuffer(bundle.read(name), dtype="<f4").copy()
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{name} contains {values.size} values; expected {expected}")
    return values.reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with zipfile.ZipFile(args.bundle) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        variables = manifest["variables"]
        units = manifest["units"]
        files = manifest.get("files", ["filmer-output.f32"])
        timestamps = manifest.get(
            "timestamps",
            [manifest.get("forecastTime")],
        )
        if len(files) != len(timestamps) or not files:
            raise ValueError(
                "Manifest files and timestamps must be non-empty and aligned"
            )
        fields = np.stack(
            [read_f32(bundle, name, (6, 99, 99)) for name in files],
            axis=0,
        )
        coordinates = manifest.get("coordinates") or {}
        latitude_file = coordinates.get("latitudeFile")
        longitude_file = coordinates.get("longitudeFile")
        latitude = (
            read_f32(bundle, latitude_file, (99, 99))
            if latitude_file
            else np.full((99, 99), np.nan, dtype=np.float32)
        )
        longitude = (
            read_f32(bundle, longitude_file, (99, 99))
            if longitude_file
            else np.full((99, 99), np.nan, dtype=np.float32)
        )

    data_vars = {
        name: (
            ("time", "y", "x"),
            fields[:, index],
            {"units": units[index], "grid_mapping": "latitude_longitude"},
        )
        for index, name in enumerate(variables)
    }
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.asarray(timestamps, dtype="datetime64[ns]"),
            "latitude": (("y", "x"), latitude, {"units": "degrees_north"}),
            "longitude": (("y", "x"), longitude, {"units": "degrees_east"}),
        },
        attrs={
            "title": "FiLMeR v1.0 conditional regional downscaling output",
            "initialization": manifest.get("initialization", ""),
            "domain": manifest["domain"],
            "trained_resolution_km": manifest.get(
                "trainedResolutionKm",
                manifest.get("resolutionKm"),
            ),
            "model": manifest["model"],
            "checkpoint_sha256": manifest["checkpointSha256"],
            "runtime_backend": manifest["backend"],
            "semantics": manifest["semantics"],
            "limitations": (
                "Not a standalone autoregressive forecast; conditioned on "
                "GFS at two consecutive three-hour times."
            ),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
