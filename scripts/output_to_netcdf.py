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
        field = read_f32(bundle, "filmer-output.f32", (6, 99, 99))
        coordinates = manifest.get("coordinates", {})
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
            ("y", "x"),
            field[index],
            {"units": units[index], "grid_mapping": "latitude_longitude"},
        )
        for index, name in enumerate(variables)
    }
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": (("y", "x"), latitude, {"units": "degrees_north"}),
            "longitude": (("y", "x"), longitude, {"units": "degrees_east"}),
        },
        attrs={
            "title": "FiLMeR v1.0 conditional regional downscaling output",
            "forecast_time": manifest["forecastTime"],
            "domain": manifest["domain"],
            "resolution_km": manifest["resolutionKm"],
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
