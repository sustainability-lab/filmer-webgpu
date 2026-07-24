#!/usr/bin/env python3
"""Build browser-ready FiLMeR static tensors from published WPS grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from preprocessing import load_static_geogrid, normalize_static, resize_bilinear


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    values = np.asarray(tensor.detach().cpu().numpy(), dtype="<f4")
    path.write_bytes(values.tobytes())
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def coordinate(dataset: xr.Dataset, primary: str, fallback: str) -> torch.Tensor:
    values = dataset[primary if primary in dataset else fallback].values.copy()
    if values.ndim == 3:
        values = values[0]
    return torch.from_numpy(values).float().unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geogrid-dir",
        type=Path,
        required=True,
        help="Directory containing geo_em.d01.nc through geo_em.d04.nc",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d01 = args.geogrid_dir / "geo_em.d01.nc"
    static_artifacts: list[dict[str, object]] = []
    for month in range(1, 13):
        static = load_static_geogrid(d01, month)
        prepared = resize_bilinear(normalize_static(static), (127, 137))
        artifact = write_f32(
            args.output_dir / f"static-month-{month:02d}.f32", prepared
        )
        artifact.update({"month": month, "shape": [30, 127, 137]})
        static_artifacts.append(artifact)

    grids: list[dict[str, object]] = []
    for domain_id in range(1, 5):
        source = args.geogrid_dir / f"geo_em.d0{domain_id}.nc"
        with xr.open_dataset(source) as dataset:
            latitude = resize_bilinear(
                coordinate(dataset, "XLAT_M", "XLAT"), (99, 99)
            ).squeeze(0)
            longitude = resize_bilinear(
                coordinate(dataset, "XLONG_M", "XLONG"), (99, 99)
            ).squeeze(0)
        lat_artifact = write_f32(
            args.output_dir / f"grid-d0{domain_id}-latitude.f32", latitude
        )
        lon_artifact = write_f32(
            args.output_dir / f"grid-d0{domain_id}-longitude.f32", longitude
        )
        grids.append(
            {
                "domainId": domain_id,
                "shape": [99, 99],
                "latitude": lat_artifact,
                "longitude": lon_artifact,
                "sourceSha256": sha256(source),
            }
        )

    manifest = {
        "schemaVersion": 1,
        "kind": "filmer-static-geogrid",
        "source": {
            "wpsNamelist": "configs/namelist.wps",
            "staticEncoderDomain": "d01",
            "d01Sha256": sha256(d01),
        },
        "normalization": (
            "LANDMASK and LANDUSEF raw; channels 1:9 per-grid z-score; "
            "bilinear resize with align_corners=false"
        ),
        "static": static_artifacts,
        "grids": grids,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
