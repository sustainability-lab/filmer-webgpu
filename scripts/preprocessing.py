"""Exact FiLMeR v1.0 preprocessing and physical-unit reconstruction."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
METADATA = json.loads((ROOT / "src/data/model-metadata.json").read_text())


def normalize_gfs(frame: torch.Tensor) -> torch.Tensor:
    """Normalize one [20,H,W] GFS frame using training-set statistics."""
    if tuple(frame.shape[:1]) != (20,):
        raise ValueError(f"Expected [20,H,W] GFS tensor; received {tuple(frame.shape)}")
    mean = torch.tensor(METADATA["gfs"]["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(METADATA["gfs"]["std"], dtype=torch.float32)[:, None, None]
    return (frame.float() - mean) / std


def normalize_static(static: torch.Tensor) -> torch.Tensor:
    """Apply the exact mixed static-field transform used by the training loader.

    Channel 0 is LANDMASK and is preserved. Channels 1:9 are standardized per
    field over space. Channels 9:30 are LANDUSEF one-hot fractions and preserved.
    """
    if tuple(static.shape[:1]) != (30,):
        raise ValueError(
            f"Expected [30,H,W] static tensor; received {tuple(static.shape)}"
        )
    parts = [static[0:1].float()]
    for index in range(1, 9):
        field = static[index : index + 1].float()
        std = field.std()
        if std <= 0:
            std = torch.tensor(1.0)
        parts.append((field - field.mean()) / std)
    parts.append(static[9:30].float())
    return torch.cat(parts)


def resize_bilinear(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)


def load_static_geogrid(path: Path, month: int) -> torch.Tensor:
    """Read 30 static channels from a WPS geo_em NetCDF file.

    xarray is imported lazily so parity and export do not require NetCDF support.
    """
    import xarray as xr

    month_index = month - 1
    single = ["LANDMASK", "HGT_M", "SOILTEMP", "SNOALB", "CON", "VAR"]
    monthly = ["ALBEDO12M", "GREENFRAC", "LAI12M"]
    channels: list[torch.Tensor] = []
    with xr.open_dataset(path) as dataset:
        for name in single:
            values = np.nan_to_num(dataset[name].values.copy())
            if values.ndim == 3:
                values = values[0]
            channels.append(torch.from_numpy(values).float().unsqueeze(0))
        for name in monthly:
            values = np.nan_to_num(dataset[name].values.copy())
            if values.ndim == 4:
                values = values[0]
            channels.append(
                torch.from_numpy(values[month_index]).float().unsqueeze(0)
            )
        landuse = np.nan_to_num(dataset["LANDUSEF"].values.copy())
        if landuse.ndim == 4:
            landuse = landuse[0]
        channels.append(torch.from_numpy(landuse).float())
    stacked = torch.cat(channels)
    if stacked.shape[0] != 30:
        raise ValueError(f"Expected 30 static channels; received {stacked.shape[0]}")
    return stacked


def raw_projection(domain_id: int, timestamp: datetime) -> torch.Tensor:
    domain = next(
        item for item in METADATA["domains"] if item["id"] == domain_id
    )
    lat_min, lon_min, lat_max, lon_max = domain["modelBounds"]
    resolution = domain["resolutionKm"]
    lat_center = (lat_min + lat_max) / 2.0
    lon_center = (lon_min + lon_max) / 2.0
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    area = lat_span * lon_span
    aspect = lat_span / lon_span
    if domain_id == 1:
        relative_lat = relative_lon = 0.5
    else:
        parent = METADATA["domains"][0]["modelBounds"]
        relative_lat = (lat_center - parent[0]) / (parent[2] - parent[0])
        relative_lon = (lon_center - parent[1]) / (parent[3] - parent[1])
    return torch.tensor(
        [
            lat_min,
            lon_min,
            lat_max,
            lon_max,
            lat_center,
            lon_center,
            lat_span,
            lon_span,
            area,
            aspect,
            relative_lat,
            relative_lon,
            resolution,
            timestamp.hour / 24.0,
            timestamp.timetuple().tm_yday / 365.0,
            timestamp.month / 12.0,
        ],
        dtype=torch.float32,
    )


def normalized_projection(domain_id: int, timestamp: datetime) -> torch.Tensor:
    raw = raw_projection(domain_id, timestamp)
    mean = torch.tensor(METADATA["projection"]["mean"], dtype=torch.float32)
    std = torch.tensor(METADATA["projection"]["std"], dtype=torch.float32)
    return (raw - mean) / std


def assemble_input(
    previous_gfs: torch.Tensor,
    current_gfs: torch.Tensor,
    static: torch.Tensor,
) -> torch.Tensor:
    previous = resize_bilinear(normalize_gfs(previous_gfs), (127, 137))
    current = resize_bilinear(normalize_gfs(current_gfs), (127, 137))
    static_ready = resize_bilinear(normalize_static(static), (127, 137))
    return torch.cat([previous, current, static_ready], dim=0)


def reconstruct_physical(
    state: np.ndarray,
    occurrence: np.ndarray,
    intensity: np.ndarray,
) -> np.ndarray:
    """Convert raw model heads to [T2,U10,V10,Q2,PSFC,PRECIP] physical units."""
    source_indices = METADATA["targets"]["stateIndices"]
    means = np.asarray(METADATA["targets"]["mean"], dtype=np.float32)[source_indices]
    stds = np.asarray(METADATA["targets"]["std"], dtype=np.float32)[source_indices]
    state_physical = state * stds.reshape(1, 5, 1, 1) + means.reshape(
        1, 5, 1, 1
    )
    wet = 1.0 / (1.0 + np.exp(-occurrence)) > 0.5
    precip = np.expm1(np.minimum(intensity, 10.0)) * wet
    return np.concatenate([state_physical, precip], axis=1).astype(np.float32)


def conditional_sequence_requirements(horizon_hours: int) -> dict[str, int]:
    if horizon_hours <= 0 or horizon_hours % 3:
        raise ValueError("FiLMeR v1.0 horizon must be a positive multiple of 3 hours")
    steps = horizon_hours // 3
    return {
        "output_steps": steps,
        "gfs_frames": steps + 1,
        "first_gfs_lead_hours": -3,
        "last_gfs_lead_hours": horizon_hours - 3,
    }
