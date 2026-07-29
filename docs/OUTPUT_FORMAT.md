# Browser output format

The app downloads a ZIP with:

- `manifest.json`
- one `forecast-<timestamp>.f32` file per timestamp
- `grid-latitude.f32`
- `grid-longitude.f32`

Each forecast file is little-endian float32 in C order with shape `[6, 99, 99]`.
Channels are:

1. T2, K
2. U10, m s−1
3. V10, m s−1
4. Q2, kg kg−1
5. PSFC, Pa
6. precipitation, mm

Latitude and longitude are little-endian float32 arrays with shape `[99,99]`.
The manifest records the domain, trained resolution, initialization,
timestamps, file names, checkpoint checksum, backend, units, and conditional
semantics.

Convert to NetCDF:

```bash
uv run python scripts/output_to_netcdf.py browser-output.zip output.nc
```

The public workflow exports every selected 3-hourly forecast timestamp in one
ZIP.
