# Browser output format

The app downloads a ZIP with:

- `manifest.json`
- `filmer-output.f32`
- `grid-latitude.f32`
- `grid-longitude.f32`

`filmer-output.f32` is little-endian float32 in C order with shape
`[6, 99, 99]`. Channels are:

1. T2, K
2. U10, m s−1
3. V10, m s−1
4. Q2, kg kg−1
5. PSFC, Pa
6. precipitation, mm

Latitude and longitude are little-endian float32 arrays with shape `[99,99]`.
The manifest records the domain, nominal resolution, forecast timestamp,
checkpoint checksum, backend, units, and conditional semantics.

Convert to NetCDF:

```bash
uv run python scripts/output_to_netcdf.py browser-output.zip output.nc
```

The ZIP contains one forecast time. A sequence UI currently retains and exports
the most recently computed field; it does not silently imply a four-dimensional
forecast cube.
