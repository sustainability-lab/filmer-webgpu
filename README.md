# FiLMeR WebGPU

Browser deployment of the released **FiLMeR v1.0 Variant B** regional weather
emulator. ONNX Runtime Web uses the verified fp32 WebAssembly path by default
and exposes the fp16 WebGPU path explicitly as experimental.

**Live app:** <https://sustainability-lab.github.io/filmer-webgpu/>

## What it does

- Runs the 39,768,627-parameter checkpoint entirely in the browser.
- Offers an instant cached 2025-01-01 prediction/WRF comparison with no GFS
  download, plus on-demand NOAA and UCAR GDEX inputs.
- Decodes GRIB2 locally and reproduces the training crop, channel order,
  normalization, projection vector, and physical-unit reconstruction.
- Supports only the four trained domains: d01 at 27 km and d02–d04 at 9 km.
- Produces 99×99 fields for T2, U10, V10, Q2, PSFC, and precipitation.
- Displays 10 m wind speed and 2 m relative humidity as derived fields using
  the same equations and palettes as the paper plotting script; downloaded raw
  output remains the six model channels.
- Uses the paper plotting script’s variable-specific palettes (`viridis`,
  `coolwarm`, `YlGnBu`, and `Blues`), `origin="lower"` orientation, and hPa
  pressure display; includes units-aware color bars, drag/wheel/touch map
  navigation, and play/pause/time scrubbing.
- Runs a browser-side comparison against an exact held-out WRF target with
  bias, MAE, RMSE, and prediction/reference/error views.
- Exports a ZIP with raw float32 fields, units, forecast metadata, and 2-D
  latitude/longitude arrays; `scripts/output_to_netcdf.py` converts it to
  NetCDF.

This is **conditional downscaling**, not a standalone forecast model. An output
at `t+3 h` consumes GFS at `t-3 h` and `t`. A 96-hour request therefore performs
32 independent FiLMeR steps from 33 GFS frames; outputs are never fed back as
inputs.

## Operational inputs

The checkpoint consumes 20 GFS fields at each of two consecutive times:

1. TMP 2 m, SPFH 2 m, PRES surface, UGRD 10 m, VGRD 10 m, TCDC entire
   atmosphere, and PRATE surface.
2. TMP at 1000/850/700/500 hPa.
3. UGRD, VGRD, and SPFH at 850/700/500 hPa.

The exact 0.25° global-grid crop is latitude rows `214:341` and longitude
columns `264:401`, yielding 127×137. The crop corresponds to the parent India
domain plus a 1° buffer.

The static encoder uses the published WPS configuration in
`configs/namelist.wps`. Browser artifacts were regenerated with the same
read-only WPS executable, GEOGRID table, and geography installation used by the
authors. The original `geo_em.d01.nc` was not readable to the deployment
account directly; a read-only hash through an isolated container confirmed
that it is byte-for-byte identical to the published d01 source geogrid:
`090e9033d24d7c96f050f505d1a38ebba872d52fb82bcade841665c9a6ff0918`.

NOAA’s Google Cloud mirror is the operational default. UCAR GDEX is also
selectable as the training-source archive. UCAR supports CORS byte ranges but
does not publish a GRIB index, so the app uses the matching NOAA `.idx` only to
locate records and downloads the data bytes from UCAR. NOAA ranges are read
four at a time; UCAR ranges are deliberately serialized in accordance with
GDEX’s warning against simultaneous downloads.

We audited all 40 predictor records in the UCAR/NOAA 2024-05-11 00Z
`f000/f003` pair: all 37,535,367 selected bytes were record-for-record
SHA-256 identical. Thus provider choice induces exactly zero input/output
difference for that audited same-cycle pair. This does **not** measure temporal
distribution shift or live forecast error; those require time-matched WRF or
observations. The full evidence is in
`reports/gfs-source-equivalence-20240511T00Z.json` and can be regenerated with:

```bash
node scripts/audit_gfs_source_equivalence.mjs
```

The browser UCAR smoke test then ran a 3-hour archive product from two GDEX
frames successfully: 35.7 MB transferred, 68.2 s end to end, and 919.6 ms model
compute on the tested Chromium/WASM session. Scope and inputs are recorded in
`reports/ucar-browser-smoke-20240511T00Z.json`.

Record ranges avoid each ~500 MB global GRIB file, but the selected records are
still global fields, so a 96-hour run can transfer hundreds of MB. A production
regional-subset proxy would reduce this substantially.

See [DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for exact acquisition semantics.

## Validated scope and limitations

| Question | FiLMeR v1.0 answer |
|---|---|
| Arbitrary location? | No. Only d01–d04 were trained. |
| Resolution-conditioned? | Yes. One checkpoint learned both 27 km and 9 km domains, and resolution is an explicit model input. |
| Arbitrary resolution validated? | Not yet. The UI exposes a 1–54 km conditioning-sensitivity probe; only 27 km and 9 km have training support. |
| 1 km / hourly? | No. Requires new 1 km WRF supervision and an hourly model. |
| Standalone 96 h forecast? | No. It conditionally downscales a GFS trajectory. |
| Output cadence | 3 hours. |
| Output grid | Fixed 99×99. |
| Operational status | Research prototype; not safety-critical guidance. |

FiLMeR's multi-resolution selling point is real at the architecture level: one
set of weights is conditioned on resolution and was trained across both 9 km
and 27 km domains. The browser deliberately exposes that conditioning input
from 1–54 km so the transfer hypothesis can be tested. Unsupported geography,
resolution, or cadence still requires a suitable geogrid, supervision, and
held-out validation before it becomes a forecast product. The current probe
retains the original d01 geogrid and fixed 99×99 output, so it cannot add 1 km
spatial information.

The committed conditioning probe actually runs the released checkpoint at
`1, 3, 9, 27, 54 km` values while holding the d01 input/geography fixed. On the
same 27 km WRF target, changing only the scalar from 27 km to 1 km increases T2
RMSE from `1.61 K` to `6.02 K`; the output is still `99×99`. This confirms that
the network responds to the control, but it does not validate a 1 km product.
Full results are in `reports/resolution-conditioning-probe.json`.

## Held-out WRF comparison

**Run default prediction** loads one exact cached `sample_test` case:

```text
GFS inputs   2025-01-01 00Z and 03Z
WRF target   d01, valid 2025-01-01 06Z
target grid  raw 7×129×129 → bilinear 6×99×99
semantics    state fields + (RAINC + RAINNC)
```

The target is the WRF supervision used by the FiLMeR evaluation pipeline; it is
not station data, radar, or an analysis. For the released checkpoint and
matched 27 km conditioning, Python produced:

| Variable | Bias | MAE | RMSE |
|---|---:|---:|---:|
| T2 (K) | -0.0254 | 1.2424 | 1.6108 |
| U10 (m s-1) | 0.1068 | 1.2375 | 1.8088 |
| V10 (m s-1) | 0.4433 | 1.1295 | 1.5023 |
| Q2 (kg kg-1) | -0.000785 | 0.001148 | 0.001607 |
| PSFC (Pa) | -17.55 | 142.38 | 178.72 |
| precipitation (mm) | 0.0902 | 0.3183 | 1.2280 |

These are descriptive metrics for one held-out field, not a claim of aggregate
forecast skill. The exact tensor checksums and fuller ranges are in
`public/data/validation/manifest.json` and
`reports/heldout-wrf-d01-20250101T0600.json`. Live future runs are explicitly
marked `verification pending`.

## Reproducibility

Source checkpoint:

```text
SHA-256 1e648a0a017be62416d43942f13dab5f41bb7a25b49d59eece18a9a6afb1b557
epoch   130
val loss 0.1788209960476993
```

ONNX artifacts:

| Runtime | File | Size | SHA-256 |
|---|---:|---:|---|
| WebGPU | fp16 | 79,587,004 B | `163c94f72b23d9613fa1b2d958979663d569fea90f387e1b1f9d68181891af7a` |
| WASM/CPU | fp32 | 159,065,799 B | `19abe68725be837f912b832dc9d3748982fc03f644bc334ce4b96e63e628cdbc` |

Conversion uses PyTorch export to ONNX opset 18, strict checkpoint loading, ONNX
validation, and fp16 conversion with I/O types retained as fp32:

```bash
uv sync --locked
uv run python scripts/export_model.py \
  --checkpoint /path/to/model_filmerv1.0.pth \
  --output-dir artifacts
uv run python scripts/build_fixture.py \
  --checkpoint /path/to/model_filmerv1.0.pth
uv run python scripts/build_validation_fixture.py \
  --state-dict artifacts/filmer_v1_variant_b_state_dict.pt \
  --gfs-previous /path/to/gfs_2025-01-01_00.pt \
  --gfs-current /path/to/gfs_2025-01-01_03.pt \
  --wrf-target /path/to/2025-01-01_06:00:00.pt \
  --normalized-static public/data/static/static-month-01.f32
uv run python scripts/probe_resolution_conditioning.py \
  --state-dict artifacts/filmer_v1_variant_b_state_dict.pt
uv run python scripts/validate_export.py --precision fp32
uv run python scripts/benchmark_onnx.py
```

FP32 ONNX vs PyTorch parity on the committed deterministic fixture:

- maximum raw-output absolute error: `6.23703e-4`
- state maximum: `2.66433e-5`
- precipitation-intensity maximum: `1.77622e-5`
- physical-unit maximum: `0.0625` (pressure is in Pa)
- wet-mask disagreements: `0 / 9,801`

FP16 is deliberately reported separately: CPU parity showed maximum raw drift
`0.110511`, maximum physical drift `20.21875`, and 5 wet-mask disagreements.
An initial browser run exposed an ONNX Runtime Web 1.22 device-limit omission:
FiLMeR's `Concat` needs nine storage buffers, while the WebGPU default is
eight. `scripts/patch_ort_webgpu.mjs` now requests the adapter's advertised
limit during `npm install`. After the fix, a clean Chrome WebGPU run completed
without validation errors at `1.09` maximum raw drift and `1,198 ms` model
compute. That is still far from fp32 WASM parity, so WebGPU remains an explicit
experimental path rather than a numerically interchangeable default. The
observed before/after run is recorded in
`reports/parity-webgpu-browser.json`.

Native ONNX Runtime CPU on macOS 15.7.7 arm64 measured a median
`86.318 ms/step` over 10 runs; 32 model calls are `2.762 s` compute-only.
Download, GFS decoding, static preparation, and I/O are excluded. The paper’s
WRF and GPU figures use different hardware and timing boundaries and should not
be compared as end-to-end speedups.

Machine-readable reports are in `reports/`.

## Develop

```bash
npm ci
npm test
npm run dev
```

Production build:

```bash
npm run build
```

The app is static; GitHub Pages deployment is in
`.github/workflows/pages.yml`. The tagged release retains the original ONNX
files. Pages serves checksum-pinned parts from the same origin because GitHub
Release redirects do not expose CORS headers to browser JavaScript.

## Output

Use **Download output** after an inference. Convert the ZIP:

```bash
uv run python scripts/output_to_netcdf.py filmer-output.zip filmer-output.nc
```

See [OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md).

## Provenance and licensing

- [Model and source provenance](PROVENANCE.md)
- [Survey of India boundary provenance](MAP_PROVENANCE.md)
- [Code license](LICENSE)
- [Model artifact notice](MODEL_LICENSE.md)

The map boundary is the Survey of India official 1:16M outline, not a generic
international boundary dataset.
