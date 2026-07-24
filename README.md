# FiLMeR WebGPU

Browser deployment of the released **FiLMeR v1.0 Variant B** regional weather
emulator. ONNX Runtime Web uses WebGPU with an fp16 graph when available and a
verified fp32 WebAssembly fallback.

**Live app:** <https://sustainability-lab.github.io/filmer-webgpu/>

## What it does

- Runs the 39,768,627-parameter checkpoint entirely in the browser.
- Downloads operational NOAA GFS inputs on demand from the public Google Cloud
  mirror using CORS-safe byte ranges.
- Decodes GRIB2 locally and reproduces the training crop, channel order,
  normalization, projection vector, and physical-unit reconstruction.
- Supports only the four trained domains: d01 at 27 km and d02–d04 at 9 km.
- Produces 99×99 fields for T2, U10, V10, Q2, PSFC, and precipitation.
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
account, so a byte-for-byte comparison with that file is not available.

For a static site, the Google Cloud record-range path is the only service-free
operational route we validated. It avoids each ~500 MB global GRIB file but the
selected records are still global fields, so a 96-hour run can transfer
hundreds of MB. A production regional-subset proxy would reduce this
substantially.

See [DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for exact acquisition semantics.

## Validated scope and limitations

| Question | FiLMeR v1.0 answer |
|---|---|
| Arbitrary location? | No. Only d01–d04 were trained. |
| Arbitrary resolution? | No. Only 27 km and 9 km were trained. |
| 1 km / hourly? | No. Requires new 1 km WRF supervision and an hourly model. |
| Standalone 96 h forecast? | No. It conditionally downscales a GFS trajectory. |
| Output cadence | 3 hours. |
| Output grid | Fixed 99×99. |
| Operational status | Research prototype; not safety-critical guidance. |

Unsupported geography, resolution, or cadence requires a separately trained
and held-out-validated checkpoint; the browser does not expose unvalidated
controls.

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
WebGPU is therefore a fast visualization/research path, not a numerically
interchangeable substitute for fp32.

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
