# Operational data pipeline

## Data source

Operational GFS data comes from NOAA’s public `global-forecast-system` Google
Cloud Storage bucket. The browser uses the JSON media endpoint because it
returns CORS headers and supports HTTP byte ranges:

```text
https://storage.googleapis.com/download/storage/v1/b/global-forecast-system/o/{encoded-object}?alt=media
```

For a cycle `YYYYMMDD/HH` and lead `FFF`, the objects are:

```text
gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF.idx
```

The index is parsed first. The browser selects the 20 records below, requests
their exact byte ranges, decodes GRIB2 in JavaScript, and discards all points
outside the training crop.

## Checkpoint channel order

| Channel | GRIB variable | Level |
|---:|---|---|
| 0 | TMP | 2 m above ground |
| 1 | SPFH | 2 m above ground |
| 2 | PRES | surface |
| 3 | UGRD | 10 m above ground |
| 4 | VGRD | 10 m above ground |
| 5 | TCDC | entire atmosphere |
| 6 | PRATE | surface |
| 7 | TMP | 1000 mb |
| 8–10 | TMP | 850, 700, 500 mb |
| 11–13 | UGRD | 850, 700, 500 mb |
| 14–16 | VGRD | 850, 700, 500 mb |
| 17–19 | SPFH | 850, 700, 500 mb |

This order comes from the readable training extraction script and is consistent
with the released normalization statistics. It differs from Appendix B of the
paper draft, which lists only eight predictors, and from an earlier inferred
metadata draft that incorrectly used 250 hPa.

## Crop and normalization

The training crop script assumes the canonical 0.25° GFS grid:

- latitude: 90° to −90°, 721 points
- longitude: 0° to 359.75°, 1,440 points
- d01 bounds: 6.03–35.48° N, 67.12–98.88° E
- buffer: 1°
- slice: `[:, 214:341, 264:401]`
- result: `[20, 127, 137]`

Each channel is standardized with its committed 2023–2024 training mean and
standard deviation. Two frames are stacked as `[t−3h, t]`.

## Forecast-sequence semantics

For cycle `C`, the first output at `C+3 h` requires valid-time inputs `C−3 h`
and `C`. The browser obtains them as:

- previous cycle `(C−6 h)`, forecast lead `+3 h`
- current cycle `C`, forecast lead `+0 h`

Later frames use current-cycle leads `+3, +6, …`. A horizon `H` requires
`H/3 + 1` GFS frames and emits `H/3` FiLMeR fields. This use of forecast-lead
GFS trajectories is operationally useful but is not a validation of FiLMeR as
an autonomous long-range forecast model.

## Static geography

`configs/namelist.wps` defines the four WRF grids. The app ships:

- 12 normalized d01 static tensors, one for each monthly climatology
- 99×99 latitude/longitude coordinates for d01–d04

The GMD source loader uses `geo_em.d01.nc` for the static encoder even when the
projection vector selects d02–d04; the browser preserves that behavior.

The original static NetCDF had mode `0660` and was not readable to the
deployment account. It was regenerated in an isolated temporary directory
using the authors’ published namelist and their read-only WPS executable,
GEOGRID table, and geography directory. This is reproducible configuration
parity, not a bytewise comparison with the original file.

## Bandwidth

Record-range requests avoid downloading each full ~500 MB global file, but GRIB
compression operates per global field. A long trajectory still transfers
hundreds of MB. For production use, place a small regional subset/cache service
in front of NOAA; it should return the exact 20×127×137 float32 tensor plus
source cycle, lead, object generation, record offsets, and checksums.
