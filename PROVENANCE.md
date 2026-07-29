# Provenance

## FiLMeR source

Read-only source package inspected on 24 July 2026:

```text
/home/balbir.prasad/wrf/GMD_submission
```

The package was identified through the Sustainability Lab Slack discussion with
Balbir Prasad. No source files in that directory were modified.

The held-out comparison uses the source package’s `sample_test` GFS pair at
2025-01-01 00Z/03Z and its d01 WRF target at 06Z. The owner-only WRF file was
read through a container with a read-only bind mount, copied to an isolated
temporary directory, hash-checked before and after (`5175b72e…fe938`), and the
temporary remote copy was removed. Source permissions and contents were not
changed.

Checkpoint:

```text
file: model_filmerv1.0.pth
size: 477,566,163 bytes
sha256: 1e648a0a017be62416d43942f13dab5f41bb7a25b49d59eece18a9a6afb1b557
epoch: 130
validation loss: 0.1788209960476993
```

Strict PyTorch loading matched all Variant B keys with no missing or unexpected
parameters. The architecture has 39,768,627 trainable parameters.

## Paper

Draft reviewed: *FiLMeR (v1.0): A boundary-independent, dual-encoder deep
learning emulator for multi-resolution regional weather forecasting* (38
pages, accessed through the connected Overleaf project on 24 July 2026).

Implementation details are taken from the checkpoint and readable source when
they conflict with the draft. Notable conflicts:

- Appendix B describes 8 fields per GFS frame; the checkpoint/source use 20.
- The source uses 1000/850/700/500 hPa temperature and 850/700/500 hPa U/V/Q.
- The paper positions FiLMeR as domain/resolution agnostic. The released
  checkpoint does demonstrate shared, resolution-conditioned weights across
  four domains at 27 and 9 km; this is not validation at resolutions or
  geographies absent from training.
- The paper’s speedup arithmetic/timing descriptions are internally
  inconsistent; this repository reports measured scope-specific timings only.

## Conversion

`scripts/export_model.py` reconstructs Variant B, loads the checkpoint strictly,
exports opset 18 with static shapes, validates the ONNX graph, and converts the
fp32 graph to fp16 while retaining fp32 I/O.

Artifacts are recorded in `artifacts/model-artifacts.json` during conversion
and published as GitHub release assets.

## Fixture

The committed fixture is deterministic and exists to test PyTorch/ONNX/browser
numerical parity. Its static channels are zeros because the original static
file was initially unreadable; it is not an operational meteorological sample.
Operational runs use the regenerated WPS static artifacts in
`public/data/static/`.

The regenerated d01 geogrid and the original source geogrid both hash to
`090e9033d24d7c96f050f505d1a38ebba872d52fb82bcade841665c9a6ff0918`.
The held-out validation fixture therefore uses the exact normalized static
tensor rather than an approximation.

## GFS

NOAA GFS data are public and are fetched at runtime from the
`global-forecast-system` public Google Cloud bucket. The app records no claim
of NOAA endorsement.

The training workflow’s UCAR GDEX dataset and variable-specific paper plotting
source were supplied by Balbir Prasad in the Sustainability Lab `#emulator`
thread on 29 July 2026. The plotting source assigns T2/PSFC to `viridis`,
U10/V10 to `coolwarm`, Q2 to `YlGnBu`, and precipitation to `Blues`; it renders
all rasters with `origin="lower"` and converts PSFC from Pa to hPa. The browser
now preserves those display semantics and derives wind speed/RH2 with the
script’s equations rather than presenting them as direct model outputs.

UCAR dataset `d084001` exposes the same NCEP GFS 0.25-degree products but no
`.idx` sidecar. The committed audit uses the NOAA index to request identical
byte ranges from both providers for 2024-05-11 00Z `f000` and `f003`. All 40
FiLMeR records (37,535,367 bytes) have matching SHA-256 values. This rules out
a provider-induced shift for the audited pair only; it does not establish
temporal distribution stability or forecast skill.
