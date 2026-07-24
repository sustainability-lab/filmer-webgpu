# Provenance

## FiLMeR source

Read-only source package inspected on 24 July 2026:

```text
/home/balbir.prasad/wrf/GMD_submission
```

The package was identified through the Sustainability Lab Slack discussion with
Balbir Prasad. No source files in that directory were modified.

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
- The paper calls the model domain/resolution agnostic, but validation covers
  only four fixed domains at 27 and 9 km.
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

## GFS

NOAA GFS data are public and are fetched at runtime from the
`global-forecast-system` public Google Cloud bucket. The app records no claim
of NOAA endorsement.
