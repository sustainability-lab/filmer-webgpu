#!/usr/bin/env python3
"""Export the verified FiLMeR v1.0 Variant B checkpoint to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch
from onnxconverter_common import float16

from filmer_model import FiLMeRVariantB


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = FiLMeRVariantB()
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(result)
    model.eval()

    fp32_path = args.output_dir / "filmer_v1_variant_b_fp32.onnx"
    fp16_path = args.output_dir / "filmer_v1_variant_b_fp16.onnx"
    model_path = args.output_dir / "filmer_v1_variant_b_state_dict.pt"

    input_data = torch.zeros(1, 70, 127, 137, dtype=torch.float32)
    projection = torch.zeros(1, 16, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_data, projection),
            fp32_path,
            input_names=["input_data", "projection"],
            output_names=["state", "precip_occurrence", "precip_intensity"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(fp32_path)
    onnx.checker.check_model(onnx_model)
    fp16_model = float16.convert_float_to_float16(
        onnx_model,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    onnx.save(fp16_model, fp16_path)
    onnx.checker.check_model(onnx.load(fp16_path))

    torch.save(checkpoint["model_state_dict"], model_path)
    manifest = {
        "sourceCheckpoint": artifact(args.checkpoint),
        "checkpointEpoch": checkpoint.get("epoch"),
        "checkpointValidationLoss": checkpoint.get("val_loss"),
        "architecture": "FiLMeR v1.0 Variant B",
        "opset": args.opset,
        "inputShapes": {
            "input_data": [1, 70, 127, 137],
            "projection": [1, 16],
        },
        "outputs": {
            "state": [1, 5, 99, 99],
            "precip_occurrence": [1, 1, 99, 99],
            "precip_intensity": [1, 1, 99, 99],
        },
        "artifacts": {
            "fp32": artifact(fp32_path),
            "fp16": artifact(fp16_path),
            "stateDict": artifact(model_path),
        },
    }
    (args.output_dir / "model-artifacts.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
