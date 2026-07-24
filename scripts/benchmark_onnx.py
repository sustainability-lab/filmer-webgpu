#!/usr/bin/env python3
"""Benchmark fixed-shape FiLMeR inference and extrapolate a 96-hour sequence."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, default=Path("public/data"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--report", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    fixture = json.loads((args.fixture_dir / "fixture.json").read_text())
    input_data = np.fromfile(
        args.fixture_dir / fixture["inputs"]["input"]["file"], dtype="<f4"
    ).reshape(fixture["inputs"]["input"]["shape"])
    projection = np.fromfile(
        args.fixture_dir / fixture["inputs"]["projection"]["file"], dtype="<f4"
    ).reshape(fixture["inputs"]["projection"]["shape"])
    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    feeds = {"input_data": input_data, "projection": projection}
    for _ in range(args.warmup):
        session.run(None, feeds)
    timings = []
    for _ in range(args.runs):
        started = time.perf_counter()
        session.run(None, feeds)
        timings.append((time.perf_counter() - started) * 1000)
    report = {
        "runtime": "onnxruntime-python",
        "provider": session.get_providers()[0],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runs": args.runs,
        "stepMilliseconds": {
            "median": statistics.median(timings),
            "mean": statistics.mean(timings),
            "min": min(timings),
            "max": max(timings),
        },
        "conditional96HourComputeSeconds": statistics.median(timings) * 32 / 1000,
        "excludes": ["model download", "GFS retrieval", "geogrid preparation", "I/O"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
