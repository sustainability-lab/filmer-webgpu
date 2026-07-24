#!/usr/bin/env python3
"""Split a model into GitHub-compatible, same-origin browser artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--chunk-bytes", type=int, default=80_000_000)
    args = parser.parse_args()

    if args.chunk_bytes <= 0 or args.chunk_bytes >= 100_000_000:
        raise SystemExit("--chunk-bytes must be between 1 and 99,999,999")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, str | int]] = []
    with args.source.open("rb") as source:
        index = 0
        while chunk := source.read(args.chunk_bytes):
            filename = f"{args.source.name}.part-{index:02d}"
            target = args.output_dir / filename
            target.write_bytes(chunk)
            parts.append(
                {
                    "file": filename,
                    "bytes": len(chunk),
                    "sha256": hashlib.sha256(chunk).hexdigest(),
                }
            )
            index += 1
    print(json.dumps(parts, indent=2))


if __name__ == "__main__":
    main()
