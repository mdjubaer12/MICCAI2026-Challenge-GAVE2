#!/usr/bin/env python3
"""Compare generated Task 1 PNGs with the frozen 8.16666 payload."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "reference" / "SHA256SUMS",
    )
    args = parser.parse_args()

    expected: dict[str, str] = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[Path(relative).name] = digest

    actual_names = {path.name for path in args.prediction_dir.glob("*.png")}
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    mismatched = sorted(
        name
        for name in expected_names & actual_names
        if sha256(args.prediction_dir / name) != expected[name]
    )
    if missing or extra or mismatched:
        raise SystemExit(
            "REFERENCE COMPARISON FAILED\n"
            f"missing={missing}\nextra={extra}\nmismatched={mismatched}"
        )
    print(f"REFERENCE COMPARISON PASSED: {len(expected)}/{len(expected)} files match")


if __name__ == "__main__":
    main()
