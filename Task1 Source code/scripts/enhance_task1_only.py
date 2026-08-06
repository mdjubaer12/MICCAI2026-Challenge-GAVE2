#!/usr/bin/env python3
"""Run only the exact Task 1 branch of the original topology-enhancement script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from enhance_task1_task2_topology import process_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    roi_dir = args.dataset_root / "validation" / "masks"
    if not roi_dir.is_dir():
        raise FileNotFoundError(f"Missing validation ROI directory: {roi_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    process_directory(args.input_dir, roi_dir, args.output_dir)


if __name__ == "__main__":
    main()
