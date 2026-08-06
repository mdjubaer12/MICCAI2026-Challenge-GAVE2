#!/usr/bin/env python3
"""Batch process Task 1 and Task 2 validation maps with skeleton topology enhancement."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gave2.topology_enhancement import enhance_av_probability_map  # noqa: E402


def process_directory(
    input_dir: Path,
    roi_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(glob.glob(str(input_dir / "*.png")))

    print(f"[+] Enhancing topology for {len(input_files)} maps in {input_dir.name}...")

    for path in input_files:
        filename = Path(path).name
        roi_path = roi_dir / filename

        img = np.array(Image.open(path).convert("RGB"))
        roi = np.array(Image.open(roi_path).convert("L")) > 0

        enhanced = enhance_av_probability_map(img, roi, threshold=0.35, max_gap_distance=6.0)

        Image.fromarray(enhanced, mode="RGB").save(output_dir / filename)

    print(f"[✓] Enhanced maps saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 & 2 Topology Enhancement")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset" / "GAVE2_preliminary")
    args = parser.parse_args()

    roi_dir = args.dataset_root / "validation" / "masks"

    # 1. Enhance Task 1
    t1_in = PROJECT_ROOT / "artifacts" / "task1_rrwnet" / "validation" / "rr10_c3"
    t1_out = PROJECT_ROOT / "artifacts" / "task1_enhanced"
    process_directory(t1_in, roi_dir, t1_out)

    # 2. Enhance Task 2
    t2_in = PROJECT_ROOT / "artifacts" / "task2" / "top3_hybrids" / "leader40_rr10c3_60"
    t2_out = PROJECT_ROOT / "artifacts" / "task2_enhanced"
    process_directory(t2_in, roi_dir, t2_out)


if __name__ == "__main__":
    main()
