#!/usr/bin/env python3
"""Apply the frozen Task 1 topology repair to validation probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gave2.data_index import Task, Task1Record, build_records  # noqa: E402
from gave2.evaluation import save_probability_map  # noqa: E402
from gave2.task1_postprocess import (  # noqa: E402
    Task1PostprocessConfig,
    postprocess_task1_probability,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path,
        default=PROJECT_ROOT / "dataset/GAVE2_preliminary",
    )
    parser.add_argument(
        "--split",
        choices=("training", "validation"),
        default="validation",
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=PROJECT_ROOT / "artifacts/submission_rehearsal/derived/task1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-threshold", type=float, default=0.04)
    parser.add_argument("--seed-threshold", type=float, default=0.50)
    parser.add_argument("--closing-radius", type=int, default=6)
    parser.add_argument("--exclusive-classes", action="store_true")
    parser.add_argument("--exclusive-smoothing-sigma", type=float, default=0.0)
    args = parser.parse_args()

    config = Task1PostprocessConfig(
        low_threshold=args.low_threshold,
        seed_threshold=args.seed_threshold,
        closing_radius=args.closing_radius,
        exclusive_classes=args.exclusive_classes,
        exclusive_smoothing_sigma=args.exclusive_smoothing_sigma,
    )
    records = [
        record
        for record in build_records(
            args.dataset_root.resolve(), args.split, Task.TASK1
        )
        if isinstance(record, Task1Record)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{record.case_id}.png" for record in records}
    stale = [path for path in args.output_dir.glob("*.png") if path.name not in expected_names]
    if stale:
        raise ValueError(f"Output directory contains stale PNGs: {[p.name for p in stale]}")

    files: dict[str, dict[str, str]] = {}
    for index, record in enumerate(records, start=1):
        source_path = args.input_dir / f"{record.case_id}.png"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32) / 255.0
        roi = np.asarray(Image.open(record.roi).convert("L")) > 0
        repaired = postprocess_task1_probability(source, roi, config)
        output_path = args.output_dir / source_path.name
        save_probability_map(repaired, output_path)
        files[source_path.name] = {
            "input_sha256": sha256(source_path),
            "output_sha256": sha256(output_path),
        }
        print(f"[{index}/{len(records)}] {record.case_id}: repaired", flush=True)

    report = {
        "status": "passed",
        "split": args.split,
        "case_count": len(records),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "config": asdict(config),
        "files": files,
    }
    report_path = args.output_dir / "postprocess_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
