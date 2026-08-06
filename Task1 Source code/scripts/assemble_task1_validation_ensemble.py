#!/usr/bin/env python3
"""Assemble fold and family ensembles for Task 1 validation probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gave2.data_index import Task, Task1Record, build_records  # noqa: E402
from gave2.evaluation import save_probability_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "GAVE2_preliminary",
    )
    parser.add_argument(
        "--prediction-pattern",
        action="append",
        required=True,
        help=(
            "Fold-formatted validation prediction directory. Repeat for each "
            "model family."
        ),
    )
    parser.add_argument("--family-weights", type=float, nargs="+")
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--fold-weights", type=float, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalized_weights(
    count: int,
    values: list[float] | tuple[float, ...] | None,
    *,
    name: str,
) -> np.ndarray:
    if count <= 0:
        raise ValueError(f"At least one {name} is required")
    if values is None:
        return np.full(count, 1.0 / count, dtype=np.float64)
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (count,):
        raise ValueError(f"{name} weights must contain {count} values")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} weights must be finite and non-negative")
    if weights.sum() <= 0.0:
        raise ValueError(f"{name} weights must have a positive sum")
    return weights / weights.sum()


def prediction_directory(pattern: str, fold: int) -> Path:
    path = Path(pattern.format(fold=fold))
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.folds or len(set(args.folds)) != len(args.folds):
        raise ValueError("folds must be non-empty and unique")
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("folds must lie in [0, 4]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    patterns = list(args.prediction_pattern)
    family_weights = normalized_weights(
        len(patterns), args.family_weights, name="family"
    )
    fold_weights = normalized_weights(
        len(args.folds), args.fold_weights, name="fold"
    )

    records = [
        record
        for record in build_records(args.dataset_root, "validation", Task.TASK1)
        if isinstance(record, Task1Record)
    ]
    if len(records) != 50:
        raise ValueError(f"Expected 50 validation cases, received {len(records)}")
    args.output_dir.mkdir(parents=True)
    sources: dict[str, list[list[dict[str, str]]]] = {}
    outputs: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        family_probabilities: list[np.ndarray] = []
        family_sources: list[list[dict[str, str]]] = []
        for pattern in patterns:
            fold_probabilities: list[np.ndarray] = []
            fold_sources: list[dict[str, str]] = []
            for fold in args.folds:
                path = prediction_directory(pattern, fold) / f"{record.case_id}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as image:
                    fold_probabilities.append(
                        np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                    )
                fold_sources.append(
                    {"path": str(path.resolve()), "sha256": sha256_file(path)}
                )
            family_probabilities.append(
                np.average(
                    np.stack(fold_probabilities, axis=0),
                    axis=0,
                    weights=fold_weights,
                ).astype(np.float32)
            )
            family_sources.append(fold_sources)

        probability = np.average(
            np.stack(family_probabilities, axis=0),
            axis=0,
            weights=family_weights,
        ).astype(np.float32)
        with Image.open(record.roi) as image:
            roi = np.asarray(image.convert("L")) > 0
        if probability.shape != (*roi.shape, 3):
            raise ValueError(
                f"{record.case_id}: probability {probability.shape} "
                f"!= ROI {roi.shape}"
            )
        probability *= roi[..., None]
        output = args.output_dir / f"{record.case_id}.png"
        save_probability_map(np.clip(probability, 0.0, 1.0), output)
        sources[record.case_id] = family_sources
        outputs[record.case_id] = sha256_file(output)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(records)}",
                    "case": record.case_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report = {
        "status": "passed",
        "task1_ffa_used": False,
        "case_count": len(records),
        "prediction_patterns": patterns,
        "family_weights": family_weights.tolist(),
        "folds": args.folds,
        "fold_weights": fold_weights.tolist(),
        "output_dir": str(args.output_dir.resolve()),
        "source_files": sources,
        "output_sha256": outputs,
    }
    (args.output_dir / "assembly_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "source_files"},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
