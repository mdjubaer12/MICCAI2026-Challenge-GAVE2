#!/usr/bin/env python3
"""Validate Task 1 maps and create an organizer-style Task1 ZIP."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()

    roi_dir = args.dataset_root / "validation" / "masks"
    roi_paths = sorted(roi_dir.glob("g_*.png"))
    if len(roi_paths) != 50:
        raise ValueError(f"Expected 50 validation ROI masks, found {len(roi_paths)}")

    expected_names = {path.name for path in roi_paths}
    actual_names = {path.name for path in args.prediction_dir.glob("g_*.png")}
    if actual_names != expected_names:
        raise ValueError(
            f"Prediction set mismatch: missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    for roi_path in roi_paths:
        prediction_path = args.prediction_dir / roi_path.name
        with Image.open(prediction_path) as image:
            if image.mode != "RGB":
                raise ValueError(f"{prediction_path}: expected RGB, got {image.mode}")
            prediction = np.asarray(image)
        with Image.open(roi_path) as image:
            roi = np.asarray(image.convert("L")) > 0
        if prediction.shape != (*roi.shape, 3) or prediction.dtype != np.uint8:
            raise ValueError(f"{prediction_path}: invalid shape or dtype")
        if np.any(prediction[~roi] != 0):
            raise ValueError(f"{prediction_path}: nonzero output outside ROI")
        if np.any(prediction[..., 1] != np.maximum(prediction[..., 0], prediction[..., 2])):
            raise ValueError(f"{prediction_path}: vessel channel is not max(A,V)")

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    if args.output_zip.exists():
        raise FileExistsError(args.output_zip)
    with zipfile.ZipFile(
        args.output_zip,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in sorted(expected_names):
            archive.write(args.prediction_dir / name, arcname=f"Task1/{name}")

    print(f"Wrote validated Task 1 ZIP: {args.output_zip}")


if __name__ == "__main__":
    main()
