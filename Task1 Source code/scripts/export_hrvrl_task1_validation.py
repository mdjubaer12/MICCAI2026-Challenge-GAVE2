#!/usr/bin/env python3
"""Resumably export one fine-tuned HRVRL fold on validation CFP images."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_hrvrl_task1 import load_model, predict_image  # noqa: E402
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
        "--hrvrl-root",
        type=Path,
        default=PROJECT_ROOT / "external" / "HRVRL",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "external"
        / "HRVRL"
        / "weights"
        / "G_pretrain.pkl",
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--minimum-checkpoint-coverage", type=float, default=0.95)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_existing(path: Path, shape: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            return image.mode == "RGB" and image.size == (shape[1], shape[0])
    except OSError:
        return False


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.fold not in range(5):
        raise ValueError("fold must lie in [0, 4]")
    if min(args.patch_size, args.stride, args.batch_size) <= 0:
        raise ValueError("Inference dimensions and batch size must be positive")
    if args.stride > args.patch_size:
        raise ValueError("stride cannot exceed patch-size")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    if int(checkpoint_args.get("fold", -1)) != args.fold:
        raise ValueError(
            f"Checkpoint fold {checkpoint_args.get('fold')} != requested fold {args.fold}"
        )
    if checkpoint.get("task1_ffa_used") is not False:
        raise ValueError("Checkpoint does not explicitly certify CFP-only Task 1 use")
    device = torch.device(args.device)
    model, model_metadata = load_model(
        args.hrvrl_root,
        args.pretrained_checkpoint,
        device,
        args.minimum_checkpoint_coverage,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records = [
        record
        for record in build_records(args.dataset_root, "validation", Task.TASK1)
        if isinstance(record, Task1Record)
    ]
    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    expected = {f"{record.case_id}.png" for record in records}
    stale = {
        path.name for path in predictions_dir.glob("*.png")
    } - expected
    if stale:
        raise ValueError(f"Output contains stale predictions: {sorted(stale)}")

    started = time.monotonic()
    written = 0
    skipped = 0
    for index, record in enumerate(records, start=1):
        image = np.asarray(Image.open(record.cfp).convert("RGB"))
        roi = np.asarray(Image.open(record.roi).convert("L")) > 0
        output = predictions_dir / f"{record.case_id}.png"
        if valid_existing(output, roi.shape):
            skipped += 1
            print(
                f"[{index}/{len(records)}] {record.case_id}: resume-skip",
                flush=True,
            )
            continue
        probability = predict_image(
            model,
            image,
            roi,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            amp=amp,
            use_global_semantic=True,
        )
        if (
            probability.shape != (*roi.shape, 3)
            or not np.all(np.isfinite(probability))
            or float(probability.min()) < 0.0
            or float(probability.max()) > 1.0
        ):
            raise RuntimeError(f"Invalid prediction for {record.case_id}")
        save_probability_map(probability, output)
        written += 1
        print(
            f"[{index}/{len(records)}] {record.case_id}: written",
            flush=True,
        )

    actual = {path.name for path in predictions_dir.glob("*.png")}
    if actual != expected:
        raise RuntimeError(
            f"Prediction set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    report = {
        "status": "passed",
        "task1_ffa_used": False,
        "fold": args.fold,
        "case_count": len(records),
        "written_this_run": written,
        "resume_skipped": skipped,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation": checkpoint.get("validation"),
        "pretrained_checkpoint": str(args.pretrained_checkpoint.resolve()),
        "pretrained_checkpoint_sha256": sha256(args.pretrained_checkpoint),
        "model": model_metadata,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "amp": amp,
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "inference_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
