#!/usr/bin/env python3
"""Apply the official RRWNet stand-alone refiner to Task 1 probability maps.

GAVE probability maps use RGB=(artery, vessel, vein). RRWNet uses
RGB=(artery, vein, vessel). This adapter performs that conversion explicitly,
keeps the source vessel channel unchanged, reapplies the competition ROI, and
records enough provenance to reproduce every output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gave2.evaluation import evaluate_task1_arrays, save_probability_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT
        / "external"
        / "RRWNet"
        / "weights"
        / "rrwnet_RITE_refinement.pth",
    )
    parser.add_argument(
        "--rrwnet-root",
        type=Path,
        default=PROJECT_ROOT / "external" / "RRWNet",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "GAVE2_preliminary",
    )
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifests" / "training_folds.csv",
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--recursions",
        type=int,
        default=5,
        help=(
            "RRWNet recurrent-loop count. The official default of 5 produces "
            "six saved refinement stages: one initial pass plus five recursions."
        ),
    )
    parser.add_argument("--padding-multiple", type=int, default=32)
    parser.add_argument(
        "--minimal-padding",
        action="store_true",
        help="Do not add a full padding multiple when a dimension is divisible.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--topology-paths", type=int, default=0)
    parser.add_argument("--topology-seed", type=int, default=20260725)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gave_to_rrwnet(probability: np.ndarray) -> np.ndarray:
    """Convert A/Vessel/Vein to RRWNet's A/Vein/Vessel order."""

    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 probability map, received {array.shape}")
    return np.ascontiguousarray(array[..., (0, 2, 1)])


def rrwnet_to_gave(probability: np.ndarray) -> np.ndarray:
    """Convert RRWNet's A/Vein/Vessel to A/Vessel/Vein order."""

    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 probability map, received {array.shape}")
    return np.ascontiguousarray(array[..., (0, 2, 1)])


def unet_padding(
    height: int,
    width: int,
    *,
    multiple: int,
    always_pad: bool,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if min(height, width, multiple) <= 0:
        raise ValueError("Image dimensions and padding multiple must be positive")

    def axis_padding(size: int) -> tuple[int, int]:
        remainder = size % multiple
        total = multiple - remainder if remainder or always_pad else 0
        before = total // 2
        return before, total - before

    return axis_padding(height), axis_padding(width)


def crop_padding(
    array: np.ndarray,
    padding: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    (top, bottom), (left, right) = padding
    height_stop = array.shape[0] - bottom if bottom else array.shape[0]
    width_stop = array.shape[1] - right if right else array.shape[1]
    return array[top:height_stop, left:width_stop, ...]


def load_probability(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite source probability map: {path}")
    return np.clip(array, 0.0, 1.0)


def _load_fold_cases(path: Path, fold: int) -> set[str]:
    if fold not in range(5):
        raise ValueError("fold must be in [0, 4]")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) < {"case_id", "fold"}:
        raise ValueError(f"Invalid fold manifest: {path}")
    return {row["case_id"] for row in rows if int(row["fold"]) == fold}


def select_sources(args: argparse.Namespace) -> list[Path]:
    sources = sorted(args.input_dir.glob("*.png"))
    if not sources:
        raise FileNotFoundError(f"No PNG probability maps in {args.input_dir}")
    if args.fold is not None:
        fold_cases = _load_fold_cases(args.manifest, args.fold)
        sources = [path for path in sources if path.stem in fold_cases]
    if args.case_ids:
        requested = set(args.case_ids)
        sources = [path for path in sources if path.stem in requested]
        missing = requested - {path.stem for path in sources}
        if missing:
            raise FileNotFoundError(f"Requested cases not found: {sorted(missing)}")
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("max-cases must be positive")
        sources = sources[: args.max_cases]
    if not sources:
        raise ValueError("Case selection is empty")
    return sources


def load_rrwnet_class(rrwnet_root: Path) -> type[torch.nn.Module]:
    model_path = rrwnet_root / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing RRWNet model definition: {model_path}")
    spec = importlib.util.spec_from_file_location("official_rrwnet_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import RRWNet model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RRWNet


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metric_dict(
    labels: list[np.ndarray],
    predictions: list[np.ndarray],
    rois: list[np.ndarray],
    *,
    threshold: float,
    topology_paths: int,
    topology_seed: int,
) -> dict[str, Any] | None:
    if not labels:
        return None
    return evaluate_task1_arrays(
        labels,
        predictions,
        rois,
        threshold=threshold,
        topology_paths=topology_paths,
        topology_seed=topology_seed,
    ).to_dict()


def main() -> None:
    args = parse_args()
    if args.recursions < 0:
        raise ValueError("recursions must be non-negative")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if args.topology_paths < 0:
        raise ValueError("topology-paths must be non-negative")
    if args.amp and args.device != "cuda":
        raise ValueError("AMP is only supported with CUDA")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Missing RRWNet weights: {args.weights}")

    sources = select_sources(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_count = args.recursions + 1
    stage_directories = [
        args.output_dir / f"stage_{index:02d}" / "predictions"
        for index in range(1, stage_count + 1)
    ]
    for directory in stage_directories:
        directory.mkdir(parents=True)

    device = torch.device(args.device)
    model_class = load_rrwnet_class(args.rrwnet_root)
    model = model_class(iterations=args.recursions)
    state = torch.load(args.weights, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"RRWNet checkpoint mismatch: {incompatible}")
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stage_predictions: list[list[np.ndarray]] = [
        [] for _ in range(stage_count)
    ]
    labels: list[np.ndarray] = []
    rois: list[np.ndarray] = []
    source_predictions: list[np.ndarray] = []
    source_hashes: dict[str, str] = {}

    roi_root = args.dataset_root / args.split / "masks"
    label_root = args.dataset_root / args.split / "av"
    for case_index, source_path in enumerate(sources, start=1):
        case_id = source_path.stem
        roi_path = roi_root / f"{case_id}.png"
        if not roi_path.is_file():
            raise FileNotFoundError(f"Missing ROI for {case_id}: {roi_path}")
        with Image.open(roi_path) as image:
            roi = np.asarray(image.convert("L")) > 0
        source = load_probability(source_path)
        if source.shape[:2] != roi.shape:
            raise ValueError(
                f"{case_id}: probability {source.shape[:2]} != ROI {roi.shape}"
            )
        source *= roi[..., None]
        rrwnet_input = gave_to_rrwnet(source)
        padding = unet_padding(
            *roi.shape,
            multiple=args.padding_multiple,
            always_pad=not args.minimal_padding,
        )
        padded = np.pad(rrwnet_input, (*padding, (0, 0)), mode="constant")
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(
            device
        )

        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp,
        ):
            outputs = model.refine(tensor)
        if len(outputs) != stage_count:
            raise RuntimeError(
                f"Expected {stage_count} RRWNet outputs, received {len(outputs)}"
            )

        for stage_index, output in enumerate(outputs):
            rrwnet_probability = (
                output[0].float().cpu().numpy().transpose(1, 2, 0)
            )
            gave_probability = rrwnet_to_gave(
                crop_padding(rrwnet_probability, padding)
            )
            if gave_probability.shape != source.shape:
                raise RuntimeError(
                    f"{case_id}: refined shape {gave_probability.shape} "
                    f"!= source {source.shape}"
                )
            # The stand-alone RRWNet refiner is not allowed to alter the
            # source vessel-union probability.
            gave_probability[..., 1] = source[..., 1]
            gave_probability *= roi[..., None]
            gave_probability = np.clip(gave_probability, 0.0, 1.0)
            if not np.all(np.isfinite(gave_probability)):
                raise RuntimeError(f"Non-finite RRWNet output for {case_id}")
            save_probability_map(
                gave_probability,
                stage_directories[stage_index] / source_path.name,
            )
            stage_predictions[stage_index].append(gave_probability)

        label_path = label_root / f"{case_id}.png"
        if label_path.is_file():
            with Image.open(label_path) as image:
                labels.append(np.asarray(image.convert("RGB")))
        elif args.split == "training":
            raise FileNotFoundError(f"Missing training AV label: {label_path}")
        source_predictions.append(source)
        rois.append(roi)
        source_hashes[source_path.name] = sha256_file(source_path)
        print(
            json.dumps(
                {
                    "progress": f"{case_index}/{len(sources)}",
                    "case": case_id,
                    "stages": stage_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if labels and len(labels) != len(sources):
        raise RuntimeError("AV labels are available for only a subset of cases")
    metrics: dict[str, Any] = {
        "source": _metric_dict(
            labels,
            source_predictions,
            rois,
            threshold=args.threshold,
            topology_paths=args.topology_paths,
            topology_seed=args.topology_seed,
        )
    }
    for index, predictions in enumerate(stage_predictions, start=1):
        metrics[f"stage_{index:02d}"] = _metric_dict(
            labels,
            predictions,
            rois,
            threshold=args.threshold,
            topology_paths=args.topology_paths,
            topology_seed=args.topology_seed,
        )

    provenance: dict[str, Any] = {
        "status": "passed",
        "task1_ffa_used": False,
        "source_directory": str(args.input_dir.resolve()),
        "source_sha256": source_hashes,
        "case_ids": [path.stem for path in sources],
        "case_count": len(sources),
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256_file(args.weights),
        "rrwnet_root": str(args.rrwnet_root.resolve()),
        "rrwnet_commit": git_revision(args.rrwnet_root),
        "rrwnet_input_channels": ["artery", "vein", "vessel"],
        "gave_input_output_channels": ["artery", "vessel", "vein"],
        "source_vessel_channel_preserved": True,
        "recursions": args.recursions,
        "saved_refinement_stages": stage_count,
        "padding_multiple": args.padding_multiple,
        "official_always_pad_behavior": not args.minimal_padding,
        "device": str(device),
        "amp": bool(args.amp),
        "threshold": args.threshold,
        "topology_paths": args.topology_paths,
        "topology_seed": args.topology_seed,
        "torch_version": torch.__version__,
        "metrics": metrics,
    }
    if device.type == "cuda":
        provenance["cuda_device"] = torch.cuda.get_device_name(device)
        provenance["peak_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        provenance["peak_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    (args.output_dir / "report.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if device.type == "cuda":
        print(
            json.dumps(
                {
                    "peak_memory_allocated_gib": provenance[
                        "peak_memory_allocated_bytes"
                    ]
                    / (1024**3),
                    "peak_memory_reserved_gib": provenance[
                        "peak_memory_reserved_bytes"
                    ]
                    / (1024**3),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
