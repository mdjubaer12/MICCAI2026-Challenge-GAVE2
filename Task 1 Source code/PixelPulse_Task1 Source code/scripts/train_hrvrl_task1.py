#!/usr/bin/env python3
"""Fold-safe CFP-only fine-tuning of HRVRL PGNet for GAVE2 Task 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_hrvrl_task1 import load_model, predict_image  # noqa: E402
from gave2.data_index import (  # noqa: E402
    Task,
    Task1Record,
    assert_task1_isolation,
    build_records,
)
from gave2.evaluation import decode_av_label, evaluate_task1_arrays  # noqa: E402
from gave2.task1_data import (  # noqa: E402
    read_fold_assignments,
    split_records_for_fold,
)
from gave2.task1_loss import task1_segmentation_loss  # noqa: E402


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune HRVRL PGNet on one leakage-safe GAVE2 Task 1 fold"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "GAVE2_preliminary",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifests" / "training_folds.csv",
    )
    parser.add_argument(
        "--hrvrl-root",
        type=Path,
        default=PROJECT_ROOT / "external" / "HRVRL",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "external"
        / "HRVRL"
        / "weights"
        / "G_pretrain.pkl",
    )
    parser.add_argument(
        "--initial-finetuned-checkpoint",
        type=Path,
        help=(
            "Optional fold-matched checkpoint produced by this script. Load only "
            "its model state, then start a new optimizer/scheduler and artifact "
            "directory for a controlled continuation ablation."
        ),
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "task1_hrvrl"
        / "zero_shot_full"
        / "predictions",
        help="Full-image zero-shot probability PNGs; only training-fold cases are read",
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--patches-per-case", type=int, default=8)
    parser.add_argument("--vessel-sampling-probability", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print training progress every N micro-batches",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)
    parser.add_argument(
        "--encoder-unfreeze-blocks",
        type=int,
        default=2,
        help="Number of final top-level sn_unet modules to unfreeze after warm-up",
    )
    parser.add_argument(
        "--channel-weights",
        type=float,
        nargs=3,
        default=(1.25, 0.75, 1.0),
        metavar=("ARTERY", "VESSEL", "VEIN"),
    )
    parser.add_argument("--dice-loss-weight", type=float, default=0.55)
    parser.add_argument("--focal-tversky-weight", type=float, default=0.15)
    parser.add_argument("--cldice-weight", type=float, default=0.0)
    parser.add_argument("--skeleton-iterations", type=int, default=5)
    parser.add_argument("--endpoint-connectivity-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-radius", type=int, default=4)
    parser.add_argument("--endpoint-boost", type=float, default=2.0)
    parser.add_argument("--endpoint-focal-gamma", type=float, default=2.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.35)
    parser.add_argument("--tversky-beta", type=float, default=0.65)
    parser.add_argument("--tversky-gamma", type=float, default=0.75)
    parser.add_argument("--teacher-weight", type=float, default=0.10)
    parser.add_argument(
        "--teacher-channel-weights",
        type=float,
        nargs=3,
        default=(0.25, 1.0, 1.0),
        metavar=("ARTERY", "VESSEL", "VEIN"),
    )
    parser.add_argument("--hierarchy-weight", type=float, default=0.02)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--validation-stride", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--validation-threshold", type=float, default=0.45)
    parser.add_argument(
        "--selection-weights",
        type=float,
        nargs=3,
        default=(0.50, 0.15, 0.35),
        metavar=("ARTERY", "VESSEL", "VEIN"),
    )
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-validation-cases", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--minimum-checkpoint-coverage", type=float, default=0.95)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _normalized_chw(image: np.ndarray) -> torch.Tensor:
    array = image.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


class HRVRLTask1PatchDataset(Dataset[dict[str, object]]):
    """Aligned local/context patches with optional zero-shot soft targets."""

    def __init__(
        self,
        records: Sequence[Task1Record],
        *,
        teacher_root: Path | None,
        patch_size: int,
        context_scale: float,
        patches_per_case: int,
        vessel_sampling_probability: float,
        augment: bool,
        seed: int,
    ) -> None:
        if not records:
            raise ValueError("At least one Task 1 record is required")
        assert_task1_isolation(records)
        if any(record.av_label is None for record in records):
            raise ValueError("Fine-tuning records must contain AV labels")
        if patch_size <= 0 or patches_per_case <= 0:
            raise ValueError("Patch parameters must be positive")
        context_size = int(round(patch_size * context_scale))
        if context_size < patch_size or (context_size - patch_size) % 2:
            raise ValueError(
                "context-scale must produce an even, integer margin around the patch"
            )
        if not 0.0 <= vessel_sampling_probability <= 1.0:
            raise ValueError("vessel-sampling-probability must lie in [0, 1]")
        if teacher_root is not None:
            missing = [
                record.case_id
                for record in records
                if not (teacher_root / f"{record.case_id}.png").is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing zero-shot teacher predictions: {missing}"
                )
        self.records = list(records)
        self.teacher_root = teacher_root
        self.patch_size = patch_size
        self.context_size = context_size
        self.margin = (context_size - patch_size) // 2
        self.patches_per_case = patches_per_case
        self.vessel_sampling_probability = vessel_sampling_probability
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_case

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + self.epoch * len(self) + index)

    @staticmethod
    def _pad_to_patch(
        image: np.ndarray,
        target: np.ndarray,
        roi: np.ndarray,
        teacher: np.ndarray | None,
        patch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        pad_h = max(0, patch_size - image.shape[0])
        pad_w = max(0, patch_size - image.shape[1])
        if not pad_h and not pad_w:
            return image, target, roi, teacher
        image = np.pad(
            image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
        )
        target = np.pad(
            target, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
        )
        roi = np.pad(roi, ((0, pad_h), (0, pad_w)), mode="constant")
        if teacher is not None:
            teacher = np.pad(
                teacher, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
            )
        return image, target, roi, teacher

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[(index // self.patches_per_case) % len(self.records)]
        rng = self._rng(index)
        image = np.asarray(Image.open(record.cfp).convert("RGB"))
        assert record.av_label is not None
        target = decode_av_label(
            np.asarray(Image.open(record.av_label).convert("RGB"))
        )
        roi = np.asarray(Image.open(record.roi).convert("L")) > 0
        teacher = None
        if self.teacher_root is not None:
            teacher = (
                np.asarray(
                    Image.open(self.teacher_root / f"{record.case_id}.png").convert(
                        "RGB"
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
        image, target, roi, teacher = self._pad_to_patch(
            image, target, roi, teacher, self.patch_size
        )

        vessel_points = np.argwhere((target[..., 1] > 0.5) & roi)
        roi_points = np.argwhere(roi)
        sampled_vessel = (
            len(vessel_points) > 0
            and rng.random() < self.vessel_sampling_probability
        )
        candidates = vessel_points if sampled_vessel else roi_points
        if len(candidates):
            center_y, center_x = candidates[rng.randrange(len(candidates))]
        else:
            center_y = rng.randrange(image.shape[0])
            center_x = rng.randrange(image.shape[1])
        top = int(
            np.clip(
                center_y - self.patch_size // 2,
                0,
                image.shape[0] - self.patch_size,
            )
        )
        left = int(
            np.clip(
                center_x - self.patch_size // 2,
                0,
                image.shape[1] - self.patch_size,
            )
        )
        bottom = top + self.patch_size
        right = left + self.patch_size
        local = image[top:bottom, left:right].copy()
        target_patch = target[top:bottom, left:right].copy()
        roi_patch = roi[top:bottom, left:right].copy()
        teacher_patch = (
            teacher[top:bottom, left:right].copy() if teacher is not None else None
        )

        padded_image = np.pad(
            image,
            ((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
            mode="constant",
        )
        context = padded_image[
            top : top + self.context_size,
            left : left + self.context_size,
        ]
        context = cv2.resize(
            context,
            (self.patch_size, self.patch_size),
            interpolation=cv2.INTER_LINEAR,
        )

        if self.augment:
            arrays: list[np.ndarray] = [local, context, target_patch, roi_patch]
            if teacher_patch is not None:
                arrays.append(teacher_patch)
            if rng.random() < 0.5:
                arrays = [np.flip(array, axis=1) for array in arrays]
            if rng.random() < 0.5:
                arrays = [np.flip(array, axis=0) for array in arrays]
            rotations = rng.randrange(4)
            if rotations:
                arrays = [np.rot90(array, rotations) for array in arrays]
            local, context, target_patch, roi_patch = arrays[:4]
            if teacher_patch is not None:
                teacher_patch = arrays[4]
            contrast = rng.uniform(0.92, 1.08)
            brightness = rng.uniform(-0.04, 0.04)
            local_float = local.astype(np.float32) / 255.0
            context_float = context.astype(np.float32) / 255.0
            local = np.clip(
                (local_float - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0
            )
            context = np.clip(
                (context_float - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0
            )
            local = np.rint(local * 255.0).astype(np.uint8)
            context = np.rint(context * 255.0).astype(np.uint8)

        result: dict[str, object] = {
            "local": _normalized_chw(np.ascontiguousarray(local)),
            "context": _normalized_chw(np.ascontiguousarray(context)),
            "target": torch.from_numpy(
                np.ascontiguousarray(target_patch.transpose(2, 0, 1))
            ).float(),
            "roi": torch.from_numpy(
                np.ascontiguousarray(roi_patch[None])
            ).float(),
            "case_id": record.case_id,
            "sampled_vessel": sampled_vessel,
        }
        if teacher_patch is not None:
            result["teacher"] = torch.from_numpy(
                np.ascontiguousarray(teacher_patch.transpose(2, 0, 1))
            ).float()
        return result


def _set_finetuning_state(
    model: torch.nn.Module,
    *,
    epoch: int,
    freeze_encoder_epochs: int,
    encoder_unfreeze_blocks: int,
) -> list[str]:
    always_frozen = ("base_layers_global_momentum", "cenBlock", "bn_out")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not any(token in name for token in always_frozen)

    encoder_modules = list(model.sn_unet.children())
    for parameter in model.sn_unet.parameters():
        parameter.requires_grad = False
    unfrozen_names: list[str] = []
    if epoch >= freeze_encoder_epochs and encoder_unfreeze_blocks:
        start = max(0, len(encoder_modules) - encoder_unfreeze_blocks)
        for index, module in enumerate(encoder_modules[start:], start=start):
            for parameter in module.parameters():
                parameter.requires_grad = True
            unfrozen_names.append(f"sn_unet.{index}")
    model.base_layers_global_momentum.eval()
    for name in ("cenBlock1", "cenBlockMid", "cenBlockFinal", "bn_out"):
        module = getattr(model, name, None)
        if module is not None:
            module.eval()
    return unfrozen_names


def _distillation_loss(
    probabilities: torch.Tensor,
    teacher: torch.Tensor,
    roi: torch.Tensor,
    channel_weights: Sequence[float],
) -> torch.Tensor:
    mask = roi.float()
    squared_error = (probabilities.float() - teacher.float()).square() * mask
    denominator = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
    per_channel = squared_error.sum(dim=(0, 2, 3)) / denominator
    weights = probabilities.new_tensor(channel_weights, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-8)
    return (per_channel * weights).sum()


def _hierarchy_loss(
    probabilities: torch.Tensor,
    roi: torch.Tensor,
) -> torch.Tensor:
    vessel = probabilities[:, 1:2]
    violations = torch.relu(probabilities[:, 0:1] - vessel) + torch.relu(
        probabilities[:, 2:3] - vessel
    )
    mask = roi.float()
    return (violations.float() * mask).sum() / (
        2.0 * mask.sum().clamp_min(1.0)
    )


def _validation(
    model: torch.nn.Module,
    records: Sequence[Task1Record],
    *,
    args: argparse.Namespace,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    selected = list(records)
    if args.max_validation_cases is not None:
        selected = selected[: args.max_validation_cases]
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    rois: list[np.ndarray] = []
    model.eval()
    for record in selected:
        image = np.asarray(Image.open(record.cfp).convert("RGB"))
        assert record.av_label is not None
        labels.append(np.asarray(Image.open(record.av_label).convert("RGB")))
        roi = np.asarray(Image.open(record.roi).convert("L")) > 0
        rois.append(roi)
        predictions.append(
            predict_image(
                model,
                image,
                roi,
                patch_size=args.patch_size,
                stride=args.validation_stride,
                batch_size=args.validation_batch_size,
                device=device,
                amp=amp,
                use_global_semantic=True,
            )
        )
    metrics = evaluate_task1_arrays(
        labels,
        predictions,
        rois,
        threshold=args.validation_threshold,
        topology_paths=0,
    )
    weights = np.asarray(args.selection_weights, dtype=np.float64)
    weights /= weights.sum()
    selection_score = float(
        weights
        @ np.asarray(
            [metrics.artery.dice, metrics.vessel_dice, metrics.vein.dice]
        )
    )
    return {
        "artery_dice": metrics.artery.dice,
        "artery_sensitivity": metrics.artery.sensitivity,
        "artery_specificity": metrics.artery.specificity,
        "vessel_dice": metrics.vessel_dice,
        "vein_dice": metrics.vein.dice,
        "vein_sensitivity": metrics.vein.sensitivity,
        "vein_specificity": metrics.vein.specificity,
        "mean_av_dice": (metrics.artery.dice + metrics.vein.dice) / 2.0,
        "selection_score": selection_score,
        "case_count": float(len(selected)),
    }


def _cached_zero_shot_validation(
    records: Sequence[Task1Record],
    teacher_root: Path,
    threshold: float,
) -> dict[str, float]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    rois: list[np.ndarray] = []
    for record in records:
        assert record.av_label is not None
        labels.append(np.asarray(Image.open(record.av_label).convert("RGB")))
        predictions.append(
            np.asarray(
                Image.open(teacher_root / f"{record.case_id}.png").convert("RGB")
            )
        )
        rois.append(np.asarray(Image.open(record.roi).convert("L")))
    metrics = evaluate_task1_arrays(
        labels, predictions, rois, threshold=threshold, topology_paths=0
    )
    return {
        "artery_dice": metrics.artery.dice,
        "vessel_dice": metrics.vessel_dice,
        "vein_dice": metrics.vein.dice,
        "mean_av_dice": (metrics.artery.dice + metrics.vein.dice) / 2.0,
    }


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _atomic_torch_save(payload: object, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.patches_per_case = 1
        args.batch_size = 1
        args.accumulation_steps = 1
        args.workers = 0
        args.max_train_steps = args.max_train_steps or 1
        args.max_validation_cases = args.max_validation_cases or 1
        args.validation_every = 1
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.fold not in range(5):
        raise ValueError("fold must be in [0, 4]")
    if min(
        args.epochs,
        args.batch_size,
        args.accumulation_steps,
        args.validation_every,
        args.validation_stride,
        args.log_every,
    ) <= 0:
        raise ValueError("Epoch, batch, accumulation, and validation values must be positive")
    if args.validation_stride > args.patch_size:
        raise ValueError("validation-stride cannot exceed patch-size")
    if args.encoder_unfreeze_blocks < 0:
        raise ValueError("encoder-unfreeze-blocks cannot be negative")
    if any(value < 0 for value in args.channel_weights):
        raise ValueError("channel-weights must be non-negative")
    if args.teacher_weight < 0 or args.hierarchy_weight < 0:
        raise ValueError("Auxiliary loss weights cannot be negative")
    if (
        args.dice_loss_weight
        + args.focal_tversky_weight
        + args.cldice_weight
        + args.endpoint_connectivity_weight
        > 1.0
    ):
        raise ValueError("Supervised loss component weights cannot exceed one")
    if min(
        args.dice_loss_weight,
        args.focal_tversky_weight,
        args.cldice_weight,
        args.endpoint_connectivity_weight,
    ) < 0:
        raise ValueError("Supervised loss component weights cannot be negative")
    if args.skeleton_iterations < 1:
        raise ValueError("skeleton-iterations must be positive")
    if not 0 <= args.endpoint_radius <= 16:
        raise ValueError("endpoint-radius must be in [0, 16]")
    if args.endpoint_boost < 0:
        raise ValueError("endpoint-boost must be non-negative")
    if args.endpoint_focal_gamma <= 0:
        raise ValueError("endpoint-focal-gamma must be positive")
    if args.resume and args.initial_finetuned_checkpoint:
        raise ValueError(
            "--resume and --initial-finetuned-checkpoint are mutually exclusive"
        )

    seed_everything(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    output_dir = (
        args.output_dir
        or (args.resume.parent if args.resume else None)
        or (
            PROJECT_ROOT
            / "artifacts"
            / "task1_hrvrl"
            / "finetune_precision_teacher"
            / f"fold_{args.fold}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(
        (output_dir / name).exists()
        for name in ("history.jsonl", "latest.pt", "best.pt")
    ):
        raise FileExistsError(
            f"Run artifacts already exist in {output_dir}; use --resume or a new directory"
        )

    all_records = build_records(args.dataset_root, "training", Task.TASK1)
    records = [
        record for record in all_records if isinstance(record, Task1Record)
    ]
    train_records, validation_records = split_records_for_fold(
        records, read_fold_assignments(args.manifest), args.fold
    )
    teacher_root = args.teacher_root if args.teacher_weight > 0 else None
    dataset = HRVRLTask1PatchDataset(
        train_records,
        teacher_root=teacher_root,
        patch_size=args.patch_size,
        context_scale=args.context_scale,
        patches_per_case=args.patches_per_case,
        vessel_sampling_probability=args.vessel_sampling_probability,
        augment=True,
        seed=args.seed,
    )
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        generator=generator,
    )

    model, pretrained_metadata = load_model(
        args.hrvrl_root,
        args.checkpoint,
        device,
        args.minimum_checkpoint_coverage,
    )
    initial_finetuned_sha256: str | None = None
    initial_finetuned_metadata: dict[str, object] | None = None
    if args.initial_finetuned_checkpoint:
        initial_path = args.initial_finetuned_checkpoint.resolve()
        initial_checkpoint = torch.load(initial_path, map_location=device)
        if not isinstance(initial_checkpoint, dict) or "model" not in initial_checkpoint:
            raise ValueError(
                "Initial fine-tuned checkpoint must contain a model state dictionary"
            )
        source_args = initial_checkpoint.get("args")
        if not isinstance(source_args, dict) or "fold" not in source_args:
            raise ValueError("Initial fine-tuned checkpoint is missing its fold")
        source_fold = int(source_args["fold"])
        if source_fold != args.fold:
            raise ValueError(
                f"Initial checkpoint fold {source_fold} != requested fold {args.fold}"
            )
        if initial_checkpoint.get("task1_ffa_used") is not False:
            raise ValueError(
                "Initial fine-tuned checkpoint does not prove Task 1 FFA isolation"
            )
        source_manifest_sha256 = initial_checkpoint.get("manifest_sha256")
        current_manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        if source_manifest_sha256 != current_manifest_sha256:
            raise ValueError(
                "Initial fine-tuned checkpoint manifest does not match this run"
            )
        model.load_state_dict(initial_checkpoint["model"], strict=True)
        initial_finetuned_sha256 = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        initial_finetuned_metadata = {
            "path": str(initial_path),
            "sha256": initial_finetuned_sha256,
            "fold": source_fold,
            "source_epoch": int(initial_checkpoint.get("epoch", -1)),
            "source_best_score": float(
                initial_checkpoint.get("best_score", float("nan"))
            ),
            "source_manifest_sha256": source_manifest_sha256,
        }
        print(
            json.dumps(
                {"initial_finetuned_checkpoint": initial_finetuned_metadata},
                sort_keys=True,
            ),
            flush=True,
        )
    encoder_ids = {id(parameter) for parameter in model.sn_unet.parameters()}
    encoder_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) in encoder_ids
    ]
    decoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if id(parameter) not in encoder_ids
        and not name.startswith("base_layers_global_momentum.")
        and not name.startswith("cenBlock")
        and not name.startswith("bn_out.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": args.encoder_learning_rate,
                "name": "encoder",
            },
            {
                "params": decoder_parameters,
                "lr": args.learning_rate,
                "name": "decoder_fusion",
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=1e-7,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch = 0
    global_micro_steps = 0
    optimizer_steps = 0
    best_score = -1.0
    validations_without_improvement = 0
    history: list[dict[str, object]] = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_fold = int(checkpoint["args"]["fold"])
        if checkpoint_fold != args.fold:
            raise ValueError(
                f"Resume checkpoint fold {checkpoint_fold} != requested fold {args.fold}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_micro_steps = int(checkpoint.get("global_micro_steps", 0))
        optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        best_score = float(checkpoint.get("best_score", -1.0))
        validations_without_improvement = int(
            checkpoint.get("validations_without_improvement", 0)
        )
        history_path = output_dir / "history.jsonl"
        if history_path.is_file():
            history = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        print(
            json.dumps(
                {
                    "resumed_from": str(args.resume),
                    "start_epoch": start_epoch,
                    "best_score": best_score,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    zero_shot_fold = _cached_zero_shot_validation(
        validation_records, args.teacher_root, args.validation_threshold
    )
    print(
        json.dumps({"cached_zero_shot_validation": zero_shot_fold}, sort_keys=True),
        flush=True,
    )

    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stopped_early = False
    last_validation: dict[str, float] | None = None

    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        generator.manual_seed(args.seed + epoch)
        model.train()
        unfrozen_encoder_modules = _set_finetuning_state(
            model,
            epoch=epoch,
            freeze_encoder_epochs=args.freeze_encoder_epochs,
            encoder_unfreeze_blocks=args.encoder_unfreeze_blocks,
        )
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: dict[str, list[float]] = {
            "total": [],
            "supervised": [],
            "cldice": [],
            "endpoint_connectivity": [],
            "distillation": [],
            "hierarchy": [],
        }
        updates_this_epoch = 0
        for batch_index, batch in enumerate(loader):
            local = batch["local"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            roi = batch["roi"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                logits, _ = model(local, context)
                supervised_result = task1_segmentation_loss(
                    logits,
                    target,
                    roi,
                    channel_weights=tuple(args.channel_weights),
                    dice_weight=args.dice_loss_weight,
                    focal_tversky_weight=args.focal_tversky_weight,
                    cldice_weight=args.cldice_weight,
                    endpoint_connectivity_weight=(
                        args.endpoint_connectivity_weight
                    ),
                    tversky_alpha=args.tversky_alpha,
                    tversky_beta=args.tversky_beta,
                    tversky_gamma=args.tversky_gamma,
                    skeleton_iterations=args.skeleton_iterations,
                    endpoint_radius=args.endpoint_radius,
                    endpoint_boost=args.endpoint_boost,
                    endpoint_focal_gamma=args.endpoint_focal_gamma,
                )
                supervised = supervised_result["loss"]
                cldice = supervised_result["cldice_loss"]
                endpoint_connectivity = supervised_result[
                    "endpoint_connectivity_loss"
                ]
                probabilities = torch.sigmoid(logits)
                distillation = probabilities.new_zeros(())
                if args.teacher_weight > 0:
                    teacher = batch["teacher"].to(device, non_blocking=True)
                    distillation = _distillation_loss(
                        probabilities,
                        teacher,
                        roi,
                        args.teacher_channel_weights,
                    )
                hierarchy = _hierarchy_loss(probabilities, roi)
                total = (
                    supervised
                    + args.teacher_weight * distillation
                    + args.hierarchy_weight * hierarchy
                )
                scaled = total / args.accumulation_steps
            scaler.scale(scaled).backward()
            global_micro_steps += 1
            for name, value in (
                ("total", total),
                ("supervised", supervised),
                ("cldice", cldice),
                ("endpoint_connectivity", endpoint_connectivity),
                ("distillation", distillation),
                ("hierarchy", hierarchy),
            ):
                epoch_losses[name].append(float(value.detach().cpu()))
            if (
                batch_index == 0
                or (batch_index + 1) % args.log_every == 0
                or batch_index + 1 == len(loader)
            ):
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": epoch,
                            "batch": batch_index + 1,
                            "batches": len(loader),
                            "total_loss": epoch_losses["total"][-1],
                            "global_micro_steps": global_micro_steps,
                            "allocated_vram_mib": (
                                float(
                                    torch.cuda.memory_allocated(device) / 1024**2
                                )
                                if device.type == "cuda"
                                else 0.0
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            reached_limit = (
                args.max_train_steps is not None
                and global_micro_steps >= args.max_train_steps
            )
            update = (
                (batch_index + 1) % args.accumulation_steps == 0
                or batch_index + 1 == len(loader)
                or reached_limit
            )
            if update:
                scaler.unscale_(optimizer)
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                            and parameter.grad is not None
                        ],
                        args.gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                updates_this_epoch += 1
            if reached_limit:
                break

        should_validate = (
            (epoch + 1) % args.validation_every == 0
            or epoch + 1 == args.epochs
            or args.smoke_test
        )
        validation = None
        if should_validate:
            print(
                json.dumps(
                    {
                        "event": "validation_start",
                        "epoch": epoch,
                        "cases": (
                            min(
                                len(validation_records),
                                args.max_validation_cases,
                            )
                            if args.max_validation_cases is not None
                            else len(validation_records)
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            validation = _validation(
                model,
                validation_records,
                args=args,
                device=device,
                amp=amp,
            )
        improved = False
        if validation is not None:
            last_validation = validation
            improved = validation["selection_score"] > (
                best_score + args.minimum_delta
            )
            if improved:
                best_score = validation["selection_score"]
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            scheduler.step(validation["selection_score"])

        epoch_result: dict[str, object] = {
            "epoch": epoch,
            "losses": {
                name: float(np.mean(values)) if values else float("nan")
                for name, values in epoch_losses.items()
            },
            "global_micro_steps": global_micro_steps,
            "optimizer_steps": optimizer_steps,
            "updates_this_epoch": updates_this_epoch,
            "learning_rates": {
                str(group["name"]): float(group["lr"])
                for group in optimizer.param_groups
            },
            "unfrozen_encoder_modules": unfrozen_encoder_modules,
            "validation": validation,
            "best_selection_score": best_score,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, sort_keys=True), flush=True)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_result, sort_keys=True) + "\n")

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": _jsonable_args(args),
            "validation": validation,
            "last_validation": last_validation,
            "best_score": best_score,
            "global_micro_steps": global_micro_steps,
            "optimizer_steps": optimizer_steps,
            "validations_without_improvement": validations_without_improvement,
            "manifest_sha256": manifest_sha256,
            "pretrained_checkpoint_sha256": checkpoint_sha256,
            "pretrained_metadata": pretrained_metadata,
            "initial_finetuned_checkpoint": initial_finetuned_metadata,
            "task1_ffa_used": False,
        }
        _atomic_torch_save(payload, output_dir / "latest.pt")
        if improved:
            _atomic_torch_save(payload, output_dir / "best.pt")

        if (
            not args.smoke_test
            and args.patience > 0
            and validations_without_improvement >= args.patience
        ):
            stopped_early = True
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "best_selection_score": best_score,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break
        if (
            args.max_train_steps is not None
            and global_micro_steps >= args.max_train_steps
        ):
            break

    report = {
        "status": "passed",
        "mode": "smoke" if args.smoke_test else "train",
        "task1_ffa_used": False,
        "fold": args.fold,
        "train_cases": len(train_records),
        "validation_cases": len(validation_records),
        "cached_zero_shot_validation": zero_shot_fold,
        "best_selection_score": best_score,
        "last_validation": last_validation,
        "completed_epochs_this_invocation": max(0, len(history) - start_epoch),
        "stopped_early": stopped_early,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count_at_end": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "elapsed_seconds": time.monotonic() - started,
        "manifest_sha256": manifest_sha256,
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "initial_finetuned_checkpoint": initial_finetuned_metadata,
        "args": _jsonable_args(args),
    }
    (output_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
