"""CFP-only data pipeline for GAVE2 Task 1 artery/vein segmentation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data_index import Task1Record, assert_task1_isolation
from .evaluation import decode_av_label


IMAGE_MEAN = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
IMAGE_STD = np.asarray((0.25, 0.25, 0.25), dtype=np.float32)


def read_fold_assignments(path: Path | str) -> dict[str, int]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) < {"case_id", "fold"}:
        raise ValueError(f"Invalid fold manifest: {path}")
    assignments = {row["case_id"]: int(row["fold"]) for row in rows}
    if len(assignments) != len(rows):
        raise ValueError(f"Duplicate case IDs in fold manifest: {path}")
    return assignments


def split_records_for_fold(
    records: Sequence[Task1Record],
    assignments: dict[str, int],
    fold: int,
) -> tuple[list[Task1Record], list[Task1Record]]:
    assert_task1_isolation(records)
    record_ids = {record.case_id for record in records}
    if record_ids != set(assignments):
        missing = sorted(record_ids - set(assignments))
        extra = sorted(set(assignments) - record_ids)
        raise ValueError(f"Manifest mismatch: missing={missing}, extra={extra}")
    train = [record for record in records if assignments[record.case_id] != fold]
    validation = [record for record in records if assignments[record.case_id] == fold]
    if not train or not validation:
        raise ValueError(f"Fold {fold} produced an empty split")
    return train, validation


def normalize_cfp(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - IMAGE_MEAN) / IMAGE_STD


def load_full_case(
    record: Task1Record,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Load normalized CFP plus raw label and ROI for full-image validation."""

    if record.av_label is None:
        raise ValueError(f"Training label is missing for {record.case_id}")
    image = np.asarray(Image.open(record.cfp).convert("RGB"))
    label = np.asarray(Image.open(record.av_label).convert("RGBA"))
    roi = np.asarray(Image.open(record.roi).convert("L")) > 0
    tensor = torch.from_numpy(normalize_cfp(image).transpose(2, 0, 1)).float()
    return tensor, label, roi


def load_validation_case(
    record: Task1Record,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load normalized CFP and ROI for an unlabeled validation case."""

    if record.av_label is not None:
        raise ValueError(f"Validation record unexpectedly has a label: {record.case_id}")
    image = np.asarray(Image.open(record.cfp).convert("RGB"))
    roi = np.asarray(Image.open(record.roi).convert("L")) > 0
    if image.shape[:2] != roi.shape:
        raise ValueError(
            f"Image/ROI shape mismatch for {record.case_id}: "
            f"{image.shape[:2]} versus {roi.shape}"
        )
    tensor = torch.from_numpy(normalize_cfp(image).transpose(2, 0, 1)).float()
    return tensor, roi


class Task1PatchDataset(Dataset[dict[str, object]]):
    """Deterministic vessel-aware patch sampler that cannot expose FFA inputs."""

    def __init__(
        self,
        records: Sequence[Task1Record],
        *,
        patch_size: int = 512,
        patches_per_case: int = 8,
        augment: bool = True,
        vessel_sampling_probability: float = 0.75,
        seed: int = 20260719,
    ) -> None:
        if not records:
            raise ValueError("Task1PatchDataset requires at least one record")
        if patch_size <= 0 or patches_per_case <= 0:
            raise ValueError("patch_size and patches_per_case must be positive")
        if not 0.0 <= vessel_sampling_probability <= 1.0:
            raise ValueError("vessel_sampling_probability must lie in [0, 1]")
        assert_task1_isolation(records)
        if any(record.av_label is None for record in records):
            raise ValueError("Every patch-training record must have an AV label")
        self.records = list(records)
        self.patch_size = patch_size
        self.patches_per_case = patches_per_case
        self.augment = augment
        self.vessel_sampling_probability = vessel_sampling_probability
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_case

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + self.epoch * len(self) + index)

    @staticmethod
    def _pad(
        image: np.ndarray,
        target: np.ndarray,
        roi: np.ndarray,
        patch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = roi.shape
        pad_height = max(0, patch_size - height)
        pad_width = max(0, patch_size - width)
        if not pad_height and not pad_width:
            return image, target, roi
        image = np.pad(
            image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="reflect",
        )
        target = np.pad(
            target,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
        )
        roi = np.pad(
            roi,
            ((0, pad_height), (0, pad_width)),
            mode="constant",
        )
        return image, target, roi

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[(index // self.patches_per_case) % len(self.records)]
        rng = self._rng(index)
        image = np.asarray(Image.open(record.cfp).convert("RGB"))
        assert record.av_label is not None
        label = np.asarray(Image.open(record.av_label).convert("RGBA"))
        target = decode_av_label(label)
        roi = np.asarray(Image.open(record.roi).convert("L")) > 0
        image, target, roi = self._pad(image, target, roi, self.patch_size)

        candidates = np.argwhere(roi)
        vessel_candidates = np.argwhere((target[..., 1] > 0.5) & roi)
        use_vessel = (
            len(vessel_candidates) > 0
            and rng.random() < self.vessel_sampling_probability
        )
        sample_from = vessel_candidates if use_vessel else candidates
        if len(sample_from):
            center_y, center_x = sample_from[rng.randrange(len(sample_from))]
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
        image = image[top:bottom, left:right].copy()
        target = target[top:bottom, left:right].copy()
        roi = roi[top:bottom, left:right].copy()

        if self.augment:
            if rng.random() < 0.5:
                image, target, roi = (
                    np.flip(image, axis=1),
                    np.flip(target, axis=1),
                    np.flip(roi, axis=1),
                )
            if rng.random() < 0.5:
                image, target, roi = (
                    np.flip(image, axis=0),
                    np.flip(target, axis=0),
                    np.flip(roi, axis=0),
                )
            rotations = rng.randrange(4)
            if rotations:
                image = np.rot90(image, rotations)
                target = np.rot90(target, rotations)
                roi = np.rot90(roi, rotations)
            image_float = image.astype(np.float32) / 255.0
            contrast = rng.uniform(0.9, 1.1)
            brightness = rng.uniform(-0.05, 0.05)
            image = np.clip((image_float - 0.5) * contrast + 0.5 + brightness, 0, 1)
            normalized = (image - IMAGE_MEAN) / IMAGE_STD
        else:
            normalized = normalize_cfp(image)

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(normalized.transpose(2, 0, 1))
            ).float(),
            "target": torch.from_numpy(
                np.ascontiguousarray(target.transpose(2, 0, 1))
            ).float(),
            "roi": torch.from_numpy(np.ascontiguousarray(roi[None])).float(),
            "case_id": record.case_id,
            "sampled_vessel": use_vessel,
        }
