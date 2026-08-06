"""Leakage-safe indexing for the local GAVE2 dataset.

Task 1 is intentionally represented by a separate record type that never
contains FFA paths. This makes accidental Task 1 multimodal leakage fail closed
at the data boundary instead of relying on training-loop discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping


class DatasetLayoutError(RuntimeError):
    """Raised when the on-disk dataset does not match the expected schema."""


class Task(str, Enum):
    TASK1 = "task1"
    TASK2 = "task2"
    TASK3 = "task3"


@dataclass(frozen=True)
class Task1Record:
    case_id: str
    cfp: Path
    roi: Path
    av_label: Path | None


@dataclass(frozen=True)
class MultimodalRecord:
    case_id: str
    cfp: Path
    ffa_early: Path
    ffa_late: Path
    roi: Path
    av_label: Path | None
    biomarker: Path | None


Record = Task1Record | MultimodalRecord

_IMAGE_SUFFIX = ".png"
_TARGET_SUFFIX = ".txt"


def _stems(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        raise DatasetLayoutError(f"Missing required directory: {directory}")
    return {path.stem for path in directory.glob(f"*{suffix}") if path.is_file()}


def _require_equal_case_sets(
    root: Path,
    modality_suffixes: Mapping[str, str],
) -> list[str]:
    by_modality = {
        modality: _stems(root / modality, suffix)
        for modality, suffix in modality_suffixes.items()
    }
    reference_name = next(iter(by_modality))
    reference = by_modality[reference_name]
    errors: list[str] = []
    for modality, cases in by_modality.items():
        missing = sorted(reference - cases)
        extra = sorted(cases - reference)
        if missing or extra:
            errors.append(
                f"{modality}: missing={missing or '[]'}, extra={extra or '[]'}"
            )
    if errors:
        raise DatasetLayoutError(
            f"Inconsistent case sets under {root}: " + "; ".join(errors)
        )
    if not reference:
        raise DatasetLayoutError(f"No cases found under {root}")
    return sorted(reference)


def build_records(dataset_root: Path | str, split: str, task: Task | str) -> list[Record]:
    """Build validated records for one split and task.

    Args:
        dataset_root: Directory containing ``training`` and ``validation``.
        split: Either ``training`` or ``validation``.
        task: ``task1``, ``task2``, or ``task3``.
    """

    root = Path(dataset_root).expanduser().resolve()
    try:
        parsed_task = task if isinstance(task, Task) else Task(task)
    except ValueError as error:
        raise DatasetLayoutError(f"Unknown task: {task!r}") from error
    if split not in {"training", "validation"}:
        raise DatasetLayoutError(f"Unknown split: {split!r}")

    split_root = root / split
    if parsed_task is Task.TASK1:
        modalities = {"images": _IMAGE_SUFFIX, "masks": _IMAGE_SUFFIX}
        if split == "training":
            modalities["av"] = _IMAGE_SUFFIX
        case_ids = _require_equal_case_sets(split_root, modalities)
        records: list[Record] = [
            Task1Record(
                case_id=case_id,
                cfp=split_root / "images" / f"{case_id}.png",
                roi=split_root / "masks" / f"{case_id}.png",
                av_label=(
                    split_root / "av" / f"{case_id}.png"
                    if split == "training"
                    else None
                ),
            )
            for case_id in case_ids
        ]
        assert_task1_isolation(records)
        return records

    modalities = {
        "images": _IMAGE_SUFFIX,
        "FFA_A": _IMAGE_SUFFIX,
        "FFA_AV": _IMAGE_SUFFIX,
        "masks": _IMAGE_SUFFIX,
    }
    if split == "training":
        modalities.update({"av": _IMAGE_SUFFIX, "biomarker": _TARGET_SUFFIX})
    case_ids = _require_equal_case_sets(split_root, modalities)
    return [
        MultimodalRecord(
            case_id=case_id,
            cfp=split_root / "images" / f"{case_id}.png",
            ffa_early=split_root / "FFA_A" / f"{case_id}.png",
            ffa_late=split_root / "FFA_AV" / f"{case_id}.png",
            roi=split_root / "masks" / f"{case_id}.png",
            av_label=(
                split_root / "av" / f"{case_id}.png"
                if split == "training"
                else None
            ),
            biomarker=(
                split_root / "biomarker" / f"{case_id}.txt"
                if split == "training"
                else None
            ),
        )
        for case_id in case_ids
    ]


def assert_task1_isolation(records: Iterable[Task1Record]) -> None:
    """Reject any Task 1 record that exposes an FFA-named field or path."""

    for record in records:
        fields = vars(record)
        for name, value in fields.items():
            if "ffa" in name.lower():
                raise DatasetLayoutError(
                    f"Task 1 record {record.case_id} exposes forbidden field {name!r}"
                )
            if isinstance(value, Path) and any(
                "ffa" in part.lower() for part in value.parts
            ):
                raise DatasetLayoutError(
                    f"Task 1 record {record.case_id} exposes forbidden path {value}"
                )
