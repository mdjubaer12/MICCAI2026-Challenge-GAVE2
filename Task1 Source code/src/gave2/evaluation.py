"""GAVE artery/vein encoding and 2025-official-compatible metrics.

The topology algorithm is an independently structured implementation of the
MIT-licensed GAVE 2025 evaluator:
https://github.com/liuzw20/GAVE/blob/main/Code/Tool/topo_metric.py

GAVE2 has not yet published an executable evaluator. Results from this module
must therefore be described as a reproduction of the official 2025 evaluator
and a GAVE2 proxy until the organizers confirm parity.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from skimage import graph, morphology


CHANNEL_NAMES = ("artery", "vessel", "vein")


@dataclass(frozen=True)
class ClassMetrics:
    dice: float
    sensitivity: float
    specificity: float
    accuracy: float
    inf: float
    corr: float


@dataclass(frozen=True)
class Task1Metrics:
    artery: ClassMetrics
    vein: ClassMetrics
    vessel_dice: float
    score_2025_proxy: float
    score_live_leaderboard: float
    topology_paths_per_case: int
    topology_seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decode_av_label(label_rgb: np.ndarray) -> np.ndarray:
    """Decode RGB/RGBA annotation into artery, vessel-union, and vein masks."""

    label = np.asarray(label_rgb)
    if label.ndim != 3 or label.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 AV label, received {label.shape}")
    rgb = label[..., :3]
    red = rgb[..., 0] > 127
    green = rgb[..., 1] > 127
    blue = rgb[..., 2] > 127
    known = red | green | blue
    invalid = known & (
        (red.astype(np.uint8) + green.astype(np.uint8) + blue.astype(np.uint8))
        != 1
    )
    if np.any(invalid):
        colors = np.unique(rgb[invalid].reshape(-1, 3), axis=0)
        raise ValueError(f"Invalid mixed AV-label colors: {colors.tolist()}")
    return np.stack((red | green, known, blue | green), axis=-1).astype(
        np.float32
    )


def encode_probability_map(probabilities: np.ndarray) -> np.ndarray:
    """Encode HxWx3 artery/vessel/vein probabilities as an RGB uint8 PNG."""

    array = np.asarray(probabilities)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 probabilities, received {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Probability map contains non-finite values")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("Probability map values must lie in [0, 1]")
    return np.rint(array * 255.0).astype(np.uint8)


def save_probability_map(probabilities: np.ndarray, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encode_probability_map(probabilities), mode="RGB").save(
        destination
    )


def _normalize_prediction(prediction: np.ndarray) -> np.ndarray:
    array = np.asarray(prediction, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3 prediction, received {array.shape}")
    array = array[..., :3]
    if array.size and float(array.max()) > 1.0:
        array = array / 255.0
    if not np.all(np.isfinite(array)):
        raise ValueError("Prediction contains non-finite values")
    return np.clip(array, 0.0, 1.0)


def _normalize_roi(roi: np.ndarray) -> np.ndarray:
    array = np.asarray(roi)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=-1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D ROI mask, received {array.shape}")
    return array > 0.5


def _dice_from_counts(intersection: int, target: int, predicted: int) -> float:
    if target == 0 and predicted == 0:
        return float("nan")
    return float((2.0 * intersection) / (target + predicted + 1e-7))


def _classification_from_counts(
    true_positive: int,
    true_negative: int,
    false_positive: int,
    false_negative: int,
) -> tuple[float, float, float]:
    sensitivity_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    total = (
        true_positive + true_negative + false_positive + false_negative
    )
    sensitivity = (
        true_positive / sensitivity_denominator
        if sensitivity_denominator
        else float("nan")
    )
    specificity = (
        true_negative / specificity_denominator
        if specificity_denominator
        else float("nan")
    )
    accuracy = (
        (true_positive + true_negative) / total if total else float("nan")
    )
    return float(sensitivity), float(specificity), float(accuracy)


def topology_path_counts(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
    paths: int,
    rng: random.Random,
) -> tuple[int, int, int]:
    """Return infeasible, length-inaccurate, and correct sampled path counts."""

    if paths < 0:
        raise ValueError("paths must be non-negative")
    ground = np.asarray(ground_truth) > 0.5
    predicted = np.asarray(prediction) > threshold
    ground_centerline = morphology.skeletonize(ground)
    predicted_centerline = morphology.skeletonize(predicted)
    ground_points = np.argwhere(ground_centerline)
    if paths and len(ground_points) == 0:
        raise ValueError("Cannot sample topology paths from an empty target")

    ground_components = morphology.label(ground_centerline)
    predicted_components = morphology.label(predicted)
    ground_cost = np.where(ground_centerline, 1.0, 10000.0)
    predicted_cost = np.where(predicted_centerline, 1.0, 10000.0)
    predicted_points = np.argwhere(predicted_centerline)
    counts = [0, 0, 0]

    for _ in range(paths):
        first = ground_points[rng.randint(0, len(ground_points) - 1)]
        component = ground_components[tuple(first)]
        connected_points = np.argwhere(ground_components == component)
        second = connected_points[rng.randint(0, len(connected_points) - 1)]
        first_tuple = tuple(int(value) for value in first)
        second_tuple = tuple(int(value) for value in second)

        if (
            predicted_components[first_tuple]
            != predicted_components[second_tuple]
            or predicted_components[first_tuple] == 0
        ):
            counts[0] += 1
            continue

        distance_first = np.square(predicted_points - first).sum(axis=1)
        distance_second = np.square(predicted_points - second).sum(axis=1)
        corresponding_first = predicted_points[np.argmin(distance_first)]
        corresponding_second = predicted_points[np.argmin(distance_second)]
        ground_path, _ = graph.route_through_array(
            ground_cost,
            first_tuple,
            second_tuple,
        )
        predicted_path, _ = graph.route_through_array(
            predicted_cost,
            tuple(int(value) for value in corresponding_first),
            tuple(int(value) for value in corresponding_second),
        )
        ground_path_array = np.asarray(ground_path)
        predicted_path_array = np.asarray(predicted_path)
        if predicted_path_array.shape[0] < 2:
            counts[2] += 1
            continue
        ground_length = np.sqrt(
            np.square(np.diff(ground_path_array, axis=0)).sum(axis=1)
        ).sum()
        predicted_length = np.sqrt(
            np.square(np.diff(predicted_path_array, axis=0)).sum(axis=1)
        ).sum()
        ratio = ground_length / predicted_length
        counts[1 if ratio < 0.9 or ratio > 1.1 else 2] += 1
    return tuple(counts)  # type: ignore[return-value]


def task1_score_2025_proxy(artery: ClassMetrics, vein: ClassMetrics) -> float:
    """Apply the written GAVE2 30/40/30 score after averaging A/V classes."""

    def class_score(metrics: ClassMetrics) -> float:
        classification = (
            0.3 * metrics.sensitivity
            + 0.3 * metrics.specificity
            + 0.4 * metrics.accuracy
        )
        topology = 0.5 * metrics.corr + 0.5 * (1.0 - metrics.inf)
        return 0.3 * classification + 0.4 * metrics.dice + 0.3 * topology

    return float(10.0 * (class_score(artery) + class_score(vein)) / 2.0)


def task1_score_live_leaderboard(
    artery: ClassMetrics,
    vein: ClassMetrics,
) -> float:
    """Apply the 40/20/40 score reconstructed from the live leaderboard.

    The supplied July 20, 2026 leaderboard rows match 40% classification,
    20% DSC, and 40% topology to displayed-value rounding. This differs from
    the 30%/40%/30% formula in the written competition introduction, so both
    scores are retained explicitly until the organizer resolves the mismatch.
    """

    def class_score(metrics: ClassMetrics) -> float:
        classification = (
            0.3 * metrics.sensitivity
            + 0.3 * metrics.specificity
            + 0.4 * metrics.accuracy
        )
        topology = 0.5 * metrics.corr + 0.5 * (1.0 - metrics.inf)
        return 0.4 * classification + 0.2 * metrics.dice + 0.4 * topology

    return float(10.0 * (class_score(artery) + class_score(vein)) / 2.0)


def evaluate_task1_arrays(
    labels_rgb: Sequence[np.ndarray],
    predictions_rgb: Sequence[np.ndarray],
    roi_masks: Sequence[np.ndarray],
    *,
    threshold: float = 0.5,
    topology_paths: int = 0,
    topology_seed: int = 20260719,
) -> Task1Metrics:
    """Evaluate aligned arrays using the official 2025 aggregation behavior."""

    if not (
        len(labels_rgb) == len(predictions_rgb) == len(roi_masks)
        and len(labels_rgb) > 0
    ):
        raise ValueError("Labels, predictions, and ROI masks must align and be non-empty")

    dice_counts = {
        name: {"intersection": 0, "target": 0, "predicted": 0}
        for name in CHANNEL_NAMES
    }
    classification_counts = {
        name: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for name in ("artery", "vein")
    }
    topology_values = {
        name: {"inf": [], "corr": []} for name in ("artery", "vein")
    }
    rng = random.Random(topology_seed)

    for label_rgb, prediction_rgb, roi_mask in zip(
        labels_rgb,
        predictions_rgb,
        roi_masks,
        strict=True,
    ):
        target = decode_av_label(label_rgb)
        prediction = _normalize_prediction(prediction_rgb)
        roi = _normalize_roi(roi_mask)
        if target.shape[:2] != prediction.shape[:2] or roi.shape != target.shape[:2]:
            raise ValueError(
                "Spatial mismatch: "
                f"target={target.shape}, prediction={prediction.shape}, roi={roi.shape}"
            )

        for index, name in enumerate(CHANNEL_NAMES):
            target_binary = target[..., index] > 0.5
            prediction_binary = prediction[..., index] > threshold
            target_roi = target_binary & roi
            prediction_roi = prediction_binary & roi
            dice_counts[name]["intersection"] += int(
                np.count_nonzero(target_roi & prediction_roi)
            )
            dice_counts[name]["target"] += int(np.count_nonzero(target_roi))
            dice_counts[name]["predicted"] += int(
                np.count_nonzero(prediction_roi)
            )

        for index, name in ((0, "artery"), (2, "vein")):
            target_binary = target[..., index] > 0.5
            prediction_binary = prediction[..., index] > threshold
            counts = classification_counts[name]
            counts["tp"] += int(
                np.count_nonzero(target_binary & prediction_binary & roi)
            )
            counts["tn"] += int(
                np.count_nonzero(~target_binary & ~prediction_binary & roi)
            )
            counts["fp"] += int(
                np.count_nonzero(~target_binary & prediction_binary & roi)
            )
            counts["fn"] += int(
                np.count_nonzero(target_binary & ~prediction_binary & roi)
            )
            if topology_paths:
                infeasible, _, correct = topology_path_counts(
                    target[..., index],
                    prediction[..., index],
                    threshold,
                    topology_paths,
                    rng,
                )
                topology_values[name]["inf"].append(
                    infeasible / topology_paths
                )
                topology_values[name]["corr"].append(correct / topology_paths)

    classes = {}
    for name in ("artery", "vein"):
        dice = dice_counts[name]
        counts = classification_counts[name]
        sensitivity, specificity, accuracy = _classification_from_counts(
            counts["tp"], counts["tn"], counts["fp"], counts["fn"]
        )
        inf = (
            float(np.mean(topology_values[name]["inf"]))
            if topology_paths
            else float("nan")
        )
        corr = (
            float(np.mean(topology_values[name]["corr"]))
            if topology_paths
            else float("nan")
        )
        classes[name] = ClassMetrics(
            dice=_dice_from_counts(
                dice["intersection"], dice["target"], dice["predicted"]
            ),
            sensitivity=sensitivity,
            specificity=specificity,
            accuracy=accuracy,
            inf=inf,
            corr=corr,
        )

    vessel = dice_counts["vessel"]
    score = (
        task1_score_2025_proxy(classes["artery"], classes["vein"])
        if topology_paths
        else float("nan")
    )
    live_score = (
        task1_score_live_leaderboard(classes["artery"], classes["vein"])
        if topology_paths
        else float("nan")
    )
    return Task1Metrics(
        artery=classes["artery"],
        vein=classes["vein"],
        vessel_dice=_dice_from_counts(
            vessel["intersection"], vessel["target"], vessel["predicted"]
        ),
        score_2025_proxy=score,
        score_live_leaderboard=live_score,
        topology_paths_per_case=topology_paths,
        topology_seed=topology_seed,
    )


def load_aligned_task1_directories(
    label_dir: Path | str,
    prediction_dir: Path | str,
    roi_dir: Path | str,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Load strictly aligned PNG directories, resizing predictions if required."""

    directories = [Path(label_dir), Path(prediction_dir), Path(roi_dir)]
    case_sets = [
        {path.stem for path in directory.glob("*.png")} for directory in directories
    ]
    if not case_sets[0] or any(cases != case_sets[0] for cases in case_sets[1:]):
        raise ValueError(
            "Label, prediction, and ROI directories must contain identical PNG stems"
        )
    cases = sorted(case_sets[0])
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    rois: list[np.ndarray] = []
    for case_id in cases:
        with Image.open(directories[0] / f"{case_id}.png") as image:
            label_image = image.convert("RGB")
            size = label_image.size
            labels.append(np.asarray(label_image))
        with Image.open(directories[1] / f"{case_id}.png") as image:
            prediction_image = image.convert("RGB")
            if prediction_image.size != size:
                prediction_image = prediction_image.resize(
                    size, Image.Resampling.BILINEAR
                )
            predictions.append(np.asarray(prediction_image))
        with Image.open(directories[2] / f"{case_id}.png") as image:
            roi_image = image.convert("L")
            if roi_image.size != size:
                roi_image = roi_image.resize(size, Image.Resampling.NEAREST)
            rois.append(np.asarray(roi_image))
    return cases, labels, predictions, rois
