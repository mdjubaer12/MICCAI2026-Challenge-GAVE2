"""Topology-aware Task 1 probability calibration and mask repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import morphology


@dataclass(frozen=True)
class Task1PostprocessConfig:
    """Bounded, auditable postprocessing parameters."""

    low_threshold: float
    seed_threshold: float
    vessel_support_threshold: float | None = None
    class_dominance_ratio: float = 1.0
    closing_radius: int = 0
    minimum_component_size: int = 0
    exclusive_classes: bool = False
    exclusive_smoothing_sigma: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.low_threshold <= self.seed_threshold < 1.0:
            raise ValueError("Thresholds must satisfy 0 < low <= seed < 1")
        if self.vessel_support_threshold is not None and not (
            0.0 < self.vessel_support_threshold < 1.0
        ):
            raise ValueError("Vessel support threshold must lie in (0, 1)")
        if self.class_dominance_ratio <= 0 or not np.isfinite(
            self.class_dominance_ratio
        ):
            raise ValueError("Class dominance ratio must be finite and positive")
        if self.closing_radius < 0 or self.minimum_component_size < 0:
            raise ValueError("Morphology parameters must be non-negative")
        if (
            self.exclusive_smoothing_sigma < 0
            or not np.isfinite(self.exclusive_smoothing_sigma)
        ):
            raise ValueError("Exclusive smoothing sigma must be finite and non-negative")


def organizer_aligned_probability(
    probability: np.ndarray,
    effective_threshold: float,
) -> np.ndarray:
    """Logit-shift probabilities so ``effective_threshold`` maps to 0.5."""

    array = np.asarray(probability, dtype=np.float32)
    if not 0.0 < effective_threshold < 1.0:
        raise ValueError("Effective threshold must lie in (0, 1)")
    if not np.all(np.isfinite(array)) or array.min() < 0 or array.max() > 1:
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    epsilon = np.float32(1e-6)
    clipped = np.clip(array, epsilon, 1.0 - epsilon)
    logits = np.log(clipped) - np.log1p(-clipped)
    threshold_logit = np.log(effective_threshold) - np.log1p(
        -effective_threshold
    )
    shifted = 1.0 / (1.0 + np.exp(-(logits - threshold_logit)))
    shifted[array <= 0] = 0.0
    shifted[array >= 1] = 1.0
    return shifted.astype(np.float32)


def _seeded_components(weak: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(
        np.asarray(weak, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if count == 0 or not np.any(seeds):
        return np.zeros_like(weak, dtype=bool)
    selected = np.unique(labels[np.asarray(seeds, dtype=bool)])
    selected = selected[selected != 0]
    if len(selected) == 0:
        return np.zeros_like(weak, dtype=bool)
    keep = np.zeros(count + 1, dtype=bool)
    keep[selected] = True
    return keep[labels]


def repair_class_mask(
    probability: np.ndarray,
    other_class_probability: np.ndarray,
    vessel_probability: np.ndarray,
    roi: np.ndarray,
    config: Task1PostprocessConfig,
) -> np.ndarray:
    """Retain weak vessel networks anchored by strong class predictions."""

    current = np.asarray(probability, dtype=np.float32)
    other = np.asarray(other_class_probability, dtype=np.float32)
    vessel = np.asarray(vessel_probability, dtype=np.float32)
    valid = np.asarray(roi, dtype=bool)
    if current.shape != other.shape or current.shape != vessel.shape:
        raise ValueError("Probability channel shapes must match")
    if current.shape != valid.shape or current.ndim != 2:
        raise ValueError("Probability and ROI shapes must be equal and 2-D")
    if not all(np.all(np.isfinite(value)) for value in (current, other, vessel)):
        raise ValueError("Probability channels contain non-finite values")

    weak = current > config.low_threshold
    if config.vessel_support_threshold is not None:
        vessel_support = vessel > config.vessel_support_threshold
        class_compatible = current > (
            config.class_dominance_ratio * np.maximum(other, 1e-6)
        )
        weak |= vessel_support & class_compatible
    weak &= valid
    seeds = (current > config.seed_threshold) & valid
    repaired = _seeded_components(weak, seeds)
    if config.closing_radius:
        repaired = morphology.binary_closing(
            repaired,
            footprint=morphology.disk(config.closing_radius),
        ) & valid
    if config.minimum_component_size:
        repaired = morphology.remove_small_objects(
            repaired,
            min_size=config.minimum_component_size,
            connectivity=2,
        )
    return np.asarray(repaired, dtype=bool) & valid


def postprocess_task1_probability(
    probability: np.ndarray,
    roi: np.ndarray,
    config: Task1PostprocessConfig,
) -> np.ndarray:
    """Return organizer-aligned A/vessel/V probabilities with repaired masks."""

    source = np.asarray(probability, dtype=np.float32)
    valid = np.asarray(roi, dtype=bool)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("Expected an HxWx3 probability map")
    if source.shape[:2] != valid.shape:
        raise ValueError("Probability and ROI shapes must match")
    artery = repair_class_mask(
        source[..., 0], source[..., 2], source[..., 1], valid, config
    )
    vein = repair_class_mask(
        source[..., 2], source[..., 0], source[..., 1], valid, config
    )
    if config.exclusive_classes:
        overlap = artery & vein
        if config.exclusive_smoothing_sigma:
            support = np.asarray(artery | vein, dtype=np.float32)
            epsilon = np.float32(1e-6)
            evidence = np.log(source[..., 0] + epsilon) - np.log(
                source[..., 2] + epsilon
            )
            sigma = config.exclusive_smoothing_sigma
            numerator = ndimage.gaussian_filter(
                evidence * support,
                sigma=sigma,
                mode="nearest",
            )
            denominator = ndimage.gaussian_filter(
                support,
                sigma=sigma,
                mode="nearest",
            )
            smoothed_evidence = numerator / np.maximum(denominator, epsilon)
            artery_wins = smoothed_evidence >= 0.0
        else:
            artery_wins = source[..., 0] >= source[..., 2]
        artery &= ~overlap | artery_wins
        vein &= ~overlap | ~artery_wins
    vessel = repair_class_mask(
        source[..., 1], np.zeros_like(source[..., 1]), source[..., 1], valid,
        Task1PostprocessConfig(
            low_threshold=config.low_threshold,
            seed_threshold=config.seed_threshold,
            closing_radius=config.closing_radius,
            minimum_component_size=config.minimum_component_size,
        ),
    )
    vessel |= artery | vein

    output = np.empty_like(source, dtype=np.float32)
    masks = (artery, vessel, vein)
    for channel, mask in enumerate(masks):
        aligned = organizer_aligned_probability(
            source[..., channel], config.low_threshold
        )
        aligned[mask] = np.maximum(aligned[mask], 128.0 / 255.0)
        aligned[~mask] = np.minimum(aligned[~mask], 127.0 / 255.0)
        output[..., channel] = aligned
    output *= valid[..., None]
    return output
