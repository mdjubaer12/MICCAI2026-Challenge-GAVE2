"""ROI-masked segmentation losses for Task 1."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_focal_tversky_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    channel_weights: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    smooth: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return recall-biased focal Tversky loss and per-channel indices."""

    if alpha < 0.0 or beta < 0.0 or alpha + beta <= 0.0:
        raise ValueError("Tversky alpha/beta must be non-negative and non-zero")
    if gamma <= 0.0:
        raise ValueError("Tversky gamma must be positive")
    dimensions = (0, 2, 3)
    masked_probability = probabilities * mask
    masked_target = target * mask
    true_positive = (masked_probability * masked_target).sum(dim=dimensions)
    false_negative = ((1.0 - masked_probability) * masked_target).sum(
        dim=dimensions
    )
    false_positive = (masked_probability * (1.0 - masked_target) * mask).sum(
        dim=dimensions
    )
    index = (true_positive + smooth) / (
        true_positive
        + alpha * false_negative
        + beta * false_positive
        + smooth
    )
    per_channel = (1.0 - index).clamp_min(0.0).pow(gamma)
    return (per_channel * channel_weights).sum(), index


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    vertical = -F.max_pool2d(
        -image,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0),
    )
    horizontal = -F.max_pool2d(
        -image,
        kernel_size=(1, 3),
        stride=1,
        padding=(0, 1),
    )
    return torch.minimum(vertical, horizontal)


def _soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def _soft_open(image: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(image))


def soft_skeletonize(image: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """Differentiable morphology from the clDice reference formulation."""

    if iterations < 1:
        raise ValueError("Skeletonization iterations must be positive")
    opened = _soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = _soft_erode(image)
        opened = _soft_open(image)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def masked_soft_cldice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    channel_weights: torch.Tensor,
    iterations: int = 5,
    smooth: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ROI-masked soft-clDice loss and per-channel clDice values."""

    masked_probability = probabilities * mask
    masked_target = target * mask
    predicted_skeleton = soft_skeletonize(
        masked_probability,
        iterations=iterations,
    )
    target_skeleton = soft_skeletonize(masked_target, iterations=iterations)
    dimensions = (0, 2, 3)
    topology_precision = (
        (predicted_skeleton * masked_target).sum(dim=dimensions) + smooth
    ) / (predicted_skeleton.sum(dim=dimensions) + smooth)
    topology_sensitivity = (
        (target_skeleton * masked_probability).sum(dim=dimensions) + smooth
    ) / (target_skeleton.sum(dim=dimensions) + smooth)
    cldice = (
        2.0
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity).clamp_min(1e-6)
    )
    return ((1.0 - cldice) * channel_weights).sum(), cldice


def masked_endpoint_connectivity_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    channel_weights: torch.Tensor,
    skeleton_iterations: int = 5,
    endpoint_radius: int = 4,
    endpoint_boost: float = 2.0,
    focal_gamma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize weak target-centerline support, emphasizing bounded endpoints.

    The target skeleton and endpoint neighborhoods are fixed supervision, not
    predicted graph operations. Every supervised pixel lies on the target
    centerline and inside the ROI. ``endpoint_radius`` bounds how far endpoint
    emphasis can extend, while the unboosted centerline term still penalizes
    internal breaks sampled by the organizer's path metrics.
    """

    if skeleton_iterations < 1:
        raise ValueError("Skeletonization iterations must be positive")
    if endpoint_radius < 0 or endpoint_radius > 16:
        raise ValueError("Endpoint radius must be in [0, 16]")
    if endpoint_boost < 0.0:
        raise ValueError("Endpoint boost must be non-negative")
    if focal_gamma <= 0.0:
        raise ValueError("Endpoint focal gamma must be positive")

    masked_target = target.float() * mask.float()
    with torch.no_grad():
        target_skeleton = (
            soft_skeletonize(
                masked_target,
                iterations=skeleton_iterations,
            )
            > 1e-4
        ).to(dtype=torch.float32)
        channels = target_skeleton.shape[1]
        neighbor_kernel = target_skeleton.new_ones((channels, 1, 3, 3))
        neighbor_kernel[:, :, 1, 1] = 0.0
        neighbor_count = F.conv2d(
            target_skeleton,
            neighbor_kernel,
            stride=1,
            padding=1,
            groups=channels,
        )
        endpoints = target_skeleton * (neighbor_count <= 1.0).to(
            dtype=torch.float32
        )
        if endpoint_radius:
            kernel_size = 2 * endpoint_radius + 1
            endpoint_neighborhood = F.max_pool2d(
                endpoints,
                kernel_size=kernel_size,
                stride=1,
                padding=endpoint_radius,
            )
        else:
            endpoint_neighborhood = endpoints
        supervision_weight = target_skeleton * (
            1.0 + endpoint_boost * endpoint_neighborhood
        )
        supervision_weight *= mask.float()

    miss = (1.0 - probabilities.float()).clamp(0.0, 1.0).pow(focal_gamma)
    dimensions = (0, 2, 3)
    denominator = supervision_weight.sum(dim=dimensions).clamp_min(1.0)
    per_channel_loss = (
        miss * supervision_weight
    ).sum(dim=dimensions) / denominator
    endpoint_count = endpoints.sum(dim=dimensions)
    return (
        (per_channel_loss * channel_weights).sum(),
        per_channel_loss,
        endpoint_count,
    )


def task1_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    roi: torch.Tensor,
    *,
    channel_weights: tuple[float, float, float] = (1.0, 0.75, 1.0),
    dice_weight: float = 0.6,
    focal_tversky_weight: float = 0.0,
    cldice_weight: float = 0.0,
    endpoint_connectivity_weight: float = 0.0,
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
    tversky_gamma: float = 0.75,
    skeleton_iterations: int = 5,
    endpoint_radius: int = 4,
    endpoint_boost: float = 2.0,
    endpoint_focal_gamma: float = 2.0,
) -> dict[str, torch.Tensor]:
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError(
            f"Expected matching Bx3xHxW logits/target, got {logits.shape}/{target.shape}"
        )
    if roi.shape != (logits.shape[0], 1, logits.shape[2], logits.shape[3]):
        raise ValueError(f"Expected Bx1xHxW ROI, got {roi.shape}")
    component_weights = (
        dice_weight,
        focal_tversky_weight,
        cldice_weight,
        endpoint_connectivity_weight,
    )
    if any(weight < 0.0 for weight in component_weights):
        raise ValueError("Loss component weights must be non-negative")
    bce_weight = 1.0 - sum(component_weights)
    if bce_weight < -1e-8:
        raise ValueError(
            "Dice, Tversky, clDice, and endpoint weights must sum to at most one"
        )

    # Keep reductions and differentiable morphology in FP32 under AMP. A
    # 512x512 half-precision sum can exceed the finite fp16 range.
    logits_float = logits.float()
    target_float = target.float()
    mask = roi.to(dtype=torch.float32)
    weights = logits_float.new_tensor(channel_weights)
    weights = weights / weights.sum()

    pixel_bce = F.binary_cross_entropy_with_logits(
        logits_float,
        target_float,
        reduction="none",
    )
    denominator = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
    bce_per_channel = (pixel_bce * mask).sum(dim=(0, 2, 3)) / denominator
    bce = (bce_per_channel * weights).sum()

    probabilities = torch.sigmoid(logits_float)
    masked_probabilities = probabilities * mask
    masked_target = target_float * mask
    dimensions = (0, 2, 3)
    intersection = (masked_probabilities * masked_target).sum(dim=dimensions)
    cardinality = masked_probabilities.sum(dim=dimensions) + masked_target.sum(
        dim=dimensions
    )
    dice_per_channel = (2.0 * intersection + 1.0) / (cardinality + 1.0)
    dice_loss = ((1.0 - dice_per_channel) * weights).sum()

    if focal_tversky_weight > 0.0:
        focal_tversky, tversky_index = masked_focal_tversky_loss(
            probabilities,
            target_float,
            mask,
            channel_weights=weights,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
        )
    else:
        focal_tversky = logits_float.new_zeros(())
        tversky_index = logits_float.new_ones(3)

    if cldice_weight > 0.0:
        cldice_loss, soft_cldice = masked_soft_cldice_loss(
            probabilities,
            target_float,
            mask,
            channel_weights=weights,
            iterations=skeleton_iterations,
        )
    else:
        cldice_loss = logits_float.new_zeros(())
        soft_cldice = logits_float.new_ones(3)

    if endpoint_connectivity_weight > 0.0:
        (
            endpoint_connectivity_loss,
            endpoint_connectivity_per_channel,
            target_endpoint_count,
        ) = masked_endpoint_connectivity_loss(
            probabilities,
            target_float,
            mask,
            channel_weights=weights,
            skeleton_iterations=skeleton_iterations,
            endpoint_radius=endpoint_radius,
            endpoint_boost=endpoint_boost,
            focal_gamma=endpoint_focal_gamma,
        )
    else:
        endpoint_connectivity_loss = logits_float.new_zeros(())
        endpoint_connectivity_per_channel = logits_float.new_zeros(3)
        target_endpoint_count = logits_float.new_zeros(3)

    loss = (
        max(0.0, bce_weight) * bce
        + dice_weight * dice_loss
        + focal_tversky_weight * focal_tversky
        + cldice_weight * cldice_loss
        + endpoint_connectivity_weight * endpoint_connectivity_loss
    )
    return {
        "loss": loss,
        "bce": bce.detach(),
        "dice_loss": dice_loss.detach(),
        "soft_dice": dice_per_channel.detach(),
        "focal_tversky_loss": focal_tversky.detach(),
        "tversky_index": tversky_index.detach(),
        "cldice_loss": cldice_loss.detach(),
        "soft_cldice": soft_cldice.detach(),
        "endpoint_connectivity_loss": endpoint_connectivity_loss.detach(),
        "endpoint_connectivity_per_channel": (
            endpoint_connectivity_per_channel.detach()
        ),
        "target_endpoint_count": target_endpoint_count.detach(),
    }


def task1_recursive_segmentation_loss(
    stage_logits: list[torch.Tensor] | tuple[torch.Tensor, ...],
    target: torch.Tensor,
    roi: torch.Tensor,
    *,
    stage_decay: float = 0.7,
    **loss_kwargs: object,
) -> dict[str, torch.Tensor]:
    """Apply increasing deep supervision to coarse and refinement stages."""

    if not stage_logits:
        raise ValueError("At least one prediction stage is required")
    if not 0.0 < stage_decay <= 1.0:
        raise ValueError("stage_decay must lie in (0, 1]")
    results = [
        task1_segmentation_loss(logits, target, roi, **loss_kwargs)
        for logits in stage_logits
    ]
    reference = results[-1]["loss"]
    weights = reference.new_tensor(
        [stage_decay ** (len(results) - index - 1) for index in range(len(results))]
    )
    weights = weights / weights.sum()
    total = sum(
        weight * result["loss"] for weight, result in zip(weights, results, strict=True)
    )
    output = dict(results[-1])
    output["loss"] = total
    output["stage_losses"] = torch.stack(
        [result["loss"].detach() for result in results]
    )
    output["stage_weights"] = weights.detach()
    for name in (
        "bce",
        "dice_loss",
        "focal_tversky_loss",
        "cldice_loss",
        "endpoint_connectivity_loss",
    ):
        output[name] = sum(
            weight * result[name]
            for weight, result in zip(weights, results, strict=True)
        ).detach()
    return output
