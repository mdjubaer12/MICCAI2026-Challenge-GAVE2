"""Overlap-tiled inference for full-resolution fundus images."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def gaussian_patch_weight(
    patch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.linspace(-1.0, 1.0, patch_size, device=device)
    one_dimensional = torch.exp(-0.5 * (coordinates / 0.5).square())
    weight = torch.outer(one_dimensional, one_dimensional).clamp_min(1e-3)
    return weight.to(dtype=dtype)[None, None]


@torch.inference_mode()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    patch_size: int = 512,
    overlap: int = 128,
    amp: bool = True,
) -> torch.Tensor:
    """Return CPU HxWx3 probabilities for a CHW normalized image."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW image with three channels, got {image.shape}")
    if not 0 <= overlap < patch_size:
        raise ValueError("overlap must be in [0, patch_size)")
    device = next(model.parameters()).device
    image = image.to(device=device, dtype=torch.float32)
    original_height, original_width = image.shape[-2:]
    pad_height = max(0, patch_size - original_height)
    pad_width = max(0, patch_size - original_width)
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
    height, width = image.shape[-2:]
    stride = patch_size - overlap
    y_starts = _starts(height, patch_size, stride)
    x_starts = _starts(width, patch_size, stride)
    probabilities = torch.zeros((1, 3, height, width), device=device)
    normalizer = torch.zeros((1, 1, height, width), device=device)
    weight = gaussian_patch_weight(
        patch_size,
        device=device,
        dtype=probabilities.dtype,
    )

    model_was_training = model.training
    model.eval()
    for y in y_starts:
        for x in x_starts:
            patch = image[:, y : y + patch_size, x : x + patch_size][None]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                prediction = torch.sigmoid(model(patch)).float()
            probabilities[:, :, y : y + patch_size, x : x + patch_size] += (
                prediction * weight
            )
            normalizer[:, :, y : y + patch_size, x : x + patch_size] += weight
    if model_was_training:
        model.train()
    probabilities = probabilities / normalizer.clamp_min(
        torch.finfo(probabilities.dtype).eps
    )
    result = probabilities[0, :, :original_height, :original_width]
    if not torch.isfinite(result).all():
        raise RuntimeError("Sliding-window inference produced non-finite values")
    return result.permute(1, 2, 0).cpu()
