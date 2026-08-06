#!/usr/bin/env python3
"""Evaluate the HRVRL AV-pretraining generator on GAVE2 Task 1.

This adapter intentionally uses only CFP images.  It does not use HRVRL's
``finetune`` directory because that code performs image-level disease
classification rather than artery/vein segmentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gave2.evaluation import evaluate_task1_arrays, save_probability_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of HRVRL's CFP AV segmenter on GAVE2"
    )
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
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "external"
        / "HRVRL"
        / "weights"
        / "G_pretrain.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task1_hrvrl" / "zero_shot",
    )
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topology-paths", type=int, default=0)
    parser.add_argument("--topology-seed", type=int, default=20260719)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--minimum-checkpoint-coverage",
        type=float,
        default=0.95,
        help="Minimum fraction of model state elements loaded from the checkpoint",
    )
    return parser.parse_args()


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected a state-dict-like checkpoint, got {type(checkpoint).__name__}"
        )
    candidate: Mapping[object, object] = checkpoint
    for key in ("model", "state_dict", "model_ema", "netG"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            candidate = value
            break
    state: dict[str, torch.Tensor] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            continue
        key = raw_key
        for prefix in ("module.", "netG."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        state[key] = value
    if not state:
        raise ValueError("No tensor state dictionary was found in the checkpoint")
    return state


def load_model(
    hrvrl_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    minimum_coverage: float,
) -> tuple[torch.nn.Module, dict[str, object]]:
    av_root = hrvrl_root / "pretrain" / "AV"
    if not av_root.is_dir():
        raise FileNotFoundError(f"HRVRL AV source is missing: {av_root}")
    sys.path.insert(0, str(av_root))
    try:
        from models.network import PGNet
    except ModuleNotFoundError as error:
        if error.name == "einops":
            raise RuntimeError(
                "HRVRL requires einops. Install it in the selected environment."
            ) from error
        raise

    try:
        raw_checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(raw_checkpoint)
    use_global_semantic = any(
        key.startswith(("pg_fusion.", "base_layers_global_momentum."))
        for key in state
    )
    use_centerness = any(key.startswith("cenBlock") for key in state)
    model = PGNet(
        input_ch=3,
        resnet="convnext_tiny",
        num_classes=3,
        use_cuda=device.type == "cuda",
        pretrained=False,
        centerness=use_centerness,
        centerness_map_size=[256, 256],
        use_global_semantic=use_global_semantic,
    )
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    loaded_elements = sum(value.numel() for value in compatible.values())
    total_elements = sum(value.numel() for value in model_state.values())
    coverage = loaded_elements / max(total_elements, 1)
    if coverage < minimum_coverage:
        raise RuntimeError(
            "Checkpoint coverage is too low: "
            f"{coverage:.4%} < {minimum_coverage:.4%}. "
            "Refusing to evaluate a partially initialized model."
        )
    incompatible = model.load_state_dict(compatible, strict=False)
    model.to(device).eval()
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_keys": len(state),
        "compatible_keys": len(compatible),
        "checkpoint_coverage": coverage,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "use_global_semantic": use_global_semantic,
        "use_centerness": use_centerness,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, metadata


def _normalize(batch: np.ndarray) -> torch.Tensor:
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (batch.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(
        np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))
    )


def _pad_for_sliding_window(
    image: np.ndarray,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    target_height = max(height, patch_size)
    target_width = max(width, patch_size)
    target_height += (stride - (target_height - patch_size) % stride) % stride
    target_width += (stride - (target_width - patch_size) % stride) % stride
    return np.pad(
        image,
        ((0, target_height - height), (0, target_width - width), (0, 0)),
        mode="constant",
    )


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image: np.ndarray,
    roi: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
    use_global_semantic: bool,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, received {image.shape}")
    if roi.shape != image.shape[:2]:
        raise ValueError(f"Image/ROI mismatch: {image.shape[:2]} versus {roi.shape}")
    padded = _pad_for_sliding_window(image, patch_size, stride)
    height, width = image.shape[:2]
    padded_height, padded_width = padded.shape[:2]
    context_margin = patch_size // 4
    context_size = patch_size + 2 * context_margin
    context_source = np.pad(
        padded,
        (
            (context_margin, context_margin),
            (context_margin, context_margin),
            (0, 0),
        ),
        mode="constant",
    )
    positions = [
        (top, left)
        for top in range(0, padded_height - patch_size + 1, stride)
        for left in range(0, padded_width - patch_size + 1, stride)
    ]
    probability_sum = np.zeros((3, padded_height, padded_width), dtype=np.float32)
    probability_count = np.zeros((padded_height, padded_width), dtype=np.float32)
    for start in range(0, len(positions), batch_size):
        selected = positions[start : start + batch_size]
        local_batch = np.stack(
            [
                padded[top : top + patch_size, left : left + patch_size]
                for top, left in selected
            ]
        )
        local_tensor = _normalize(local_batch).to(device, non_blocking=True)
        if use_global_semantic:
            context_batch = np.stack(
                [
                    cv2.resize(
                        context_source[
                            top : top + context_size,
                            left : left + context_size,
                        ],
                        (patch_size, patch_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for top, left in selected
                ]
            )
            context_tensor = _normalize(context_batch).to(
                device,
                non_blocking=True,
            )
        else:
            context_tensor = None
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            logits, _ = model(local_tensor, context_tensor)
            probabilities = torch.sigmoid(logits).float().cpu().numpy()
        for (top, left), patch_probability in zip(
            selected,
            probabilities,
            strict=True,
        ):
            probability_sum[
                :,
                top : top + patch_size,
                left : left + patch_size,
            ] += patch_probability
            probability_count[
                top : top + patch_size,
                left : left + patch_size,
            ] += 1.0
    if float(probability_count.min()) < 1.0:
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    prediction = (
        probability_sum / probability_count[None, :, :]
    )[:, :height, :width].transpose(1, 2, 0)
    prediction *= roi[..., None]
    return np.clip(prediction, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    if args.patch_size <= 0 or args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("Patch size, stride, and batch size must be positive")
    if args.stride > args.patch_size:
        raise ValueError("Stride cannot exceed patch size")
    if not 0.0 <= args.minimum_checkpoint_coverage <= 1.0:
        raise ValueError("Checkpoint coverage must lie in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model, model_metadata = load_model(
        args.hrvrl_root,
        args.checkpoint,
        device,
        args.minimum_checkpoint_coverage,
    )
    print(json.dumps({"model": model_metadata}, sort_keys=True), flush=True)

    training_root = args.dataset_root / "training"
    image_paths = sorted((training_root / "images").glob("g_*.png"))
    if args.max_cases is not None:
        image_paths = image_paths[: args.max_cases]
    if not image_paths:
        raise FileNotFoundError(f"No training images found under {training_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    rois: list[np.ndarray] = []
    amp = device.type == "cuda" and not args.no_amp
    for image_path in image_paths:
        case_id = image_path.stem
        label_path = training_root / "av" / f"{case_id}.png"
        roi_path = training_root / "masks" / f"{case_id}.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        label = np.asarray(Image.open(label_path).convert("RGBA"))
        roi = np.asarray(Image.open(roi_path).convert("L")) > 0
        prediction = predict_image(
            model,
            image,
            roi,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            amp=amp,
            use_global_semantic=bool(model_metadata["use_global_semantic"]),
        )
        save_probability_map(prediction, prediction_dir / f"{case_id}.png")
        labels.append(label)
        predictions.append(prediction)
        rois.append(roi)
        print(
            json.dumps(
                {
                    "case": case_id,
                    "mean_probability_in_roi": [
                        float(prediction[..., channel][roi].mean())
                        for channel in range(3)
                    ],
                    "positive_fraction_in_roi": [
                        float(
                            np.mean(
                                prediction[..., channel][roi] > args.threshold
                            )
                        )
                        for channel in range(3)
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    metrics = evaluate_task1_arrays(
        labels,
        predictions,
        rois,
        threshold=args.threshold,
        topology_paths=args.topology_paths,
        topology_seed=args.topology_seed,
    )
    result = {
        "source": "sulab-wmu/HRVRL pretrain/AV zero-shot",
        "task1_ffa_used": False,
        "case_count": len(image_paths),
        "threshold": args.threshold,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "amp": amp,
        "model": model_metadata,
        "metrics": metrics.to_dict(),
    }
    result_path = args.output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
