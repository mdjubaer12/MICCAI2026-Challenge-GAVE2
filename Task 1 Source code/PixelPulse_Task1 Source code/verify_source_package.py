#!/usr/bin/env python3
"""Verify immutable Task 1 source and confirm that large binaries are absent."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPECTED = {
    "manifests/training_folds.csv": "a182eac3715521d7b4094b5fdfc368b853341e330419319b37e9da0d0f155740",
    "scripts/train_hrvrl_task1.py": "440e74812ceed558bf212c91d8bc58a3a734928a7d99bcf4067c9bf0a1dfe0f9",
    "scripts/export_hrvrl_task1_validation.py": "edd25e8cf41b33f6bd31f8b80777025b1033847f318198647a82622bb3a4ddeb",
    "scripts/assemble_task1_validation_ensemble.py": "58b4c53071945c4fea33632f166c5df103d23a65276b5c1988bfe96add970f1a",
    "scripts/refine_task1_rrwnet.py": "af4ff2fde157dd585466b49b0f0e9689906f7adbc5086a166a1d3cb09b43e7de",
    "scripts/apply_task1_postprocess.py": "e0f0b460b4b2896de4d441d1e574ac272c75178cb00838554c9939ef45082c02",
    "scripts/enhance_task1_task2_topology.py": "6234d31aac09e8c1c83aeac06dc87d24f9d0f3593c87179ca062b63797e8a481",
    "src/gave2/task1_inference.py": "7d8cdf7060149e4790909064e1c9ef6428208568cc29bdd17f2256f32d9a343b",
    "src/gave2/task1_postprocess.py": "8491596379d0b5a81ff79e6e28568cd9bc6097dcbbb24f3bbbba10700740892a",
    "src/gave2/topology_enhancement.py": "d3800a11308aeb1753323c6ac2f01dea33bdd97ae73534f256970de0fc234c15",
    "reference/SHA256SUMS": "d239841cd10ac62030f1811ec41962dcff81b531d1c783107fbe4ad88a3ecfa1",
}

OMITTED_BINARIES = (
    "models/hrvrl/fold_0_best.pt",
    "models/hrvrl/fold_1_best.pt",
    "models/hrvrl/fold_2_best.pt",
    "models/hrvrl/fold_3_best.pt",
    "models/hrvrl/fold_4_best.pt",
    "external/HRVRL/weights/G_pretrain.pkl",
    "external/RRWNet/weights/rrwnet_RITE_refinement.pth",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-weights",
        action="store_true",
        help="Allow separately supplied model binaries to be present.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif (actual := sha256(path)) != expected:
            failures.append(f"hash mismatch: {relative}: {actual}")

    if not args.allow_weights:
        unexpected = [
            relative for relative in OMITTED_BINARIES if (ROOT / relative).exists()
        ]
        failures.extend(
            f"large binary unexpectedly present: {path}" for path in unexpected
        )

    protected = [ROOT / name for name in ("dataset", "data", "training", "validation")]
    failures.extend(
        f"protected-data directory present: {path.name}"
        for path in protected
        if path.exists()
    )

    if failures:
        raise SystemExit("SOURCE PACKAGE VERIFICATION FAILED\n" + "\n".join(failures))
    print("SOURCE PACKAGE VERIFICATION PASSED")


if __name__ == "__main__":
    main()
