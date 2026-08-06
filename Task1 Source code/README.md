# PixelPulse GAVE2 Task 1 Source Code

This lightweight archive contains the Task 1 source code and reproducibility
documentation for PixelPulse's official preliminary Task 1 score of
**8.16666**.

Large binary files are intentionally excluded:

- five fine-tuned HRVRL checkpoints;
- the public HRVRL pretrained weight;
- the public RRWNet pretrained weight; and
- the 50 frozen preliminary prediction PNGs.

This reduces the package from approximately 1.4 GB to well below 1 MB.

## Reproducibility meaning

The source archive is sufficient to inspect the full method and retrain it.
Exact byte-level reproduction of the submitted preliminary predictions still
requires the exact five selected checkpoints. Retraining can produce a
scientifically equivalent model, but floating-point and optimization
differences mean it cannot be guaranteed to recreate identical checkpoint
bytes.

The omitted binary hashes are recorded in [WEIGHTS.md](WEIGHTS.md). The team
should retain the full private archive and provide the binaries separately
only if the organizers request them.

## Included method

The selected Task 1 pipeline is:

1. Five fold-specific HRVRL PGNet models using CFP and ROI only.
2. Sliding-window inference with patch size 512, stride 150, batch size 4,
   and mixed precision.
3. One RRWNet refinement stage for each fold.
4. Equal averaging within each model family.
5. Family blend `0.90 HRVRL + 0.10 RRWNet`.
6. Seeded component repair using low threshold 0.002, seed threshold 0.80,
   and closing radius 3.
7. Skeleton enhancement using threshold 0.35, maximum gap 6 pixels, and
   centerline floor 0.65.
8. Vessel channel reconstruction as `max(artery, vein)`.
9. ROI-masked 8-bit RGB output in artery, vessel, vein order.

No FFA image or FFA-derived feature is used by Task 1.

## Package contents

```text
PixelPulse_Task1_8_16666_Source/
├── README.md
├── WEIGHTS.md
├── environment_task1.yml
├── requirements.txt
├── verify_source_package.py
├── train_task1_folds.sh
├── run_task1_inference.sh
├── compare_reference.py
├── package_task1_predictions.py
├── manifests/training_folds.csv
├── scripts/
├── src/gave2/
├── external/
│   ├── HRVRL source and license
│   └── RRWNet source and license
├── provenance/
├── reference/SHA256SUMS
└── docs/
```

The original pipeline source files are unmodified and retain the hashes
reported in the technical report.

## Dataset layout

The official dataset must be supplied separately:

```text
GAVE2_preliminary/
├── training/
│   ├── images/g_001.png ... g_050.png
│   ├── masks/g_001.png  ... g_050.png
│   └── av/g_001.png     ... g_050.png
└── validation/
    ├── images/g_051.png ... g_100.png
    └── masks/g_051.png  ... g_100.png
```

## Environment

```bash
conda env create -f environment_task1.yml
conda activate pixelpulse-task1
python verify_source_package.py
```

On the original workstation:

```bash
/home/iot/anaconda3/envs/env/bin/python verify_source_package.py
```

Expected result:

```text
SOURCE PACKAGE VERIFICATION PASSED
```

## Retrain the five folds

First place the public HRVRL initialization at:

```text
external/HRVRL/weights/G_pretrain.pkl
```

Then run:

```bash
PYTHON=/home/iot/anaconda3/envs/env/bin/python \
bash train_task1_folds.sh \
  /absolute/path/to/GAVE2_preliminary \
  /absolute/path/to/new_task1_training
```

This command:

1. generates the 50 fixed HRVRL teacher maps;
2. trains folds 0 through 4 for 24 epochs; and
3. writes `best.pt`, training history, and run metadata for each fold.

The exact training configuration is embedded in the shell script and recorded
in `provenance/training/`.

## Run exact inference when weights are available

Place files according to [WEIGHTS.md](WEIGHTS.md), then run:

```bash
PYTHON=/home/iot/anaconda3/envs/env/bin/python \
bash run_task1_inference.sh \
  /absolute/path/to/GAVE2_preliminary \
  /absolute/path/to/task1_reproduction
```

Final products:

```text
/absolute/path/to/task1_reproduction/final/
/absolute/path/to/task1_reproduction/PixelPulse_Task1_predictions.zip
```

The pipeline compares generated predictions against the frozen SHA-256
manifest. A byte-identical preliminary reproduction prints:

```text
REFERENCE COMPARISON PASSED: 50/50 files match
```

## Official result

| Class | AV DSC | Sensitivity | Specificity | Accuracy | INF | COR |
|---|---:|---:|---:|---:|---:|---:|
| Artery | 0.6702 | 0.9521 | 0.9589 | 0.9587 | 0.2614 | 0.7290 |
| Vein | 0.6702 | 0.9639 | 0.9569 | 0.9572 | 0.2290 | 0.7558 |

Official Task 1 score: **8.16666**

The identical Task 1 payload received 8.14713 and 8.16666 in repeated
evaluations because the organizer's path-based topology evaluation appears to
use stochastic sampling.

## External resources

- HRVRL: <https://github.com/sulab-wmu/HRVRL>
- RRWNet: <https://github.com/j-morano/rrwnet>
- RRWNet source revision:
  `3ca17cfd5c4403cad0f439aa4d5bdbd030fe04d1`

The included third-party source and license files retain their original MIT
license notices.
