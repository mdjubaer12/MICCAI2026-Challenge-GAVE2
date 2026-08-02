# GAVE2 MICCAI 2026 Challenge
## Generalized Analysis of Vessels in Eye (GAVE2) Challenge @ MICCAI 2026

<p align="center">
  <img src="https://img.shields.io/badge/MICCAI-2026-blue" />
  <img src="https://img.shields.io/badge/Challenge-GAVE2-success" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" />
  <img src="https://img.shields.io/badge/Domain-Retinal%20Image%20Analysis-brightgreen" />
</p>

---

# Project Description

This repository contains our official implementation for the **MICCAI 2026 GAVE2 Challenge (Generalized Analysis of Vessels in Eye)**. The challenge focuses on comprehensive retinal vascular analysis using multimodal retinal imaging, including color fundus photography (CFP) and fluorescein angiography (FFA).

The overall pipeline is composed of **three interconnected tasks**, where accurate vessel segmentation serves as the foundation for quantitative retinal biomarker extraction.

---

# Task 1 — Artery and Vein Segmentation from Color Fundus Images

## Objective

The first task aims to accurately segment retinal arteries and veins directly from color fundus photographs (CFP).

Given only a CFP image, the model predicts separate binary masks for:

- Arteries
- Veins

The segmentation quality is evaluated using multiple metrics including:

- Dice Similarity Coefficient (DSC)
- Sensitivity (Sen)
- Specificity (Spec)
- Accuracy (Acc)
- INF
- COR

High-quality artery-vein segmentation is essential because downstream retinal biomarkers strongly depend on precise vessel delineation.

---

# Task 2 — Cross-Modal Artery and Vein Segmentation

## Objective

Task 2 extends the segmentation problem by utilizing information from both:

- Color Fundus Photography (CFP)
- Fluorescein Angiography (FFA)

Unlike Task 1, this task leverages multimodal retinal imaging to improve vessel discrimination, particularly for challenging artery-vein boundaries and thin vessels.

The evaluation metrics remain identical to Task 1:

- Dice Similarity Coefficient (DSC)
- Sensitivity
- Specificity
- Accuracy
- INF
- COR

Cross-modal learning provides richer structural and vascular information, enabling more accurate retinal vessel segmentation.

---

# Task 3 — Retinal Vascular Biomarker Quantification

## Objective

The final task predicts clinically meaningful vascular biomarkers from the segmented artery and vein maps.

Five retinal biomarkers are evaluated:

| Biomarker | Description |
|------------|-------------|
| AVR | Artery-to-Vein Ratio |
| Artery Density | Percentage of retinal area occupied by arteries |
| Vein Density | Percentage of retinal area occupied by veins |
| Artery Fractal Dimension | Complexity of arterial branching patterns |
| Vein Fractal Dimension | Complexity of venous branching patterns |

Performance is measured using:

- Mean Absolute Error (MAE)
- Symmetric Mean Absolute Percentage Error (SMAPE)

Accurate biomarker estimation enables quantitative assessment of retinal vascular health and facilitates computer-aided diagnosis of ophthalmic diseases.

---

# Overall Pipeline

```
Color Fundus Image
        │
        ▼
Task 1:
Artery / Vein Segmentation
        │
        ▼
Task 2:
Cross-Modal Vessel Segmentation
(CFP + FFA)
        │
        ▼
Task 3:
Retinal Biomarker Quantification from Task 2 Output
(AVR, Density, Fractal Dimension)
```

---

# Performance Comparison

## Task 1 — Color Fundus Artery & Vein Segmentation

| Metric | Baseline | Ours (PixelPulse) | Improvement |
|---------|----------|------------------|-------------|
| AV Dice | **0.7656** | **0.6264** | -0.1392 |
| Artery Sensitivity | 0.6709 | **0.9478** | **+0.2769** |
| Vein Sensitivity | 0.7146 | **0.9614** | **+0.2468** |
| Artery Specificity | **0.9937** | 0.9612 | -0.0325 |
| Vein Specificity | **0.9930** | 0.9589 | -0.0341 |
| Artery Accuracy | **0.9828** | 0.9608 | -0.0220 |
| Vein Accuracy | **0.9808** | 0.9590 | -0.0218 |
| Artery INF | **0.6924** | 0.2540 | -0.4384 |
| Vein INF | **0.7672** | 0.2380 | -0.5292 |
| Artery COR | 0.3058 | **0.7372** | **+0.4314** |
| Vein COR | 0.2322 | **0.7472** | **+0.5150** |
| **Challenge Score** | 6.2039 | **8.0785** | **+1.8746** |

---

## Task 2 — Cross-Modal AV Segmentation

| Metric | Baseline | Ours (PixelPulse) | Improvement |
|---------|----------|------------------|-------------|
| AV Dice | **0.7621** | 0.6753 | -0.0868 |
| Artery Sensitivity | 0.6892 | **0.9332** | **+0.2440** |
| Vein Sensitivity | 0.7025 | **0.9480** | **+0.2455** |
| Artery Specificity | **0.9921** | 0.9671 | -0.0250 |
| Vein Specificity | **0.9924** | 0.9659 | -0.0265 |
| Artery Accuracy | **0.9819** | 0.9659 | -0.0160 |
| Vein Accuracy | **0.9797** | 0.9651 | -0.0146 |
| Artery INF | **0.6672** | 0.2846 | -0.3826 |
| Vein INF | **0.7852** | 0.2574 | -0.5278 |
| Artery COR | 0.3302 | **0.7078** | **+0.3776** |
| Vein COR | 0.2132 | **0.7348** | **+0.5216** |
| **Challenge Score** | 6.2104 | **8.0845** | **+1.8741** |

---

## Task 3 — Retinal Biomarker Quantification

| Metric | Baseline | Ours (PixelPulse) |
|---------|----------|------------------|
| AVR MAE | **0.5106** | 0.6895 |
| AVR SMAPE | **0.5617** | 0.7168 |
| Artery Density MAE | **0.6303** | 0.6920 |
| Artery Density SMAPE | **0.5708** | 0.7467 |
| Vein Density MAE | **0.6244** | 0.7132 |
| Vein Density SMAPE | **0.5218** | 0.7416 |
| Artery Fractal Dimension MAE | **0.7174** | 0.7810 |
| Artery Fractal Dimension SMAPE | **0.7146** | 0.7893 |
| Vein Fractal Dimension MAE | **0.6560** | 0.7628 |
| Vein Fractal Dimension SMAPE | **0.6595** | 0.7717 |
| **Challenge Score** | 6.1672 | **7.4045** |

---

# Overall Challenge Performance

| Method | Task 1 | Task 2 | Task 3 | Overall Score |
|---------|--------|--------|--------|---------------|
| Official Baseline | 6.2039 | 6.2104 | 6.1672 | 6.1918 |
| **PixelPulse (Islamic University)** | **8.0785** | **8.0845** | **7.4045** | **7.8113** |

---

# Leaderboard Snapshot

| Rank | Team | Organization | Overall Score |
|------|------|--------------|--------------|
| **11** | **PixelPulse** | Islamic University, Bangladesh | **7.8113** |

Submission Time:

```
2026-07-26 14:36
```

---

# Repository Structure



---

# Evaluation Metrics

## Segmentation

- Dice Similarity Coefficient (DSC)
- Sensitivity
- Specificity
- Accuracy
- INF
- COR

## Biomarker Prediction

- Mean Absolute Error (MAE)
- Symmetric Mean Absolute Percentage Error (SMAPE)

---
