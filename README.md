# GAVE2 MICCAI 2026 Challenge

<p align="center">
  <a href="https://aistudio.baidu.com/competition/detail/1463/0/leaderboard">
    <img src="https://img.shields.io/badge/MICCAI%202026-GAVE2%20Challenge-blue?style=for-the-badge" alt="MICCAI 2026 GAVE2 Challenge"/>
  </a>
</p>

---

# Project Description

This repository contains our implementation for the **MICCAI 2026 GAVE2 (Generalized Analysis of Vessels in Eye) Challenge**. The challenge focuses on retinal vessel analysis using **Color Fundus Photography (CFP)** and **Fluorescein Angiography (FFA)** images.

The project consists of **three sequential tasks**. The first two tasks focus on artery-vein segmentation, while the final task estimates clinically relevant retinal vascular biomarkers from the segmentation results.

---

# Task 1 — Artery & Vein Segmentation from Color Fundus Images

The goal of Task 1 is to segment retinal **arteries** and **veins** directly from **Color Fundus Photography (CFP)** images.

For each input fundus image, the model predicts:

- Artery segmentation mask
- Vein segmentation mask

The segmentation performance is evaluated using:

- Dice Similarity Coefficient (DSC)
- Sensitivity (Sen)
- Specificity (Spec)
- Accuracy (Acc)
- INF
- COR

> **Note:** Higher **COR** values and lower **INF** values indicate better performance. Since **COR + INF = 1.00**, improvements in COR correspond to reductions in INF.

---

# Task 2 — Cross-Modal Artery & Vein Segmentation

Task 2 extends the segmentation problem by utilizing both:

- Color Fundus Photography (CFP)
- Fluorescein Angiography (FFA)

The objective is to produce more accurate artery-vein segmentation by leveraging complementary information from both imaging modalities.

The evaluation metrics are the same as Task 1:

- Dice Similarity Coefficient (DSC)
- Sensitivity (Sen)
- Specificity (Spec)
- Accuracy (Acc)
- INF
- COR

> **Note:** Higher **COR** values and lower **INF** values indicate better performance. Since **COR + INF = 1.00**, improvements in COR correspond to reductions in INF.

---

# Task 3 — Retinal Biomarker Quantification

Task 3 predicts quantitative retinal vascular biomarkers from the **Task 2 artery and vein segmentation results**.

Our pipeline extracts the following retinal biomarkers:

| Biomarker | Description |
|-----------|-------------|
| **CRAE** | Central Retinal Artery Equivalent |
| **CRVE** | Central Retinal Vein Equivalent |
| **AVR** | Artery-to-Vein Ratio |
| **Artery Density** | Density of the arterial vessels |
| **Vein Density** | Density of the venous vessels |
| **Artery Fractal Dimension** | Complexity of the arterial branching pattern |
| **Vein Fractal Dimension** | Complexity of the venous branching pattern |

> **Note:** Although our pipeline also computes **CRAE** and **CRVE**, the official GAVE2 Task 3 evaluation is performed using only the following five biomarkers:
>
> - AVR
> - Artery Density
> - Vein Density
> - Artery Fractal Dimension
> - Vein Fractal Dimension

---

# Overall Pipeline

```text
                 Color Fundus Image (CFP)
                           │
                           ▼
        ┌────────────────────────────────────┐
        │ Task 1: Artery & Vein Segmentation │
        └────────────────────────────────────┘
                           │
                           ▼
             CFP + Fluorescein Angiography
                           │
                           ▼
      ┌──────────────────────────────────────────┐
      │ Task 2: Cross-Modal AV Segmentation      │
      └──────────────────────────────────────────┘
                           │
                           ▼
        Artery & Vein Segmentation Masks
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │ Task 3: Retinal Biomarker Quantification    │
    └─────────────────────────────────────────────┘
                           │
                           ▼
 CRAE • CRVE • AVR • Vessel Density • Fractal Dimension
```

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

> **Note:** Higher **COR** values and lower **INF** values indicate better performance. Since **COR + INF = 1.00**, improvements in COR correspond to reductions in INF.

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

> **Note:** Higher **COR** values and lower **INF** values indicate better performance. Since **COR + INF = 1.00**, improvements in COR correspond to reductions in INF.

---

## Task 3 — Retinal Biomarker Quantification

| Metric | Official Baseline | Ours (PixelPulse) |
|---------|------------------:|------------------:|
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

> **Note:** For Task 3, lower **MAE** and **SMAPE** values indicate better performance.

---

# Overall Challenge Performance

| Method | Task 1 | Task 2 | Task 3 | Overall Score |
|--------|--------:|--------:|--------:|--------------:|
| Official Baseline | 6.2039 | 6.2104 | 6.1672 | 6.1918 |
| **PixelPulse** | **8.0785** | **8.0845** | **7.4045** | **7.8113** |

---

## Leaderboard

| Rank | Team | Organization | Overall Score |
|-----:|------|--------------|--------------:|
| **11** | **PixelPulse** | Islamic University, Bangladesh | **7.8113** |

**Submission Time:** `2026-07-26 14:36`

---

# Repository Structure

```text
.
```

---

# Citation

```bibtex

```
