# PixelPulse GAVE2 Task 1 Preliminary Result

**Team:** PixelPulse  
**Task:** Task 1 — CFP-only artery/vein segmentation  
**Official Task 1 score:** **8.16666**  
**Status:** Finished  
**Submission time:** 2026-07-30 12:31

## Official metrics

| Class | AV_DSC | Sensitivity | Specificity | Accuracy | INF ↓ | COR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Artery | 0.6702 | 0.9521 | 0.9589 | 0.9587 | 0.2614 | 0.7290 |
| Vein | 0.6702 | 0.9639 | 0.9569 | 0.9572 | 0.2290 | 0.7558 |

The GAVE2 leaderboard exposes one shared `Task1_AV_DSC` value rather than
separate artery and vein Dice values.

## Score reconstruction

```text
Task1 =
    2 × AV_DSC
  + (4/6) × (A_Sen + A_Spe + A_Acc + V_Sen + V_Spe + V_Acc)
  + (1 - A_INF) + A_COR + (1 - V_INF) + V_COR

Dice           = 1.3404
Classification = 3.8318
Topology       = 2.9944
Displayed sum  = 8.1666
Official score = 8.16666
```

## Comparison with organizer baseline

| Result | Task 1 score | Absolute difference |
|---|---:|---:|
| Organizer baseline | 6.20390 | — |
| PixelPulse | **8.16666** | **+1.96276** |

| Class | DSC | Sensitivity | Specificity | Accuracy | INF ↓ | COR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Artery baseline | 0.7656 | 0.6709 | 0.9937 | 0.9828 | 0.6924 | 0.3058 |
| Vein baseline | 0.7656 | 0.7146 | 0.9930 | 0.9808 | 0.7672 | 0.2322 |
| Artery PixelPulse | 0.6702 | 0.9521 | 0.9589 | 0.9587 | 0.2614 | 0.7290 |
| Vein PixelPulse | 0.6702 | 0.9639 | 0.9569 | 0.9572 | 0.2290 | 0.7558 |

## Exact Task 1 payload

```text
Directory:
artifacts/frozen_task1_8_16666/Task1/

Per-file checksum manifest:
artifacts/frozen_task1_8_16666/SHA256SUMS

SHA-256 of SHA256SUMS:
d239841cd10ac62030f1811ec41962dcff81b531d1c783107fbe4ad88a3ecfa1
```

The directory contains 50 RGB PNG files for `g_051`–`g_100`. All files are
1024 × 1536, zero outside the retinal ROI, and use channel order
`(artery, vessel union, vein)`.

## Repeated evaluation note

The same Task 1 payload received 8.14713 at 2026-07-30 01:56 and 8.16666 at
2026-07-30 12:31. Dice and pixel-classification metrics were effectively
unchanged; the difference came from small changes in path-based INF/COR
measurements, consistent with stochastic path sampling by the evaluator.
