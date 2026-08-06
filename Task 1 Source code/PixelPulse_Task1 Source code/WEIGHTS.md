# Required model binaries

The lightweight source archive intentionally omits all model binaries.

## Public pretrained weights

Place the public weights at:

```text
external/HRVRL/weights/G_pretrain.pkl
external/RRWNet/weights/rrwnet_RITE_refinement.pth
```

Expected SHA-256 values:

```text
115eea6d33471910610a13685ae029094d1da4e497071c5aad95a49a05acf7e6  external/HRVRL/weights/G_pretrain.pkl
9881bf5c86c94da9099fc141fff3f738df241c6567f9d366469a83dee54a048f  external/RRWNet/weights/rrwnet_RITE_refinement.pth
```

Sources:

- <https://github.com/sulab-wmu/HRVRL>
- <https://github.com/j-morano/rrwnet>

## PixelPulse fine-tuned checkpoints

For exact inference, place the five selected checkpoints at:

```text
models/hrvrl/fold_0_best.pt
models/hrvrl/fold_1_best.pt
models/hrvrl/fold_2_best.pt
models/hrvrl/fold_3_best.pt
models/hrvrl/fold_4_best.pt
```

Expected SHA-256 values:

```text
c0fe756cd8962efd9311fa1ba44fd026cf090a0ff8c1384086e26fa985e8a272  models/hrvrl/fold_0_best.pt
217844ab69f05e2dbd057ba86c49d7eac62ec7698c012c4c1bcfa1602463d6b7  models/hrvrl/fold_1_best.pt
ef4ffb669065d7fec13b6164a12bf3f51021eda65ae1a5a10dabefe092c21474  models/hrvrl/fold_2_best.pt
d286a6310b9c2bf16f84163500e9009e1cffb07354364ee9170730353082bd14  models/hrvrl/fold_3_best.pt
58993995decdf6b840bbf36aafa422766308d09ed4646423e06cebfcbfe9587b  models/hrvrl/fold_4_best.pt
```

These are private team-generated checkpoints. They can be supplied separately
to the organizers if model-weight verification is requested. Alternatively,
run `train_task1_folds.sh` to retrain the five folds.

The full private backup archive containing all seven binaries is:

```text
PixelPulse_Task1_8_16666_Code_Models.zip
```
