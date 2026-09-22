# Test task — setup

The task itself is described in [TASK.md](TASK.md). This README only covers
how to bring the skeleton up so you can start.

## Requirements

- Python ≥ 3.11
- A single GPU (CUDA / MPS) or CPU is fine — Colab T4 is enough.
- `pip install torch torchvision numpy scikit-learn` is all the skeleton needs
  (`scikit-learn` is only used to fit the PCA during setup).
  Anything else (matplotlib, jupyter, etc.) is up to you.

## One-time setup

Run these once, in order, from this directory:

```bash
python prepare_dataset.py     # downloads CIFAR-10, fits PCA, writes dataset.pt
python quality_probe.py       # trains the fixed quality-probe classifier (a few minutes on CPU)
```

After this you will have:

| File | What it is |
|------|------------|
| `data/` | Raw CIFAR-10 cache from torchvision |
| `pca_cache.npz` | PCA fit used by `extract_condition`. **Do not delete or modify.** |
| `dataset.pt` | Paired (image, condition, label) tensors, all three splits |
| `probe_weights.pt` | Frozen CIFAR-10 classifier used as the external quality judge |

All four are gitignored.

## Importing the dataset

```python
from prepare_dataset import PairedCIFAR10

train_dataset = PairedCIFAR10("train")
val_dataset   = PairedCIFAR10("val")
test_dataset  = PairedCIFAR10("test")

image, cond, label = train_dataset[0]
# image: (3, 32, 32) float in [-1, 1]
# cond:  (16,) float
# label: scalar long in [0, 9]  -- diagnostic only; the model must NOT see it
```

## What you may / must not change

- `pseudo_crossmodal.py` is fixed. The conditioning signal must be
  deterministic across runs — do not modify it.
- Everything else (model, training loop, samplers, evaluation notebooks) is
  yours to write.

## Deliverable

Commit one or several Jupyter notebooks plus any supporting `.py` modules,
together with a short top-level README describing how to reproduce your
results (commands, expected runtime, what each notebook produces). See
[TASK.md](TASK.md) for the full prompt and what we look for.
