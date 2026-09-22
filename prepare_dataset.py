"""
prepare_dataset.py
------------------
Run this ONCE before starting the task. It will:

  1. Download CIFAR-10 via torchvision into ./data
  2. Fit the PCA cache used by the conditioning extractor and save it
     to pca_cache.npz (DO NOT delete or modify this file).
  3. Materialize a paired (image, condition) dataset to dataset.pt for fast loading.

After this you can import `PairedCIFAR10` from this file and use it directly:

    from prepare_dataset import PairedCIFAR10
    train_ds = PairedCIFAR10(split="train")
    val_ds   = PairedCIFAR10(split="val")
    test_ds  = PairedCIFAR10(split="test")

Splits: 45_000 train / 5_000 val / 10_000 test (test = CIFAR-10 test set).
Images are returned as float tensors in [-1, 1] of shape (3, 32, 32).
Condition vectors are returned as float tensors of shape (16,).

Usage:
    python prepare_dataset.py [--force] [--data_dir DIR] [--cache_path PATH]
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from pseudo_crossmodal import (
    COND_DIM,
    IMAGE_SIZE,
    PCAState,
    extract_condition,
    fit_pca,
    load_pca,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_CACHE = os.path.join(os.path.dirname(__file__), "dataset.pt")
SEED = 0


def to_chw_uint8(dataset: datasets.CIFAR10) -> np.ndarray:
    """Return CIFAR-10 image data as a (N, 3, 32, 32) uint8 numpy array (CHW)."""
    # torchvision stores .data as (N, 32, 32, 3) HWC uint8.
    return dataset.data.transpose(0, 3, 1, 2).astype(np.uint8)


def compute_conditions(images_unit: np.ndarray, pca: PCAState, batch_size: int = 1024) -> np.ndarray:
    """Compute the 16-dim conditioning vector for every image in chunks.

    :param images_unit: (N, 3, IMAGE_SIZE, IMAGE_SIZE) array of intensities in [0, 1].
    :param pca: Fitted PCA state.
    :param batch_size: Number of images per chunk; bounds peak memory.
    :return: (N, COND_DIM) float32 array of conditioning vectors.
    """
    conditions = np.empty((images_unit.shape[0], COND_DIM), dtype=np.float32)
    for start in range(0, images_unit.shape[0], batch_size):
        chunk = torch.from_numpy(images_unit[start:start + batch_size])
        conditions[start:start + batch_size] = extract_condition(chunk, pca).numpy()
    return conditions


def prepare(
    *, force: bool = False, data_dir: str = DATA_DIR, cache_path: str = DATASET_CACHE
) -> None:
    """Download CIFAR-10, fit PCA, and materialize the paired dataset cache.

    :param force: Re-run even if ``cache_path`` already exists.
    :param data_dir: Directory where torchvision should store raw CIFAR-10 files.
    :param cache_path: Destination path for the cached paired dataset (.pt).
    """
    if os.path.exists(cache_path) and not force:
        print(f"[prepare] dataset cache already exists at {cache_path}; skipping.")
        return

    os.makedirs(data_dir, exist_ok=True)
    print("[prepare] downloading CIFAR-10 (if needed)...")
    train_full = datasets.CIFAR10(data_dir, train=True, download=True)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True)

    train_full_images = to_chw_uint8(train_full)  # (50000, 3, 32, 32)
    train_full_labels = np.asarray(train_full.targets, dtype=np.int64)
    test_images = to_chw_uint8(test_set)
    test_labels = np.asarray(test_set.targets, dtype=np.int64)

    rng = np.random.default_rng(SEED)
    shuffled_idx = rng.permutation(len(train_full_images))
    val_idx = shuffled_idx[:5_000]
    train_idx = shuffled_idx[5_000:]

    train_images_unit = train_full_images[train_idx].astype(np.float32) / 255.0
    val_images_unit = train_full_images[val_idx].astype(np.float32) / 255.0
    test_images_unit = test_images.astype(np.float32) / 255.0

    print("[prepare] fitting PCA on the training split...")
    fit_pca(train_images_unit)  # saves to pca_cache.npz
    pca = load_pca()

    print("[prepare] computing conditioning vectors...")
    train_cond = compute_conditions(train_images_unit, pca)
    val_cond = compute_conditions(val_images_unit, pca)
    test_cond = compute_conditions(test_images_unit, pca)

    payload = {
        "train": {
            "images": torch.from_numpy(train_images_unit),  # (N, 3, 32, 32) in [0, 1]
            "cond": torch.from_numpy(train_cond),
            "labels": torch.from_numpy(train_full_labels[train_idx]),
        },
        "val": {
            "images": torch.from_numpy(val_images_unit),
            "cond": torch.from_numpy(val_cond),
            "labels": torch.from_numpy(train_full_labels[val_idx]),
        },
        "test": {
            "images": torch.from_numpy(test_images_unit),
            "cond": torch.from_numpy(test_cond),
            "labels": torch.from_numpy(test_labels),
        },
    }
    torch.save(payload, cache_path)
    print(f"[prepare] saved {cache_path}")
    print(
        f"          train / val / test = "
        f"{len(payload['train']['images'])} / "
        f"{len(payload['val']['images'])} / "
        f"{len(payload['test']['images'])}"
    )


class PairedCIFAR10(Dataset):
    """Paired (image, conditioning vector, label) CIFAR-10 dataset."""

    def __init__(self, split: str = "train", path: str = DATASET_CACHE) -> None:
        """Load one split of the cached paired dataset into memory.

        :param split: One of ``"train"``, ``"val"``, or ``"test"``.
        :param path: Path to the cache produced by ``prepare()``.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run `python prepare_dataset.py` first.")
        cache = torch.load(path)
        if split not in cache:
            raise KeyError(f"unknown split '{split}'; expected one of {list(cache)}")
        self.images = cache[split]["images"]  # (N, 3, 32, 32) in [0, 1]
        self.cond = cache[split]["cond"]
        self.labels = cache[split]["labels"]

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a single paired sample.

        :param idx: Sample index.
        :return: Tuple ``(image, cond, label)`` where ``image`` is (3, 32, 32) in
            [-1, 1], ``cond`` is (16,), and ``label`` is a scalar long tensor in
            [0, 9]. The label is provided for diagnostics only and the model
            must NOT consume it as input.
        """
        image = self.images[idx] * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return image, self.cond[idx], self.labels[idx]


def build_arg_parser() -> argparse.ArgumentParser:
    """argparse parser for the paired-dataset preparation CLI."""
    parser = argparse.ArgumentParser(description="Build the paired CIFAR-10 dataset cache")
    parser.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    parser.add_argument("--data_dir", default=DATA_DIR, help="where torchvision stores raw CIFAR-10")
    parser.add_argument("--cache_path", default=DATASET_CACHE, help="output .pt path")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if IMAGE_SIZE != 32:  # guard against accidental drift in pseudo_crossmodal.py
        raise RuntimeError(f"IMAGE_SIZE expected 32 for CIFAR-10; got {IMAGE_SIZE}")
    prepare(force=args.force, data_dir=args.data_dir, cache_path=args.cache_path)
    dataset = PairedCIFAR10("train", path=args.cache_path)
    image, condition, label = dataset[0]
    print("image:", image.shape, image.min().item(), image.max().item())
    print("cond :", condition.shape, condition[:4].tolist())
    print("label:", label.item())


if __name__ == "__main__":
    main()
