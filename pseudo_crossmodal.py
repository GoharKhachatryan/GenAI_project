"""
pseudo_crossmodal.py
--------------------
Fixed extractor that maps a 32x32 RGB image (values in [0, 1]) to a
16-dimensional conditioning vector. Do NOT modify this file. The conditioning
signal must be deterministic and identical across runs.

Layout of the 16-dim vector:
    indices 0..3   : row-sum quartile means on luminance (4 features)
    indices 4..7   : column-sum quartile means on luminance (4 features)
    indices 8..10  : per-channel mean intensity (R, G, B) (3 features)
    indices 11     : global std of luminance (1 feature)
    indices 12..15 : 4 PCA components fit on flattened RGB CIFAR-10 train set

Luminance is computed with ITU-R BT.601 weights (0.299 R + 0.587 G + 0.114 B).
The PCA fit is cached to disk (pca_cache.npz) so all runs use the same
projection. Run `prepare_dataset.py` once to materialize it.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA

PCA_CACHE_PATH = os.path.join(os.path.dirname(__file__), "pca_cache.npz")
PCA_N_COMPONENTS = 4
COND_DIM = 16
IMAGE_SIZE = 32
QUARTILE_LEN = IMAGE_SIZE // 4  # 8 pixels per quartile

# ITU-R BT.601 luminance weights (R, G, B).
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


@dataclass
class PCAState:
    """Cached PCA projection used by the conditioning extractor."""

    mean: np.ndarray  # (3072,)
    components: np.ndarray  # (PCA_N_COMPONENTS, 3072)
    scale: np.ndarray  # (PCA_N_COMPONENTS,)  -- std-dev per component for normalization


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to luminance with BT.601 weights.

    :param rgb: (B, 3, H, W) tensor in [0, 1].
    :return: (B, H, W) luminance tensor.
    """
    weight_red, weight_green, weight_blue = LUMA_WEIGHTS
    return weight_red * rgb[:, 0] + weight_green * rgb[:, 1] + weight_blue * rgb[:, 2]


def row_col_quartile_means(luma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean row-sum and column-sum within four contiguous quartile bands.

    :param luma: (B, IMAGE_SIZE, IMAGE_SIZE) luminance tensor in [0, 1].
    :return: Tuple of (B, 4) row-quartile means and (B, 4) column-quartile means.
    """
    row_sums = luma.sum(dim=2)  # (B, IMAGE_SIZE)
    col_sums = luma.sum(dim=1)  # (B, IMAGE_SIZE)
    row_quartile_means = row_sums.view(-1, 4, QUARTILE_LEN).mean(dim=2)  # (B, 4)
    col_quartile_means = col_sums.view(-1, 4, QUARTILE_LEN).mean(dim=2)  # (B, 4)
    return row_quartile_means, col_quartile_means


def fit_pca(train_images: np.ndarray, save_path: str = PCA_CACHE_PATH) -> PCAState:
    """Fit PCA on flattened RGB training images. Call once during dataset prep.

    :param train_images: (N, 3, IMAGE_SIZE, IMAGE_SIZE) array of intensities in [0, 1].
    :param save_path: Destination path for the cached PCA state (.npz).
    :return: Fitted PCAState (also written to ``save_path``).
    """
    if train_images.ndim != 4 or train_images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"expected (N, 3, {IMAGE_SIZE}, {IMAGE_SIZE}); got shape {train_images.shape}"
        )
    flat_images: np.ndarray = train_images.reshape(train_images.shape[0], -1).astype(np.float64)
    # svd_solver="full" is required: the "auto" default would pick the stochastic
    # "randomized" solver at this shape, breaking run-to-run determinism.
    pca = PCA(n_components=PCA_N_COMPONENTS, svd_solver="full")
    pca.fit(flat_images)
    # standardize each component projection so all 16 features are roughly comparable
    scale = np.sqrt(pca.explained_variance_) + 1e-8
    state = PCAState(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        scale=scale.astype(np.float32),
    )
    np.savez(save_path, mean=state.mean, components=state.components, scale=state.scale)
    return state


def load_pca(path: str = PCA_CACHE_PATH) -> PCAState:
    """Load a previously fitted PCA cache from disk.

    :param path: Path to a ``.npz`` file produced by ``fit_pca``.
    :return: Reconstructed PCAState.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PCA cache not found at {path}. Run prepare_dataset.py first.")
    cache = np.load(path)
    return PCAState(mean=cache["mean"], components=cache["components"], scale=cache["scale"])


def extract_condition(images: torch.Tensor, pca: PCAState) -> torch.Tensor:
    """Map a batch of RGB images to 16-dim conditioning vectors.

    The mapping is differentiable w.r.t. the input images, so it can be used
    inside a training loss if desired. That is why the projection is applied
    with torch ops here instead of ``sklearn``'s ``PCA.transform``.

    :param images: Float tensor of shape (B, 3, IMAGE_SIZE, IMAGE_SIZE); values in
        [0, 1] or [-1, 1] (auto-detected by the extractor).
    :param pca: Fitted PCAState from ``load_pca``.
    :return: Tensor of shape (B, 16) of conditioning features.
    """
    if images.dim() != 4 or images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"bad shape {tuple(images.shape)}; expected (B, 3, {IMAGE_SIZE}, {IMAGE_SIZE})"
        )

    # bring images into [0, 1] if they are in [-1, 1]
    if images.min() < -0.01:
        images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)

    luma = luminance(images)  # (B, H, W)
    row_quartile_means, col_quartile_means = row_col_quartile_means(luma)  # (B, 4), (B, 4)
    channel_mean = images.mean(dim=(2, 3))  # (B, 3) -- R, G, B means
    luma_std = luma.std(dim=(1, 2)).unsqueeze(1)  # (B, 1)

    flat_images = images.reshape(images.shape[0], -1)  # (B, 3072)
    pca_mean = torch.as_tensor(pca.mean, dtype=flat_images.dtype, device=flat_images.device)
    pca_components = torch.as_tensor(pca.components, dtype=flat_images.dtype, device=flat_images.device)
    pca_scale = torch.as_tensor(pca.scale, dtype=flat_images.dtype, device=flat_images.device)
    projections = (flat_images - pca_mean) @ pca_components.T  # (B, k)
    projections = projections / pca_scale  # standardized

    condition = torch.cat(
        [row_quartile_means, col_quartile_means, channel_mean, luma_std, projections], dim=1
    )
    if condition.shape[1] != COND_DIM:
        raise ValueError(f"cond dim mismatch: {condition.shape[1]} (expected {COND_DIM})")
    return condition


if __name__ == "__main__":
    # Quick sanity check
    rng = np.random.default_rng(0)
    random_images = rng.random((100, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    fitted_pca = fit_pca(random_images, save_path="/tmp/_pca_test.npz")
    condition = extract_condition(torch.from_numpy(random_images), fitted_pca)
    print("cond shape:", condition.shape)
    print("per-feature mean:", condition.mean(dim=0))
    print("per-feature std :", condition.std(dim=0))
