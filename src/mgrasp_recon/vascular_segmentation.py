# import function
import argparse
import inspect
import h5py
import os
import pathlib
import numpy as np
import csv

import re
import glob
import matplotlib.pyplot as plt

from ._bootstrap import ensure_repo_paths

ensure_repo_paths()

import sigpy as sp

import importlib
from .espirit import EspiritCalib

from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_opening, binary_closing, binary_erosion
from scipy import ndimage as ndi

def build_vascular_tissue_masks_from_enhancement(
    img_init,
    frame_time_sec=3.8,
    n_baseline=5,
    early_duration_sec=120,
    brain_percentile=60,
    core_erosion_iters=12,
    roi_erosion_iters=8,
    enhancement_percentile=75,
    cleanup_min_size=10,
    cleanup_open_iters=1,
    cleanup_close_iters=1,
    show_plots=True,
):

    X = np.abs(img_init)
    mean_img = X.mean(axis=0)
    peak_img = X.max(axis=0)
    std_img = X.std(axis=0)

    # 1. brain/head mask from mean image
    thr = np.percentile(mean_img, brain_percentile)
    brain_mask = mean_img > thr
    brain_mask = ndi.binary_closing(brain_mask, iterations=2)
    brain_mask = ndi.binary_fill_holes(brain_mask)

    lab, nlab = ndi.label(brain_mask)
    sizes = ndi.sum(brain_mask, lab, index=np.arange(1, nlab + 1))
    brain_mask = (lab == (np.argmax(sizes) + 1))

    # 2. remove outer ring region
    brain_core_mask = ndi.binary_erosion(brain_mask, iterations=core_erosion_iters)

    # 3. edge band
    brain_ring_mask = brain_mask & (~brain_core_mask)

    # 4. PCA ROI
    pca_roi_mask = ndi.binary_erosion(brain_core_mask, iterations=roi_erosion_iters)

    # 5. normalized early enhancement
    n_early = int(early_duration_sec / frame_time_sec)

    baseline_idx = np.arange(0, min(n_baseline, X.shape[0]))
    early_idx = np.arange(0, min(n_early, X.shape[0]))

    baseline_mean = X[baseline_idx].mean(axis=0)
    early_sum = X[early_idx].sum(axis=0)

    norm_early_enh = (
        early_sum - len(early_idx) * baseline_mean
    ) / np.maximum(baseline_mean, 1e-6)

    norm_early_enh = ndi.gaussian_filter(norm_early_enh, sigma=1.0)

    # 6. threshold normalized enhancement directly
    enh_vals = norm_early_enh[pca_roi_mask]
    enh_thr = np.percentile(enh_vals, enhancement_percentile)

    vascular_mask = pca_roi_mask & (norm_early_enh > enh_thr)

    # 7. cleanup vascular mask
    vascular_mask = ndi.binary_opening(vascular_mask, iterations=cleanup_open_iters)
    vascular_mask = ndi.binary_closing(vascular_mask, iterations=cleanup_close_iters)

    lab, nlab = ndi.label(vascular_mask)
    sizes = ndi.sum(vascular_mask, lab, index=np.arange(1, nlab + 1))
    keep = np.where(sizes >= cleanup_min_size)[0] + 1
    vascular_mask = np.isin(lab, keep)

    vascular_mask = vascular_mask & pca_roi_mask

    # 8. tissue mask
    tissue_mask = pca_roi_mask & (~vascular_mask)

    results = {
        "X": X,
        "mean_img": mean_img,
        "peak_img": peak_img,
        "std_img": std_img,
        "baseline_mean": baseline_mean,
        "norm_early_enh": norm_early_enh,
        "brain_mask": brain_mask,
        "brain_core_mask": brain_core_mask,
        "brain_ring_mask": brain_ring_mask,
        "pca_roi_mask": pca_roi_mask,
        "vascular_mask": vascular_mask,
        "tissue_mask": tissue_mask,
        "brain_mask_threshold": float(thr),
        "enhancement_threshold": float(enh_thr),
        "baseline_idx": baseline_idx,
        "early_idx": early_idx,
    }

    print("brain mask thr:", float(thr))
    print("enh thr:", float(enh_thr))
    print("brain_mask voxels:", int(brain_mask.sum()))
    print("brain_core_mask voxels:", int(brain_core_mask.sum()))
    print("brain_ring_mask voxels:", int(brain_ring_mask.sum()))
    print("pca_roi_mask voxels:", int(pca_roi_mask.sum()))
    print("vascular voxels:", int(vascular_mask.sum()))
    print("tissue voxels:", int(tissue_mask.sum()))

    if show_plots:
        overlay = np.zeros(mean_img.shape + (3,), dtype=np.float32)
        overlay[..., 0] = vascular_mask.astype(np.float32)   # red
        overlay[..., 1] = tissue_mask.astype(np.float32)     # green

        plt.figure(figsize=(20, 12))

        plt.subplot(3, 4, 1)
        plt.imshow(mean_img, cmap="gray")
        plt.title("Mean image")
        plt.axis("off")

        plt.subplot(3, 4, 2)
        plt.imshow(baseline_mean, cmap="gray")
        plt.title("Baseline mean")
        plt.axis("off")

        plt.subplot(3, 4, 3)
        plt.imshow(norm_early_enh, cmap="hot")
        plt.title("Normalized early enhancement")
        plt.axis("off")

        plt.subplot(3, 4, 4)
        plt.imshow(std_img, cmap="hot")
        plt.title("Temporal std")
        plt.axis("off")

        plt.subplot(3, 4, 5)
        plt.imshow(brain_mask.astype(float), cmap="gray", vmin=0, vmax=1)
        plt.title("Brain mask")
        plt.axis("off")

        plt.subplot(3, 4, 6)
        plt.imshow(brain_core_mask.astype(float), cmap="gray", vmin=0, vmax=1)
        plt.title("Brain core mask")
        plt.axis("off")

        plt.subplot(3, 4, 7)
        plt.imshow(brain_ring_mask.astype(float), cmap="gray", vmin=0, vmax=1)
        plt.title("Brain ring mask")
        plt.axis("off")

        plt.subplot(3, 4, 8)
        plt.imshow(pca_roi_mask.astype(float), cmap="gray", vmin=0, vmax=1)
        plt.title("PCA ROI mask")
        plt.axis("off")

        plt.subplot(3, 4, 9)
        plt.imshow(norm_early_enh * pca_roi_mask, cmap="hot")
        plt.title("Enhancement inside ROI")
        plt.axis("off")

        plt.subplot(3, 4, 10)
        plt.imshow((norm_early_enh > enh_thr).astype(float), cmap="gray", vmin=0, vmax=1)
        plt.title("Enhancement threshold mask")
        plt.axis("off")

        plt.subplot(3, 4, 11)
        plt.imshow(vascular_mask.astype(float), cmap="gray", vmin=0, vmax=1)
        plt.title("Vascular mask")
        plt.axis("off")

        plt.subplot(3, 4, 12)
        plt.imshow(mean_img, cmap="gray")
        plt.imshow(overlay, alpha=0.40)
        plt.title("Red=vascular, Green=tissue")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        frames_to_show = [0, X.shape[0] // 2, X.shape[0] - 1]

        plt.figure(figsize=(12, 8))
        for i, t in enumerate(frames_to_show):
            plt.subplot(2, len(frames_to_show), i + 1)
            plt.imshow(X[t], cmap="gray")
            plt.title(f"Frame {t}")
            plt.axis("off")

            plt.subplot(2, len(frames_to_show), i + 1 + len(frames_to_show))
            plt.imshow(X[t], cmap="gray")
            plt.imshow(overlay, alpha=0.35)
            plt.title(f"Frame {t} + masks")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    return results


def build_dynamic_segmentation_masks(
    img_dyn,
    brain_percentile=50,
    vessel_percentile=90,
    smooth_sigma=2.0,
    opening_iters=1,
    closing_iters=1,
    erosion_iters=0,
):
    """
    Build coarse masks for brain tissue vs vascular voxels from a dynamic series.

    Parameters
    ----------
    img_dyn : ndarray
        Dynamic image series with shape (T, H, W). Real or complex input is allowed.
    brain_percentile : float
        Percentile on the smoothed mean image used to define the coarse brain mask.
    vessel_percentile : float
        Percentile on the normalized enhancement score inside the brain mask used
        to define the vascular mask.
    smooth_sigma : float
        Gaussian sigma applied to the temporal mean image for a stable brain mask.
    opening_iters, closing_iters, erosion_iters : int
        Morphology iterations for cleaning the masks.

    Returns
    -------
    dict
        {
            "mean_img": (H, W),
            "mean_img_s": (H, W),
            "peak_img": (H, W),
            "enhancement_score": (H, W),
            "brain_mask": (H, W) bool,
            "vascular_mask": (H, W) bool,
            "tissue_mask": (H, W) bool,
            "brain_threshold": float,
            "vessel_threshold": float,
        }
    """
    X = np.abs(np.asarray(img_dyn))
    if X.ndim != 3:
        raise ValueError(f"img_dyn must have shape (T, H, W), got {X.shape}")

    mean_img = np.mean(X, axis=0).astype(np.float32, copy=False)
    peak_img = np.max(X, axis=0).astype(np.float32, copy=False)
    mean_img_s = gaussian_filter(mean_img, sigma=smooth_sigma)

    brain_threshold = float(np.percentile(mean_img_s, brain_percentile))
    brain_mask = mean_img_s > brain_threshold

    if opening_iters > 0:
        brain_mask = ndi.binary_opening(brain_mask, iterations=opening_iters)
    if closing_iters > 0:
        brain_mask = ndi.binary_closing(brain_mask, iterations=closing_iters)
    brain_mask = binary_fill_holes(brain_mask)
    if erosion_iters > 0:
        brain_mask = ndi.binary_erosion(brain_mask, iterations=erosion_iters)

    enhancement_score = (peak_img - mean_img) / np.maximum(mean_img, 1e-6)
    enhancement_score = gaussian_filter(enhancement_score.astype(np.float32), sigma=1.0)

    if not np.any(brain_mask):
        raise ValueError("Computed brain_mask is empty. Try lowering brain_percentile.")

    vessel_threshold = float(
        np.percentile(enhancement_score[brain_mask], vessel_percentile)
    )
    vascular_mask = (enhancement_score >= vessel_threshold) & brain_mask
    vascular_mask = ndi.binary_opening(vascular_mask, iterations=1)
    vascular_mask = ndi.binary_closing(vascular_mask, iterations=1)
    vascular_mask = binary_fill_holes(vascular_mask)

    tissue_mask = brain_mask & (~vascular_mask)

    return {
        "mean_img": mean_img,
        "mean_img_s": mean_img_s,
        "peak_img": peak_img,
        "enhancement_score": enhancement_score,
        "brain_mask": brain_mask.astype(bool),
        "vascular_mask": vascular_mask.astype(bool),
        "tissue_mask": tissue_mask.astype(bool),
        "brain_threshold": brain_threshold,
        "vessel_threshold": vessel_threshold,
    }

    

