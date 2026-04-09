"""Vascular and tissue segmentation helpers used by the workflow classes."""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes, gaussian_filter

from .config import SegmentationConfig, SegmentationResult

LOGGER = logging.getLogger(__name__)


def _log(verbose: bool, message: str, *args) -> None:
    if verbose:
        LOGGER.info(message, *args)


def _segment_enhancement_series(img_init, config: SegmentationConfig | None = None) -> SegmentationResult:
    config = config or SegmentationConfig()

    x = np.abs(np.asarray(img_init))
    mean_img = x.mean(axis=0)
    peak_img = x.max(axis=0)
    std_img = x.std(axis=0)

    brain_threshold = float(np.percentile(mean_img, config.brain_percentile))
    brain_mask = mean_img > brain_threshold
    brain_mask = ndi.binary_closing(brain_mask, iterations=2)
    brain_mask = ndi.binary_fill_holes(brain_mask)

    labels, num_labels = ndi.label(brain_mask)
    sizes = ndi.sum(brain_mask, labels, index=np.arange(1, num_labels + 1))
    if num_labels > 0:
        brain_mask = labels == (np.argmax(sizes) + 1)

    brain_core_mask = ndi.binary_erosion(brain_mask, iterations=config.core_erosion_iters)
    if not np.any(brain_core_mask):
        brain_core_mask = brain_mask.copy()
    brain_ring_mask = brain_mask & (~brain_core_mask)

    pca_roi_mask = ndi.binary_erosion(brain_core_mask, iterations=config.roi_erosion_iters)
    if not np.any(pca_roi_mask):
        pca_roi_mask = brain_core_mask.copy()

    num_early = int(config.early_duration_sec / config.frame_time_sec)
    baseline_idx = np.arange(0, min(config.n_baseline, x.shape[0]))
    early_idx = np.arange(0, min(num_early, x.shape[0]))

    baseline_mean = x[baseline_idx].mean(axis=0)
    early_sum = x[early_idx].sum(axis=0)
    norm_early_enh = (early_sum - len(early_idx) * baseline_mean) / np.maximum(baseline_mean, 1e-6)
    norm_early_enh = ndi.gaussian_filter(norm_early_enh, sigma=1.0)

    enh_vals = norm_early_enh[pca_roi_mask]
    if enh_vals.size == 0:
        pca_roi_mask = brain_mask.copy()
        enh_vals = norm_early_enh[pca_roi_mask]
    enh_threshold = float(np.percentile(enh_vals, config.enhancement_percentile))

    vascular_mask = pca_roi_mask & (norm_early_enh > enh_threshold)
    vascular_mask = ndi.binary_opening(vascular_mask, iterations=config.cleanup_open_iters)
    vascular_mask = ndi.binary_closing(vascular_mask, iterations=config.cleanup_close_iters)

    labels, num_labels = ndi.label(vascular_mask)
    sizes = ndi.sum(vascular_mask, labels, index=np.arange(1, num_labels + 1)) if num_labels > 0 else np.array([])
    keep = np.where(sizes >= config.cleanup_min_size)[0] + 1
    vascular_mask = np.isin(labels, keep) if keep.size > 0 else vascular_mask
    vascular_mask = vascular_mask & pca_roi_mask
    tissue_mask = pca_roi_mask & (~vascular_mask)

    _log(config.verbose, "segmentation vascular=%d tissue=%d", int(vascular_mask.sum()), int(tissue_mask.sum()))

    return SegmentationResult(
        mean_img=mean_img,
        peak_img=peak_img,
        std_img=std_img,
        baseline_mean=baseline_mean,
        norm_early_enh=norm_early_enh,
        brain_mask=brain_mask.astype(bool),
        brain_core_mask=brain_core_mask.astype(bool),
        brain_ring_mask=brain_ring_mask.astype(bool),
        pca_roi_mask=pca_roi_mask.astype(bool),
        vascular_mask=vascular_mask.astype(bool),
        tissue_mask=tissue_mask.astype(bool),
        brain_mask_threshold=brain_threshold,
        enhancement_threshold=enh_threshold,
        baseline_idx=baseline_idx,
        early_idx=early_idx,
    )


def _segment_dynamic_series(img_dyn, config: SegmentationConfig | None = None) -> SegmentationResult:
    config = config or SegmentationConfig()
    x = np.abs(np.asarray(img_dyn))
    if x.ndim != 3:
        raise ValueError(f"img_dyn must have shape (T, H, W), got {x.shape}")

    mean_img = np.mean(x, axis=0).astype(np.float32, copy=False)
    peak_img = np.max(x, axis=0).astype(np.float32, copy=False)
    mean_img_s = gaussian_filter(mean_img, sigma=config.smooth_sigma)

    brain_threshold = float(np.percentile(mean_img_s, config.brain_percentile))
    brain_mask = mean_img_s > brain_threshold
    if config.opening_iters > 0:
        brain_mask = ndi.binary_opening(brain_mask, iterations=config.opening_iters)
    if config.closing_iters > 0:
        brain_mask = ndi.binary_closing(brain_mask, iterations=config.closing_iters)
    brain_mask = binary_fill_holes(brain_mask)
    if config.erosion_iters > 0:
        brain_mask = ndi.binary_erosion(brain_mask, iterations=config.erosion_iters)

    enhancement_score = (peak_img - mean_img) / np.maximum(mean_img, 1e-6)
    enhancement_score = gaussian_filter(enhancement_score.astype(np.float32), sigma=1.0)

    if not np.any(brain_mask):
        raise ValueError("Computed brain_mask is empty. Try lowering brain_percentile.")

    vessel_threshold = float(np.percentile(enhancement_score[brain_mask], config.vessel_percentile))
    vascular_mask = (enhancement_score >= vessel_threshold) & brain_mask
    vascular_mask = ndi.binary_opening(vascular_mask, iterations=1)
    vascular_mask = ndi.binary_closing(vascular_mask, iterations=1)
    vascular_mask = binary_fill_holes(vascular_mask)
    tissue_mask = brain_mask & (~vascular_mask)

    _log(config.verbose, "dynamic segmentation vascular=%d tissue=%d", int(vascular_mask.sum()), int(tissue_mask.sum()))

    return SegmentationResult(
        mean_img=mean_img,
        peak_img=peak_img,
        std_img=None,
        baseline_mean=None,
        norm_early_enh=None,
        brain_mask=brain_mask.astype(bool),
        brain_core_mask=None,
        brain_ring_mask=None,
        pca_roi_mask=None,
        vascular_mask=vascular_mask.astype(bool),
        tissue_mask=tissue_mask.astype(bool),
        brain_mask_threshold=brain_threshold,
        enhancement_threshold=vessel_threshold,
        mean_img_s=mean_img_s,
        enhancement_score=enhancement_score,
    )
