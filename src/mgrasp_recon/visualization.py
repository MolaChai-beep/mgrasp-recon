"""Visualization helpers kept separate from core numerical code."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_time_intensity_curves(
    vessel_curve: np.ndarray,
    vessel_coord: tuple[int, int],
    tissue_curves: list[np.ndarray] | None = None,
    tissue_coords: list[tuple[int, int]] | None = None,
    frame_time_sec: float | None = None,
    normalize: bool = False,
    title: str = "Time-intensity curves",
):
    tissue_curves = tissue_curves or []
    tissue_coords = tissue_coords or []

    if frame_time_sec is None:
        x = np.arange(len(vessel_curve))
        xlabel = "Frame"
    else:
        x = np.arange(len(vessel_curve)) * float(frame_time_sec)
        xlabel = "Time (s)"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        vessel_curve,
        linewidth=2.5,
        color="crimson",
        label=f"Top vessel voxel {tuple(map(int, vessel_coord))}",
    )

    for i, (curve, coord) in enumerate(zip(tissue_curves, tissue_coords), start=1):
        ax.plot(
            x,
            curve,
            linewidth=1.8,
            linestyle="--",
            label=f"Tissue {i} {tuple(map(int, coord))}",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized intensity" if normalize else "Intensity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def show_selected_voxels(
    background_img: np.ndarray,
    vessel_coord: tuple[int, int],
    tissue_coords: list[tuple[int, int]] | None = None,
    vascular_mask: np.ndarray | None = None,
    tissue_mask: np.ndarray | None = None,
    title: str = "Selected voxels",
):
    tissue_coords = tissue_coords or []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.asarray(background_img), cmap="gray")

    if vascular_mask is not None:
        ax.contour(
            np.asarray(vascular_mask).astype(bool),
            levels=[0.5],
            colors=["r"],
            linewidths=1.0,
        )
    if tissue_mask is not None:
        ax.contour(
            np.asarray(tissue_mask).astype(bool),
            levels=[0.5],
            colors=["lime"],
            linewidths=0.8,
        )

    vr, vc = map(int, vessel_coord)
    ax.scatter([vc], [vr], c="crimson", s=70, marker="o", label="Top vessel voxel")

    if tissue_coords:
        tc = np.asarray([(int(r), int(c)) for r, c in tissue_coords])
        ax.scatter(tc[:, 1], tc[:, 0], c="cyan", s=55, marker="x", label="Tissue voxels")

    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax


def plot_segmentation_summary(
    mean_img: np.ndarray,
    baseline_mean: np.ndarray,
    norm_early_enh: np.ndarray,
    std_img: np.ndarray,
    brain_mask: np.ndarray,
    brain_core_mask: np.ndarray,
    brain_ring_mask: np.ndarray,
    pca_roi_mask: np.ndarray,
    vascular_mask: np.ndarray,
    tissue_mask: np.ndarray,
    enhancement_threshold: float,
):
    overlay = np.zeros(mean_img.shape + (3,), dtype=np.float32)
    overlay[..., 0] = vascular_mask.astype(np.float32)
    overlay[..., 1] = tissue_mask.astype(np.float32)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    plots = [
        (mean_img, "gray", "Mean image"),
        (baseline_mean, "gray", "Baseline mean"),
        (norm_early_enh, "hot", "Normalized early enhancement"),
        (std_img, "hot", "Temporal std"),
        (brain_mask.astype(float), "gray", "Brain mask"),
        (brain_core_mask.astype(float), "gray", "Brain core mask"),
        (brain_ring_mask.astype(float), "gray", "Brain ring mask"),
        (pca_roi_mask.astype(float), "gray", "PCA ROI mask"),
        (norm_early_enh * pca_roi_mask, "hot", "Enhancement inside ROI"),
        ((norm_early_enh > enhancement_threshold).astype(float), "gray", "Enhancement threshold mask"),
        (vascular_mask.astype(float), "gray", "Vascular mask"),
        (mean_img, "gray", "Red=vascular, Green=tissue"),
    ]

    for ax, (image, cmap, title) in zip(axes.ravel(), plots):
        ax.imshow(image, cmap=cmap)
        if title == "Red=vascular, Green=tissue":
            ax.imshow(overlay, alpha=0.40)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def plot_segmentation_frames(
    frames: np.ndarray,
    overlay: np.ndarray,
):
    frames_to_show = [0, frames.shape[0] // 2, frames.shape[0] - 1]
    fig, axes = plt.subplots(2, len(frames_to_show), figsize=(12, 8))

    for i, t in enumerate(frames_to_show):
        axes[0, i].imshow(frames[t], cmap="gray")
        axes[0, i].set_title(f"Frame {t}")
        axes[0, i].axis("off")

        axes[1, i].imshow(frames[t], cmap="gray")
        axes[1, i].imshow(overlay, alpha=0.35)
        axes[1, i].set_title(f"Frame {t} + masks")
        axes[1, i].axis("off")

    fig.tight_layout()
    return fig, axes
