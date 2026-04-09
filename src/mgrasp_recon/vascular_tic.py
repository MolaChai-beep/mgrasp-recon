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

def extract_voxel_tic(img_dyn, coord, normalize=False):
    """
    Extract the time-intensity curve for a single voxel.

    Parameters
    ----------
    img_dyn : ndarray
        Dynamic image series with shape (T, H, W).
    coord : tuple[int, int]
        Voxel coordinate in (row, col) order.
    normalize : bool
        If True, divide by the baseline mean from the first 5 frames.
    """
    X = np.abs(np.asarray(img_dyn))
    if X.ndim != 3:
        raise ValueError(f"img_dyn must have shape (T, H, W), got {X.shape}")

    r, c = map(int, coord)
    if not (0 <= r < X.shape[1] and 0 <= c < X.shape[2]):
        raise IndexError(f"coord {coord} is outside image bounds {X.shape[1:]}")

    curve = X[:, r, c].astype(np.float32, copy=False)

    if normalize:
        n_baseline = min(5, len(curve))
        baseline = float(np.mean(curve[:n_baseline]))
        curve = curve / max(baseline, 1e-6)

    return curve





def plot_time_intensity_curves(
    img_dyn,
    vessel_coord,
    tissue_coords=None,
    frame_time_sec=None,
    normalize=False,
    title="Time-intensity curves",
):
    """
    Plot the vessel TIC together with one or more tissue TICs.
    """
    X = np.abs(np.asarray(img_dyn))
    if X.ndim != 3:
        raise ValueError(f"img_dyn must have shape (T, H, W), got {X.shape}")

    tissue_coords = [] if tissue_coords is None else list(tissue_coords)
    vessel_curve = extract_voxel_tic(X, vessel_coord, normalize=normalize)

    if frame_time_sec is None:
        x = np.arange(X.shape[0])
        xlabel = "Frame"
    else:
        x = np.arange(X.shape[0]) * float(frame_time_sec)
        xlabel = "Time (s)"

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        vessel_curve,
        linewidth=2.5,
        color="crimson",
        label=f"Top vessel voxel {tuple(map(int, vessel_coord))}",
    )

    for i, coord in enumerate(tissue_coords, start=1):
        tissue_curve = extract_voxel_tic(X, coord, normalize=normalize)
        plt.plot(
            x,
            tissue_curve,
            linewidth=1.8,
            linestyle="--",
            label=f"Tissue {i} {tuple(map(int, coord))}",
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized intensity" if normalize else "Intensity")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def show_selected_voxels(
    background_img,
    vessel_coord,
    tissue_coords=None,
    vascular_mask=None,
    tissue_mask=None,
    title="Selected voxels",
):
    """
    Show the selected vessel/tissue voxels on top of a background image.
    """
    tissue_coords = [] if tissue_coords is None else list(tissue_coords)
    bg = np.asarray(background_img)

    plt.figure(figsize=(6, 6))
    plt.imshow(bg, cmap="gray")

    if vascular_mask is not None:
        plt.contour(np.asarray(vascular_mask).astype(bool), levels=[0.5], colors=["r"], linewidths=1.0)
    if tissue_mask is not None:
        plt.contour(np.asarray(tissue_mask).astype(bool), levels=[0.5], colors=["lime"], linewidths=0.8)

    vr, vc = map(int, vessel_coord)
    plt.scatter([vc], [vr], c="crimson", s=70, marker="o", label="Top vessel voxel")

    if tissue_coords:
        tc = np.asarray([(int(r), int(c)) for r, c in tissue_coords])
        plt.scatter(tc[:, 1], tc[:, 0], c="cyan", s=55, marker="x", label="Tissue voxels")

    plt.title(title)
    plt.axis("off")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()



