"""Time-intensity curve helpers."""

from __future__ import annotations

import numpy as np

from .visualization import plot_time_intensity_curves as _plot_time_intensity_curves
from .visualization import show_selected_voxels as _show_selected_voxels


class TicAnalyzer:
    """Small analysis helper for extracting and plotting voxel TICs."""

    def extract_voxel_tic(self, img_dyn, coord, normalize=False):
        x = np.abs(np.asarray(img_dyn))
        if x.ndim != 3:
            raise ValueError(f"img_dyn must have shape (T, H, W), got {x.shape}")

        row, col = map(int, coord)
        if not (0 <= row < x.shape[1] and 0 <= col < x.shape[2]):
            raise IndexError(f"coord {coord} is outside image bounds {x.shape[1:]}")

        curve = x[:, row, col].astype(np.float32, copy=False)
        if normalize:
            baseline = float(np.mean(curve[: min(5, len(curve))]))
            curve = curve / max(baseline, 1e-6)
        return curve

    def plot_time_intensity_curves(
        self,
        img_dyn,
        vessel_coord,
        tissue_coords=None,
        frame_time_sec=None,
        normalize=False,
        title="Time-intensity curves",
    ):
        x = np.abs(np.asarray(img_dyn))
        if x.ndim != 3:
            raise ValueError(f"img_dyn must have shape (T, H, W), got {x.shape}")

        tissue_coords = [] if tissue_coords is None else list(tissue_coords)
        vessel_curve = self.extract_voxel_tic(x, vessel_coord, normalize=normalize)
        tissue_curves = [self.extract_voxel_tic(x, coord, normalize=normalize) for coord in tissue_coords]
        return _plot_time_intensity_curves(
            vessel_curve=vessel_curve,
            vessel_coord=tuple(map(int, vessel_coord)),
            tissue_curves=tissue_curves,
            tissue_coords=[tuple(map(int, coord)) for coord in tissue_coords],
            frame_time_sec=frame_time_sec,
            normalize=normalize,
            title=title,
        )

    def show_selected_voxels(
        self,
        background_img,
        vessel_coord,
        tissue_coords=None,
        vascular_mask=None,
        tissue_mask=None,
        title="Selected voxels",
    ):
        tissue_coords = [] if tissue_coords is None else list(tissue_coords)
        return _show_selected_voxels(
            background_img=np.asarray(background_img),
            vessel_coord=tuple(map(int, vessel_coord)),
            tissue_coords=[tuple(map(int, coord)) for coord in tissue_coords],
            vascular_mask=vascular_mask,
            tissue_mask=tissue_mask,
            title=title,
        )
