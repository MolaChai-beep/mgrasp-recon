"""Configuration and result types for MGRASP reconstruction workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LowResReconConfig:
    img_shape: tuple[int, int] = (96, 96)
    ns_low: int = 96
    method: str = "adjoint"
    cg_lamda: float = 1e-3
    max_cg_iter: int = 12
    cg_tol: float = 1e-6
    normalize: bool = True
    return_complex: bool = False
    rescale_traj: bool = False
    use_ramp_filter: bool = False
    ramp_filter_normalize: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class CoilCalibrationConfig:
    thresh: float = 0.01
    use_dcf: bool = True
    mask_floor: float = 0.05
    calib_width: int = 24
    crop: float = 0.95
    use_espirit: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class ReconstructionConfig:
    nbasis: int = 5
    cbasis: bool = False
    add_constant: bool = True
    lamda: float = 1e-3
    regu: str = "TV"
    regu_axes: tuple[int, ...] = (-2, -1)
    max_iter: int = 10
    solver: str = "ADMM"
    use_dcf: bool = True
    show_pbar: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class SliceReconstructionConfig:
    recon: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    coil: CoilCalibrationConfig = field(default_factory=CoilCalibrationConfig)
    save_h5: bool = False
    out_path: str | Path | None = None
    hop_id: str | None = None
    slice_idx: int | None = None
    coil_device: int = -1
    recon_device: object | None = None


@dataclass(frozen=True)
class SegmentationConfig:
    frame_time_sec: float = 3.8
    n_baseline: int = 5
    early_duration_sec: float = 120
    brain_percentile: float = 60
    core_erosion_iters: int = 12
    roi_erosion_iters: int = 8
    enhancement_percentile: float = 75
    cleanup_min_size: int = 10
    cleanup_open_iters: int = 1
    cleanup_close_iters: int = 1
    smooth_sigma: float = 2.0
    vessel_percentile: float = 90
    opening_iters: int = 1
    closing_iters: int = 1
    erosion_iters: int = 0
    verbose: bool = False


@dataclass(frozen=True)
class Recon2DResult:
    coeff_maps: np.ndarray
    img_dyn: np.ndarray
    img_dyn_abs: np.ndarray
    basisoption: np.ndarray


@dataclass(frozen=True)
class SliceReconResult:
    coeff_maps: np.ndarray
    img_dyn_cplx: np.ndarray
    img_dyn_abs: np.ndarray
    basisoption: np.ndarray
    mps: np.ndarray
    ksp: np.ndarray


@dataclass(frozen=True)
class SegmentationResult:
    mean_img: np.ndarray
    peak_img: np.ndarray
    std_img: np.ndarray | None
    baseline_mean: np.ndarray | None
    norm_early_enh: np.ndarray | None
    brain_mask: np.ndarray
    brain_core_mask: np.ndarray | None
    brain_ring_mask: np.ndarray | None
    pca_roi_mask: np.ndarray | None
    vascular_mask: np.ndarray
    tissue_mask: np.ndarray
    brain_mask_threshold: float
    enhancement_threshold: float
    baseline_idx: np.ndarray | None = None
    early_idx: np.ndarray | None = None
    mean_img_s: np.ndarray | None = None
    enhancement_score: np.ndarray | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "mean_img": self.mean_img,
            "peak_img": self.peak_img,
            "std_img": self.std_img,
            "baseline_mean": self.baseline_mean,
            "norm_early_enh": self.norm_early_enh,
            "brain_mask": self.brain_mask,
            "brain_core_mask": self.brain_core_mask,
            "brain_ring_mask": self.brain_ring_mask,
            "pca_roi_mask": self.pca_roi_mask,
            "vascular_mask": self.vascular_mask,
            "tissue_mask": self.tissue_mask,
            "brain_mask_threshold": self.brain_mask_threshold,
            "enhancement_threshold": self.enhancement_threshold,
            "baseline_idx": self.baseline_idx,
            "early_idx": self.early_idx,
            "mean_img_s": self.mean_img_s,
            "enhancement_score": self.enhancement_score,
            "brain_threshold": self.brain_mask_threshold,
            "vessel_threshold": self.enhancement_threshold,
        }
