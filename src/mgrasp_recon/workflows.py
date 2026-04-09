"""Workflow and service classes for the MGRASP reconstruction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    CoilCalibrationConfig,
    LowResReconConfig,
    ReconstructionConfig,
    SegmentationConfig,
    SegmentationResult,
    SliceReconResult,
    SliceReconstructionConfig,
)
from .interframe_recon import _radial_lowres_pca_recon_2d
from .recon_2d import _run_subspace_recon_2d, _run_subspace_recon_for_slice
from .recon_utils import (
    _estimate_coil_maps,
    _estimate_segmented_pca_bases,
    estimate_pca_basis,
    get_traj,
    list_slice_files,
    load_pca_basis_h5,
    load_slice_kspace_for_coil,
    ri_to_coil_spokes_samples,
    save_pca_basis_h5,
)
from .vascular_segmentation import _segment_dynamic_series, _segment_enhancement_series


@dataclass(frozen=True)
class BasisPreparationConfig:
    spokes_per_frame: int
    lowres: LowResReconConfig = field(default_factory=LowResReconConfig)
    coil: CoilCalibrationConfig = field(default_factory=CoilCalibrationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    nbasis: int = 5
    remove_mean: bool = True
    use_segmented_basis: bool = True
    basis_output_path: str | Path | None = None


@dataclass(frozen=True)
class PatientWorkflowConfig:
    basis: BasisPreparationConfig
    reconstruction: SliceReconstructionConfig = field(default_factory=SliceReconstructionConfig)


@dataclass(frozen=True)
class BasisPreparationResult:
    slice_file: str
    slice_idx: int
    ksp: np.ndarray
    mps: np.ndarray
    img_lowres: np.ndarray
    segmentation: SegmentationResult
    vascular_basis: np.ndarray | None
    tissue_basis: np.ndarray | None
    basis: np.ndarray
    basis_path: str | None = None


@dataclass(frozen=True)
class PatientReconResult:
    hop_id: str | None
    basis_result: BasisPreparationResult | None
    slice_results: list[SliceReconResult]
    failed_slices: list[tuple[str, str]]


class TrajectoryProvider:
    def get_slice_trajectory(self, ksp: np.ndarray, spokes_per_frame: int) -> np.ndarray:
        _, num_spokes, num_samples = np.asarray(ksp).shape
        if spokes_per_frame <= 0:
            raise ValueError("spokes_per_frame must be positive.")
        num_frames = num_spokes // spokes_per_frame
        base_res = num_samples // 2
        return np.asarray(get_traj(N_spokes=spokes_per_frame, N_time=num_frames, base_res=base_res, gind=1), dtype=np.float32)


class CoilMapEstimator:
    def __init__(self, config: CoilCalibrationConfig | None = None, device=-1):
        self.config = config or CoilCalibrationConfig()
        self.device = device

    def estimate(self, ksp: np.ndarray) -> np.ndarray:
        return _estimate_coil_maps(ksp, device=self.device, config=self.config)


class LowResReconstructor:
    def __init__(self, config: LowResReconConfig | None = None):
        self.config = config or LowResReconConfig()

    def reconstruct(self, ksp: np.ndarray, traj: np.ndarray, spokes_per_frame: int, mps: np.ndarray) -> np.ndarray:
        return _radial_lowres_pca_recon_2d(ksp, traj, spokes_per_frame, mps, self.config)


class SegmentationAnalyzer:
    def __init__(self, config: SegmentationConfig | None = None):
        self.config = config or SegmentationConfig()

    def segment_enhancement_series(self, img_dyn: np.ndarray) -> SegmentationResult:
        return _segment_enhancement_series(img_dyn, config=self.config)

    def segment_dynamic_series(self, img_dyn: np.ndarray) -> SegmentationResult:
        return _segment_dynamic_series(img_dyn, config=self.config)


class BasisEstimator:
    def __init__(self, nbasis: int = 5, remove_mean: bool = True):
        self.nbasis = nbasis
        self.remove_mean = remove_mean

    def estimate_global_basis(self, img_dyn: np.ndarray) -> np.ndarray:
        basis, _ = estimate_pca_basis(img_dyn, K=self.nbasis, remove_mean=self.remove_mean)
        return basis

    def estimate_segmented_basis(self, img_dyn: np.ndarray, vascular_mask: np.ndarray, tissue_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bases = _estimate_segmented_pca_bases(
            img_dyn,
            vascular_mask=vascular_mask,
            tissue_mask=tissue_mask,
            K=self.nbasis,
            remove_mean=self.remove_mean,
        )
        return bases["vascular_basis"], bases["tissue_basis"]

    def save_basis(self, path: str | Path, basis: np.ndarray) -> str:
        save_pca_basis_h5(path, basis, nbasis=self.nbasis)
        return str(path)

    def load_basis(self, path: str | Path, nbasis: int | None = None) -> np.ndarray:
        return load_pca_basis_h5(path, nbasis=nbasis or self.nbasis)


class BasisPreparationWorkflow:
    def __init__(
        self,
        config: BasisPreparationConfig,
        trajectory_provider: TrajectoryProvider | None = None,
        coil_estimator: CoilMapEstimator | None = None,
        lowres_reconstructor: LowResReconstructor | None = None,
        segmentation_analyzer: SegmentationAnalyzer | None = None,
        basis_estimator: BasisEstimator | None = None,
    ):
        self.config = config
        self.trajectory_provider = trajectory_provider or TrajectoryProvider()
        self.coil_estimator = coil_estimator or CoilMapEstimator(config.coil)
        self.lowres_reconstructor = lowres_reconstructor or LowResReconstructor(config.lowres)
        self.segmentation_analyzer = segmentation_analyzer or SegmentationAnalyzer(config.segmentation)
        self.basis_estimator = basis_estimator or BasisEstimator(config.nbasis, config.remove_mean)

    def select_reference_slice(self, slice_files: list[str]) -> tuple[int, str]:
        if not slice_files:
            raise ValueError("slice_files must not be empty.")
        slice_idx = len(slice_files) // 2
        return slice_idx, slice_files[slice_idx]

    def load_reference_slice(self, slice_file: str) -> np.ndarray:
        ksp_ri = load_slice_kspace_for_coil(slice_file, verbose=self.config.coil.verbose)
        return ri_to_coil_spokes_samples(ksp_ri)

    def estimate_coils(self, ksp: np.ndarray) -> np.ndarray:
        return self.coil_estimator.estimate(ksp)

    def reconstruct_lowres_series(self, ksp: np.ndarray, mps: np.ndarray, traj: np.ndarray | None = None) -> np.ndarray:
        traj = traj if traj is not None else self.trajectory_provider.get_slice_trajectory(ksp, self.config.spokes_per_frame)
        return self.lowres_reconstructor.reconstruct(ksp, traj, self.config.spokes_per_frame, mps)

    def segment_vascular_and_tissue(self, img_dyn: np.ndarray) -> SegmentationResult:
        return self.segmentation_analyzer.segment_enhancement_series(img_dyn)

    def estimate_segmented_basis(self, img_dyn: np.ndarray, segmentation: SegmentationResult) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        if self.config.use_segmented_basis:
            vascular_basis, tissue_basis = self.basis_estimator.estimate_segmented_basis(
                img_dyn,
                vascular_mask=segmentation.vascular_mask,
                tissue_mask=segmentation.tissue_mask,
            )
            return vascular_basis, tissue_basis, vascular_basis
        basis = self.basis_estimator.estimate_global_basis(img_dyn)
        return None, None, basis

    def save_basis(self, basis: np.ndarray, path: str | Path | None = None) -> str | None:
        target = path or self.config.basis_output_path
        if target is None:
            return None
        return self.basis_estimator.save_basis(target, basis)

    def run(self, slice_files: list[str], slice_idx: int | None = None, traj: np.ndarray | None = None, basis_output_path: str | Path | None = None) -> BasisPreparationResult:
        if slice_idx is None:
            slice_idx, slice_file = self.select_reference_slice(slice_files)
        else:
            slice_file = slice_files[slice_idx]

        ksp = self.load_reference_slice(slice_file)
        mps = self.estimate_coils(ksp)
        img_lowres = self.reconstruct_lowres_series(ksp, mps, traj=traj)
        segmentation = self.segment_vascular_and_tissue(img_lowres)
        vascular_basis, tissue_basis, basis = self.estimate_segmented_basis(img_lowres, segmentation)
        basis_path = self.save_basis(basis, basis_output_path)

        return BasisPreparationResult(
            slice_file=slice_file,
            slice_idx=slice_idx,
            ksp=ksp,
            mps=np.asarray(mps),
            img_lowres=img_lowres,
            segmentation=segmentation,
            vascular_basis=vascular_basis,
            tissue_basis=tissue_basis,
            basis=basis,
            basis_path=basis_path,
        )


class SliceReconstructionWorkflow:
    def __init__(self, config: SliceReconstructionConfig, trajectory_provider: TrajectoryProvider | None = None):
        self.config = config
        self.trajectory_provider = trajectory_provider or TrajectoryProvider()

    def reconstruct_slice(self, slice_file: str, traj: np.ndarray, fbasis_path: str | Path, spokes_per_frame: int, slice_idx: int | None = None) -> SliceReconResult:
        config = SliceReconstructionConfig(
            recon=self.config.recon,
            coil=self.config.coil,
            save_h5=self.config.save_h5,
            out_path=self.config.out_path,
            hop_id=self.config.hop_id,
            slice_idx=slice_idx,
            coil_device=self.config.coil_device,
            recon_device=self.config.recon_device,
        )
        return _run_subspace_recon_for_slice(slice_file=slice_file, traj=traj, fbasis_path=str(fbasis_path), spokes_per_frame=spokes_per_frame, config=config)

    def reconstruct_all_slices(self, slice_files: list[str], traj: np.ndarray, fbasis_path: str | Path, spokes_per_frame: int) -> tuple[list[SliceReconResult], list[tuple[str, str]]]:
        results: list[SliceReconResult] = []
        failures: list[tuple[str, str]] = []
        for idx, slice_file in enumerate(slice_files):
            try:
                results.append(self.reconstruct_slice(slice_file, traj, fbasis_path, spokes_per_frame, slice_idx=idx))
            except Exception as exc:  # noqa: BLE001
                failures.append((slice_file, str(exc)))
        return results, failures

    def run_slice(self, ksp: np.ndarray, traj: np.ndarray, mps: np.ndarray, fbasis_path: str | Path, spokes_per_frame: int):
        return _run_subspace_recon_2d(
            ksp=ksp,
            traj=traj,
            mps=mps,
            fbasis_path=str(fbasis_path),
            spokes_per_frame=spokes_per_frame,
            config=self.config.recon,
            device=self.config.recon_device,
        )

    def run_patient(self, slice_files: list[str], traj: np.ndarray, fbasis_path: str | Path, spokes_per_frame: int) -> PatientReconResult:
        results, failures = self.reconstruct_all_slices(slice_files, traj, fbasis_path, spokes_per_frame)
        return PatientReconResult(hop_id=self.config.hop_id, basis_result=None, slice_results=results, failed_slices=failures)


class PatientReconstructionWorkflow:
    def __init__(
        self,
        hop_id: str | None = None,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        basis_config: BasisPreparationConfig | None = None,
        reconstruction_config: SliceReconstructionConfig | None = None,
        trajectory_provider: TrajectoryProvider | None = None,
    ):
        if basis_config is None:
            raise ValueError("basis_config must be provided because spokes_per_frame is required.")

        self.hop_id = hop_id
        self.input_dir = Path(input_dir) if input_dir is not None else None
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.trajectory_provider = trajectory_provider or TrajectoryProvider()
        self.basis_workflow = BasisPreparationWorkflow(basis_config, trajectory_provider=self.trajectory_provider)

        recon_config = reconstruction_config or SliceReconstructionConfig(hop_id=hop_id)
        if recon_config.hop_id is None and hop_id is not None:
            recon_config = SliceReconstructionConfig(
                recon=recon_config.recon,
                coil=recon_config.coil,
                save_h5=recon_config.save_h5,
                out_path=recon_config.out_path,
                hop_id=hop_id,
                slice_idx=recon_config.slice_idx,
                coil_device=recon_config.coil_device,
                recon_device=recon_config.recon_device,
            )
        self.slice_workflow = SliceReconstructionWorkflow(recon_config, trajectory_provider=self.trajectory_provider)

    def _get_slice_files(self, hop_dir: str | Path | None = None) -> list[str]:
        target_dir = Path(hop_dir) if hop_dir is not None else self.input_dir
        if target_dir is None:
            raise ValueError("input_dir or hop_dir must be provided.")
        return list_slice_files(str(target_dir))

    def run_step1_basis_from_middle_slice(self, hop_dir: str | Path | None = None, traj: np.ndarray | None = None, basis_output_path: str | Path | None = None) -> BasisPreparationResult:
        slice_files = self._get_slice_files(hop_dir)
        return self.basis_workflow.run(slice_files=slice_files, traj=traj, basis_output_path=basis_output_path)

    def run_step2_reconstruct_slice(self, slice_file: str, traj: np.ndarray, basis_path: str | Path, spokes_per_frame: int | None = None, slice_idx: int | None = None) -> SliceReconResult:
        spf = spokes_per_frame or self.basis_workflow.config.spokes_per_frame
        return self.slice_workflow.reconstruct_slice(slice_file, traj, basis_path, spf, slice_idx=slice_idx)

    def run_single_slice(self, slice_file: str, traj: np.ndarray, basis_path: str | Path, spokes_per_frame: int | None = None, slice_idx: int | None = None) -> SliceReconResult:
        return self.run_step2_reconstruct_slice(slice_file, traj, basis_path, spokes_per_frame, slice_idx)

    def run_all_slices(self, hop_dir: str | Path | None, traj: np.ndarray, basis_path: str | Path, spokes_per_frame: int | None = None) -> PatientReconResult:
        slice_files = self._get_slice_files(hop_dir)
        spf = spokes_per_frame or self.basis_workflow.config.spokes_per_frame
        results, failures = self.slice_workflow.reconstruct_all_slices(slice_files, traj, basis_path, spf)
        return PatientReconResult(hop_id=self.hop_id, basis_result=None, slice_results=results, failed_slices=failures)

    def run_full_pipeline(self, hop_dir: str | Path | None = None, step1_traj: np.ndarray | None = None, step2_traj: np.ndarray | None = None, basis_output_path: str | Path | None = None) -> PatientReconResult:
        basis_result = self.run_step1_basis_from_middle_slice(hop_dir, step1_traj, basis_output_path)
        slice_files = self._get_slice_files(hop_dir)
        if step2_traj is None:
            step2_traj = self.trajectory_provider.get_slice_trajectory(basis_result.ksp, self.basis_workflow.config.spokes_per_frame)
        basis_path = basis_result.basis_path or basis_output_path
        if basis_path is None:
            raise ValueError("basis_output_path must be provided when running the full pipeline.")
        results, failures = self.slice_workflow.reconstruct_all_slices(slice_files, step2_traj, basis_path, self.basis_workflow.config.spokes_per_frame)
        return PatientReconResult(hop_id=self.hop_id, basis_result=basis_result, slice_results=results, failed_slices=failures)
