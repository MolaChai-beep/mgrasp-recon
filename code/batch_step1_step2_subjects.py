from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torch


def _add_repo_src_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if src_root.exists() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return repo_root


REPO_ROOT = _add_repo_src_to_path()

from mgrasp_recon import (  # noqa: E402
    BasisPreparationConfig,
    BasisPreparationWorkflow,
    CoilCalibrationConfig,
    LowResReconConfig,
    ReconstructionConfig,
    SegmentationConfig,
    SliceReconstructionConfig,
    SliceReconstructionWorkflow,
    TicAnalyzer,
)
from mgrasp_recon.recon_utils import get_traj, infer_kspace_dims, list_slice_files, read_csv_config  # noqa: E402
from mgrasp_recon.visualization import plot_segmentation_summary  # noqa: E402


LOWRES_SHAPE = (256, 256)
LOWRES_TAG = "lowres_256x256"
STEP1_NAME = "step1_basis_test_multi_low_res"
STEP2_NAME = "step2_test_multi_basis_recon"
STEP1_SLICE_IDX = 47
LAMBDA_VALUE = 1e-3
N_BASIS = 5
FRAME_TIME_SEC = 3.8
VOXEL_LIST = [(180, 120), (200, 135)]
GRAPH_ROOT = Path("/mnt/gdrive/DCE_MRI/outputs/graph")
TICKER = TicAnalyzer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run step1 basis estimation on slice 47, then step2 reconstruction on all slices for every subject csv.",
    )
    parser.add_argument("--csv-dir", type=Path, required=True, help="Directory containing <subject_id>_config.csv files.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/gdrive/DCE_MRI/RAVE_files/h5slices_wt_acqT"),
        help="Subject root directory. Each hop is expected at <data-root>/<subject_id>/<hop_id>/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/gdrive/DCE_MRI/outputs/outputs"),
        help="Output root. Results are written under <output-root>/<step_name>/<subject_id>/<hop_id>/lowres_256x256/",
    )
    parser.add_argument(
        "--coil-thresh",
        type=float,
        default=0.02,
        help="Coil threshold reused from the existing step1/step2 scripts.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional list of subject IDs to run. Defaults to all subjects in --csv-dir.",
    )
    return parser.parse_args()


def subject_id_from_csv(csv_path: Path) -> str:
    suffix = "_config.csv"
    if not csv_path.name.endswith(suffix):
        raise ValueError(f"CSV filename must match <subject_id>{suffix}: {csv_path.name}")
    return csv_path.name[: -len(suffix)]


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda out of memory" in text


def _save_figure(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def filter_csv_paths(csv_paths: list[Path], subjects: list[str] | None) -> list[Path]:
    if not subjects:
        return csv_paths
    selected = set(subjects)
    return [csv_path for csv_path in csv_paths if subject_id_from_csv(csv_path) in selected]


def build_traj(spokes_per_frame: int, n_time: int, base_res: int) -> np.ndarray:
    traj = np.asarray(
        get_traj(
            N_spokes=spokes_per_frame,
            N_time=n_time,
            base_res=base_res,
            gind=1,
        ),
        dtype=np.float32,
    )
    if traj.ndim == 3:
        traj = traj[:, None, :, :]
    expected_shape = (n_time, spokes_per_frame, base_res * 2, 2)
    if traj.shape != expected_shape:
        raise ValueError(f"traj shape {traj.shape} does not match expected {expected_shape}")
    return traj


def save_step1_graphs(result, graph_dir: Path) -> None:
    img_init = np.asarray(result.img_lowres)
    seg = result.segmentation

    fig_preview, axes_preview = plt.subplots(1, 3, figsize=(12, 4))
    for i, frame_idx in enumerate([0, img_init.shape[0] // 2, img_init.shape[0] - 1]):
        axes_preview[i].imshow(np.abs(img_init[frame_idx]), cmap="gray")
        axes_preview[i].set_title(f"Frame {frame_idx}")
        axes_preview[i].axis("off")
    _save_figure(fig_preview, graph_dir / "img_lowres_preview.png")

    fig_seg, _ = plot_segmentation_summary(
        mean_img=seg.mean_img,
        baseline_mean=seg.baseline_mean,
        norm_early_enh=seg.norm_early_enh,
        std_img=seg.std_img,
        brain_mask=seg.brain_mask,
        brain_core_mask=seg.brain_core_mask,
        brain_ring_mask=seg.brain_ring_mask,
        pca_roi_mask=seg.pca_roi_mask,
        vascular_mask=seg.vascular_mask,
        tissue_mask=seg.tissue_mask,
        enhancement_threshold=seg.enhancement_threshold,
    )
    _save_figure(fig_seg, graph_dir / "segmentation_summary.png")


def save_step2_tic_graphs(recon_result, graph_dir: Path) -> None:
    mean_img = recon_result.img_dyn_abs.mean(axis=0)
    for row, col in VOXEL_LIST:
        fig_voxel, ax_voxel = plt.subplots(figsize=(6, 6))
        ax_voxel.imshow(mean_img, cmap="gray")
        ax_voxel.scatter([col], [row], c="red", s=80)
        ax_voxel.text(col + 2, row - 2, f"({row}, {col})", color="yellow")
        ax_voxel.set_title(f"slice 47 voxel ({row}, {col})")
        ax_voxel.axis("off")
        _save_figure(fig_voxel, graph_dir / f"voxel_location_r{row}_c{col}.png")

        curve = TICKER.extract_voxel_tic(recon_result.img_dyn_abs, (row, col), normalize=False)
        time_axis = np.arange(len(curve)) * FRAME_TIME_SEC

        fig_tic, ax_tic = plt.subplots(figsize=(7, 4))
        ax_tic.plot(time_axis, curve, linewidth=2)
        ax_tic.set_xlabel("Time (s)")
        ax_tic.set_ylabel("Intensity")
        ax_tic.set_title(f"slice 47 TIC at voxel ({row}, {col})")
        ax_tic.grid(alpha=0.3)
        _save_figure(fig_tic, graph_dir / f"tic_r{row}_c{col}.png")


def make_step1_workflow(spokes_per_frame: int, coil_thresh: float) -> BasisPreparationWorkflow:
    return BasisPreparationWorkflow(
        BasisPreparationConfig(
            spokes_per_frame=spokes_per_frame,
            lowres=LowResReconConfig(
                img_shape=LOWRES_SHAPE,
                ns_low=256,
                method="adjoint",
                normalize=False,
                return_complex=False,
                use_ramp_filter=True,
                verbose=True,
            ),
            coil=CoilCalibrationConfig(thresh=coil_thresh, verbose=True),
            segmentation=SegmentationConfig(
                frame_time_sec=3.8,
                n_baseline=5,
                early_duration_sec=120,
                brain_percentile=60,
                core_erosion_iters=12,
                roi_erosion_iters=8,
                enhancement_percentile=75,
                cleanup_min_size=10,
            ),
            nbasis=5,
            remove_mean=True,
            use_segmented_basis=True,
        )
    )


def build_basis_for_hop(
    subject_id: str,
    hop_id: str,
    slice_files: list[str],
    traj,
    spokes_per_frame: int,
    coil_thresh: float,
    step1_dir: Path,
    graph_dir: Path,
) -> Path:
    workflow = make_step1_workflow(spokes_per_frame=spokes_per_frame, coil_thresh=coil_thresh)
    result = workflow.run(slice_files=slice_files, slice_idx=STEP1_SLICE_IDX, traj=traj)
    if result.vascular_basis is None or result.tissue_basis is None:
        raise ValueError("Segmented basis output is required but vascular/tissue basis is missing.")
    save_step1_graphs(result, graph_dir)
    basis = np.concatenate([result.vascular_basis[:, :3], result.tissue_basis[:, :3]], axis=1)
    basis_path = step1_dir / "fbasis.h5"
    workflow.save_basis(basis, basis_path)
    print(f"  step1 saved basis: {basis_path}")
    print(f"  step1 subject={subject_id} hop={hop_id} slice={STEP1_SLICE_IDX}")
    return basis_path


def reconstruct_all_slices_for_hop(
    subject_id: str,
    hop_id: str,
    slice_files: list[str],
    traj,
    basis_path: Path,
    spokes_per_frame: int,
    coil_thresh: float,
    recon_device,
    step2_dir: Path,
    graph_dir: Path,
) -> list[tuple[str, str, str, str]]:
    failures: list[tuple[str, str, str, str]] = []

    for slice_idx, slice_file in enumerate(slice_files):
        out_path = step2_dir / f"{hop_id}_slice_{slice_idx:03d}.h5"
        workflow = SliceReconstructionWorkflow(
            SliceReconstructionConfig(
                recon=ReconstructionConfig(
                    nbasis=N_BASIS,
                    cbasis=False,
                    add_constant=True,
                    lamda=LAMBDA_VALUE,
                    regu="TV",
                    regu_axes=(-2, -1),
                    max_iter=10,
                    solver="ADMM",
                    use_dcf=True,
                    show_pbar=False,
                    verbose=True,
                ),
                coil=CoilCalibrationConfig(thresh=coil_thresh, verbose=True),
                save_h5=True,
                out_path=out_path,
                hop_id=hop_id,
                slice_idx=slice_idx,
                coil_device=-1,
                recon_device=recon_device,
            )
        )

        try:
            recon_result = workflow.reconstruct_slice(
                slice_file=slice_file,
                traj=traj,
                fbasis_path=basis_path,
                spokes_per_frame=spokes_per_frame,
                slice_idx=slice_idx,
            )
            if slice_idx == STEP1_SLICE_IDX:
                save_step2_tic_graphs(recon_result, graph_dir)
            print(f"    step2 saved slice {slice_idx:03d}: {out_path}")
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc):
                raise
            failures.append((subject_id, hop_id, f"slice_{slice_idx:03d}", str(exc)))
            print(f"    FAILED slice {slice_idx:03d}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return failures


def run_subject_csv(
    csv_path: Path,
    data_root: Path,
    output_root: Path,
    coil_thresh: float,
    recon_device,
) -> list[tuple[str, str, str, str]]:
    subject_id = subject_id_from_csv(csv_path)
    subject_root = data_root / subject_id
    configs = read_csv_config(csv_path)
    failures: list[tuple[str, str, str, str]] = []

    print()
    print("=" * 80)
    print(f"subject: {subject_id}")
    print(f"csv: {csv_path}")
    print("=" * 80)

    for config in configs:
        hop_id = config["hop_id"]
        spokes_per_frame = config["spokes_per_frame"]
        hop_dir = subject_root / hop_id
        step1_dir = output_root / STEP1_NAME / subject_id / hop_id / LOWRES_TAG
        step2_dir = output_root / STEP2_NAME / subject_id / hop_id / LOWRES_TAG
        graph_dir = GRAPH_ROOT / subject_id / hop_id / LOWRES_TAG

        print()
        print(f"> hop: {hop_id}")
        print(f"  subject root: {subject_root}")
        print(f"  hop dir: {hop_dir}")
        print(f"  spokes_per_frame: {spokes_per_frame}")

        try:
            if not hop_dir.exists():
                raise FileNotFoundError(f"Directory not found: {hop_dir}")

            slice_files = list_slice_files(str(hop_dir))
            if STEP1_SLICE_IDX >= len(slice_files):
                raise IndexError(
                    f"STEP1_SLICE_IDX={STEP1_SLICE_IDX} outside available slice range 0..{len(slice_files) - 1}"
                )

            n_coils, n_samples, n_spokes, n_slices = infer_kspace_dims(slice_files[0])
            n_time = n_spokes // spokes_per_frame
            if n_time <= 0:
                raise ValueError(
                    f"n_time={n_time}. Check spokes_per_frame={spokes_per_frame} vs n_spokes={n_spokes}"
                )

            traj = build_traj(
                spokes_per_frame=spokes_per_frame,
                n_time=n_time,
                base_res=n_samples // 2,
            )
            print(
                "  inferred dims: "
                f"slices={n_slices}, spokes={n_spokes}, samples={n_samples}, coils={n_coils}, n_time={n_time}"
            )

            step1_dir.mkdir(parents=True, exist_ok=True)
            step2_dir.mkdir(parents=True, exist_ok=True)
            graph_dir.mkdir(parents=True, exist_ok=True)

            basis_path = build_basis_for_hop(
                subject_id=subject_id,
                hop_id=hop_id,
                slice_files=slice_files,
                traj=traj,
                spokes_per_frame=spokes_per_frame,
                coil_thresh=coil_thresh,
                step1_dir=step1_dir,
                graph_dir=graph_dir,
            )
            failures.extend(
                reconstruct_all_slices_for_hop(
                    subject_id=subject_id,
                    hop_id=hop_id,
                    slice_files=slice_files,
                    traj=traj,
                    basis_path=basis_path,
                    spokes_per_frame=spokes_per_frame,
                    coil_thresh=coil_thresh,
                    recon_device=recon_device,
                    step2_dir=step2_dir,
                    graph_dir=graph_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc):
                raise RuntimeError(f"OOM at subject={subject_id} hop={hop_id}") from exc
            failures.append((subject_id, hop_id, "hop", str(exc)))
            print(f"  FAILED hop {hop_id}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return failures


def main() -> int:
    args = parse_args()
    csv_paths = sorted(args.csv_dir.glob("*_config.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_config.csv files found in {args.csv_dir}")
    csv_paths = filter_csv_paths(csv_paths, args.subjects)
    if not csv_paths:
        raise FileNotFoundError("No matching subject config CSVs found for --subjects.")

    cuda_available = torch.cuda.is_available()
    recon_device = sp.Device(0 if cuda_available else -1)
    failures: list[tuple[str, str, str, str]] = []

    print(f"> torch.cuda.is_available() = {cuda_available}")
    print(f"> torch.cuda.device_count() = {torch.cuda.device_count()}")
    if cuda_available:
        print(f"> torch.cuda.current_device() = {torch.cuda.current_device()}")
        print(f"> torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
    print(f"> device {recon_device}")
    print(f"> csv count {len(csv_paths)}")
    print(f"> data root {args.data_root}")
    print(f"> output root {args.output_root}")
    print(f"> graph root {GRAPH_ROOT}")

    try:
        for csv_path in csv_paths:
            failures.extend(
                run_subject_csv(
                    csv_path=csv_path,
                    data_root=args.data_root,
                    output_root=args.output_root,
                    coil_thresh=args.coil_thresh,
                    recon_device=recon_device,
                )
            )
    except RuntimeError as exc:
        if "OOM at subject=" in str(exc):
            print()
            print(f"STOPPED: {exc}")
            return 2
        raise

    print()
    print("=" * 80)
    if failures:
        print(f"finished with {len(failures)} failures")
        for subject_id, hop_id, stage, error in failures:
            print(f"  {subject_id} | {hop_id} | {stage} | {error}")
        return 1

    print("finished successfully with no failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
