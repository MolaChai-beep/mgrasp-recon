from __future__ import annotations

import argparse
import contextlib
import gc
import io
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
    CoilMapEstimator,
    LowResReconConfig,
    ReconstructionConfig,
    SegmentationConfig,
    SliceReconstructionConfig,
    SliceReconstructionWorkflow,
    TicAnalyzer,
)
from mgrasp_recon.recon_utils import (  # noqa: E402
    infer_kspace_dims,
    list_slice_files,
    load_slice_kspace_for_coil,
    read_csv_config,
    ri_to_coil_spokes_samples,
    save_slice_h5,
)
from mgrasp_recon.visualization import plot_segmentation_summary  # noqa: E402


LOWRES_SHAPE = (256, 256)
LOWRES_TAG = "lowres_256x256"
STEP1_NAME = "step1_basis_combined_low_res"
STEP2_NAME = "step2_combined_basis_recon"
FIXED_COMBINED_SPF = 21
MIN_NON_DCE_FRAMES = 6
PREFERRED_NON_DCE_FRAMES = 8
MAX_DROPPED_RATIO = 0.15
STEP1_SLICE_IDX = 47
LAMBDA_VALUE = 1e-3
N_BASIS = 5
FRAME_TIME_SEC = 3.8
VOXEL_LIST = [(180, 120), (200, 135)]
COIL_DEVICE = -1
TICKER = TicAnalyzer()


@dataclass(frozen=True)
class SeriesInfo:
    hop_id: str
    hop_dir: Path
    slice_files: list[str]
    original_spokes_per_frame: int
    images_per_slab: int
    csv_n_par: int
    csv_n_eco: int
    n_coils: int
    n_samples: int
    n_spokes: int


@dataclass(frozen=True)
class RebinStats:
    hop_id: str
    original_spokes_per_frame: int
    target_spokes_per_frame: int
    n_spokes: int
    num_frames: int
    used_spokes: int
    dropped_spokes: int
    start_spoke: int

    @property
    def dropped_ratio(self) -> float:
        return 0.0 if self.n_spokes == 0 else self.dropped_spokes / self.n_spokes


def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
        help="Output root. Results are written under <output-root>/<step_name>/<subject_id>/<combined_name_from_csv>/lowres_256x256/",
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
    parser.add_argument(
        "--step1-hop-id",
        type=str,
        default="DCE",
        help="Sequence/hop to use for step1 lowres reconstruction, segmentation, and basis generation.",
    )
    return parser


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


def format_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s"


def filter_csv_paths(csv_paths: list[Path], subjects: list[str] | None) -> list[Path]:
    if not subjects:
        return csv_paths
    selected = set(subjects)
    return [csv_path for csv_path in csv_paths if subject_id_from_csv(csv_path) in selected]


def load_csv_paths(csv_dir: Path, subjects: list[str] | None) -> list[Path]:
    csv_paths = sorted(csv_dir.glob("*_config.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_config.csv files found in {csv_dir}")
    csv_paths = filter_csv_paths(csv_paths, subjects)
    if not csv_paths:
        raise FileNotFoundError("No matching subject config CSVs found for --subjects.")
    return csv_paths


def build_traj(spokes_per_frame: int, n_time: int, base_res: int) -> np.ndarray:
    return build_traj_with_offset(
        spokes_per_frame=spokes_per_frame,
        n_time=n_time,
        base_res=base_res,
        start_spoke=0,
    )


def build_traj_with_offset(spokes_per_frame: int, n_time: int, base_res: int, start_spoke: int) -> np.ndarray:
    n_tot_spokes = spokes_per_frame * n_time
    n_samples = base_res * 2
    base_lin = np.arange(n_samples, dtype=np.float32).reshape(1, -1) - (n_samples - 1) / 2
    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / tau
    base_rot = (start_spoke + np.arange(n_tot_spokes, dtype=np.float32)).reshape(-1, 1) * base_rad

    traj = np.zeros((n_tot_spokes, n_samples, 2), dtype=np.float32)
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin
    traj /= 2
    traj = traj.reshape(n_time, spokes_per_frame, n_samples, 2)
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
                frame_time_sec=FRAME_TIME_SEC,
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


def make_step2_workflow(
    out_path: Path,
    slice_idx: int,
    combined_hop_id: str,
    coil_thresh: float,
    recon_device,
) -> SliceReconstructionWorkflow:
    return SliceReconstructionWorkflow(
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
            save_h5=False,
            out_path=out_path,
            hop_id=combined_hop_id,
            slice_idx=slice_idx,
            coil_device=COIL_DEVICE,
            recon_device=recon_device,
        )
    )


def compute_rebin_stats(hop_id: str, original_spf: int, n_spokes: int, target_spf: int) -> RebinStats:
    num_frames = n_spokes // target_spf
    used_spokes = num_frames * target_spf
    dropped_spokes = n_spokes - used_spokes
    start_spoke = dropped_spokes
    return RebinStats(
        hop_id=hop_id,
        original_spokes_per_frame=original_spf,
        target_spokes_per_frame=target_spf,
        n_spokes=n_spokes,
        num_frames=num_frames,
        used_spokes=used_spokes,
        dropped_spokes=dropped_spokes,
        start_spoke=start_spoke,
    )


def format_stats_line(stats: RebinStats) -> str:
    stop_spoke = stats.start_spoke + stats.used_spokes
    return (
        f"{stats.hop_id}: orig_spf={stats.original_spokes_per_frame} "
        f"target_spf={stats.target_spokes_per_frame} n_spokes={stats.n_spokes} "
        f"frames={stats.num_frames} used={stats.used_spokes} dropped={stats.dropped_spokes} "
        f"discard_first={stats.start_spoke} used_range=[{stats.start_spoke}:{stop_spoke}) "
        f"drop_ratio={stats.dropped_ratio:.3f}"
    )


def choose_combined_spokes_per_frame(series_infos: list[SeriesInfo]) -> tuple[int, list[RebinStats]]:
    stats = [
        compute_rebin_stats(
            hop_id=info.hop_id,
            original_spf=info.original_spokes_per_frame,
            n_spokes=info.n_spokes,
            target_spf=FIXED_COMBINED_SPF,
        )
        for info in series_infos
    ]
    if any(item.dropped_ratio > MAX_DROPPED_RATIO for item in stats):
        detail = "\n".join(
            f"  {info.hop_id}: csv_spf={info.original_spokes_per_frame} n_spokes={info.n_spokes}"
            for info in series_infos
        )
        raise ValueError(
            "Fixed combined_spokes_per_frame is invalid for this subject.\n"
            f"combined_spf={FIXED_COMBINED_SPF}, max_dropped_ratio={MAX_DROPPED_RATIO}\n{detail}"
        )
    non_dce_stats = [item for item in stats if item.hop_id != "DCE"]
    if any(item.num_frames < MIN_NON_DCE_FRAMES for item in non_dce_stats):
        detail = "\n".join(format_stats_line(item) for item in stats)
        raise ValueError(
            "Fixed combined_spokes_per_frame does not leave enough non-DCE frames.\n"
            f"combined_spf={FIXED_COMBINED_SPF}, min_non_dce_frames={MIN_NON_DCE_FRAMES}\n{detail}"
        )
    return FIXED_COMBINED_SPF, stats


def require_series_configs(configs: list[dict[str, int]]) -> list[dict[str, int]]:
    ordered = [config for config in configs if str(config["hop_id"]).strip()]
    if not ordered:
        raise ValueError("No series rows found in CSV.")
    if not any(config["hop_id"] == "DCE" for config in ordered):
        raise ValueError("CSV must contain one DCE row in Name column.")
    return ordered


def collect_subject_series_infos(configs: list[dict[str, int]], subject_root: Path) -> list[SeriesInfo]:
    ordered_configs = require_series_configs(configs)
    series_infos: list[SeriesInfo] = []
    expected_num_slices: int | None = None
    expected_geometry: tuple[int, int] | None = None

    for config in ordered_configs:
        hop_id = config["hop_id"]
        hop_dir = subject_root / hop_id
        if not hop_dir.exists():
            raise FileNotFoundError(f"Directory not found: {hop_dir}")

        slice_files = list_slice_files(str(hop_dir))
        dims = infer_kspace_dims(slice_files[0])
        h5_n_coils, h5_n_samples, h5_n_spokes = (int(dims[0]), int(dims[1]), int(dims[2]))
        csv_n_coils = int(config["n_coils"])
        csv_n_samples = int(config["n_points"])
        csv_n_spokes = int(config["n_spokes"])
        n_coils, n_samples, n_spokes = csv_n_coils, csv_n_samples, csv_n_spokes

        if expected_num_slices is None:
            expected_num_slices = int(config["images_per_slab"])
        if len(slice_files) != int(config["images_per_slab"]):
            raise ValueError(
                f"Slice count mismatch for {hop_id}: csv images_per_slab={config['images_per_slab']}, files={len(slice_files)}"
            )
        if len(slice_files) != expected_num_slices:
            raise ValueError(
                f"Slice count mismatch for {hop_id}: expected {expected_num_slices}, got {len(slice_files)}"
            )
        if (h5_n_coils, h5_n_samples, h5_n_spokes) != (csv_n_coils, csv_n_samples, csv_n_spokes):
            raise ValueError(
                f"CSV/H5 mismatch for {hop_id}: csv={(csv_n_coils, csv_n_samples, csv_n_spokes)} "
                f"h5={(h5_n_coils, h5_n_samples, h5_n_spokes)}"
            )

        geometry = (n_coils, n_samples)
        if expected_geometry is None:
            expected_geometry = geometry
        elif geometry != expected_geometry:
            raise ValueError(f"Geometry mismatch for {hop_id}: expected {expected_geometry}, got {geometry}")

        series_infos.append(
            SeriesInfo(
                hop_id=hop_id,
                hop_dir=hop_dir,
                slice_files=slice_files,
                original_spokes_per_frame=int(config["spokes_per_frame"]),
                images_per_slab=int(config["images_per_slab"]),
                csv_n_par=int(config["n_par"]),
                csv_n_eco=int(config["n_eco"]),
                n_coils=n_coils,
                n_samples=n_samples,
                n_spokes=n_spokes,
            )
        )

    return series_infos


def select_step1_series_info(series_infos: list[SeriesInfo], step1_hop_id: str = "DCE") -> SeriesInfo:
    for info in series_infos:
        if info.hop_id == step1_hop_id:
            return info
    available = ", ".join(info.hop_id for info in series_infos)
    raise ValueError(f"step1 hop_id {step1_hop_id!r} not found. Available: {available}")


def build_combined_hop_id(series_infos: list[SeriesInfo]) -> str:
    labels = []
    for info in series_infos:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", str(info.hop_id).strip()).strip("_")
        if not safe:
            raise ValueError(f"Invalid hop_id for combined output name: {info.hop_id!r}")
        labels.append(safe)
    return "combined_" + "_".join(labels)


def build_combined_traj(series_infos: list[SeriesInfo], stats_map: dict[str, RebinStats]) -> np.ndarray:
    traj_parts = [
        build_traj_with_offset(
            spokes_per_frame=stats_map[info.hop_id].target_spokes_per_frame,
            n_time=stats_map[info.hop_id].num_frames,
            base_res=info.n_samples // 2,
            start_spoke=stats_map[info.hop_id].start_spoke,
        )
        for info in series_infos
    ]
    return np.concatenate(traj_parts, axis=0)


def build_series_traj(info: SeriesInfo, stats: RebinStats) -> np.ndarray:
    return build_traj_with_offset(
        spokes_per_frame=stats.target_spokes_per_frame,
        n_time=stats.num_frames,
        base_res=info.n_samples // 2,
        start_spoke=stats.start_spoke,
    )


def load_combined_slice_kspace(series_infos: list[SeriesInfo], stats_map: dict[str, RebinStats], slice_idx: int) -> np.ndarray:
    ksp_parts: list[np.ndarray] = []
    expected_geometry: tuple[int, int] | None = None

    for info in series_infos:
        stats = stats_map[info.hop_id]
        ksp_ri = load_slice_kspace_for_coil(info.slice_files[slice_idx], verbose=False)
        ksp = ri_to_coil_spokes_samples(ksp_ri)
        geometry = (int(ksp.shape[0]), int(ksp.shape[2]))
        if expected_geometry is None:
            expected_geometry = geometry
        elif geometry != expected_geometry:
            raise ValueError(
                f"Combined slice geometry mismatch at slice {slice_idx} for {info.hop_id}: "
                f"expected {expected_geometry}, got {geometry}"
            )
        stop_spoke = stats.start_spoke + stats.used_spokes
        ksp_parts.append(np.asarray(ksp[:, stats.start_spoke:stop_spoke, :], dtype=np.complex64))

    return np.concatenate(ksp_parts, axis=1)


def load_series_slice_kspace(info: SeriesInfo, stats: RebinStats, slice_idx: int) -> np.ndarray:
    ksp_ri = load_slice_kspace_for_coil(info.slice_files[slice_idx], verbose=False)
    ksp = ri_to_coil_spokes_samples(ksp_ri)
    stop_spoke = stats.start_spoke + stats.used_spokes
    return np.asarray(ksp[:, stats.start_spoke:stop_spoke, :], dtype=np.complex64)


def collect_slice_debug_rows(series_infos: list[SeriesInfo], stats_map: dict[str, RebinStats], slice_idx: int) -> list[str]:
    rows: list[str] = []
    for info in series_infos:
        stats = stats_map[info.hop_id]
        dims = infer_kspace_dims(info.slice_files[slice_idx])
        rows.append(
            f"{info.hop_id}: h5_shape={tuple(int(dim) for dim in dims)} "
            f"csv_expected=(coils={info.n_coils}, samples={info.n_samples}, spokes={info.n_spokes}) "
            f"used_spokes={stats.used_spokes} target_spf={stats.target_spokes_per_frame} "
            f"file={info.slice_files[slice_idx]}"
        )
    return rows


def get_recon_device():
    return sp.Device(0 if torch.cuda.is_available() else -1)


def recon_mode_label(recon_device) -> str:
    return "GPU" if torch.cuda.is_available() and getattr(recon_device, "id", 0) != -1 else "CPU"


def print_device_summary(recon_device) -> None:
    cuda_available = torch.cuda.is_available()
    print(f"> torch.cuda.is_available() = {cuda_available}")
    print(f"> torch.cuda.device_count() = {torch.cuda.device_count()}")
    if cuda_available:
        print(f"> torch.cuda.current_device() = {torch.cuda.current_device()}")
        print(f"> torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
    print(f"> recon_device = {recon_device}")
    print(f"> recon mode = {recon_mode_label(recon_device)}")
    print(f"> coil_device = {COIL_DEVICE}")
    print("> coil mode = CPU")
    print("> step1 lowres path = CPU")


def format_gpu_mem_stats(tag: str) -> str:
    if not torch.cuda.is_available():
        return f"{tag}: cuda_unavailable"

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()

    mib = 1024 ** 2
    return (
        f"{tag}: "
        f"alloc={allocated / mib:.1f} MiB "
        f"reserved={reserved / mib:.1f} MiB "
        f"max_alloc={max_allocated / mib:.1f} MiB "
        f"max_reserved={max_reserved / mib:.1f} MiB"
    )


def cleanup_slice_recon_state() -> None:
    try:
        import cupy as cp  # local import; optional at runtime

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


def get_step1_dir(output_root: Path, subject_id: str, combined_hop_id: str) -> Path:
    return output_root / STEP1_NAME / subject_id / combined_hop_id / LOWRES_TAG


def get_step2_dir(output_root: Path, subject_id: str, combined_hop_id: str) -> Path:
    return output_root / STEP2_NAME / subject_id / combined_hop_id / LOWRES_TAG


def get_step1_graph_dir(output_root: Path, subject_id: str, combined_hop_id: str) -> Path:
    return get_step1_dir(output_root, subject_id, combined_hop_id) / "graphs"


def get_step2_graph_dir(output_root: Path, subject_id: str, combined_hop_id: str) -> Path:
    return get_step2_dir(output_root, subject_id, combined_hop_id) / "graphs"


def get_basis_path(output_root: Path, subject_id: str, combined_hop_id: str) -> Path:
    return get_step1_dir(output_root, subject_id, combined_hop_id) / "fbasis.h5"


def require_basis_path(output_root: Path, subject_id: str, combined_hop_id: str) -> Path:
    basis_path = get_basis_path(output_root, subject_id, combined_hop_id)
    if not basis_path.exists():
        raise FileNotFoundError(f"Step1 basis not found for subject {subject_id}: {basis_path}")
    return basis_path


def print_subject_summary(series_infos: list[SeriesInfo], rebin_stats: list[RebinStats], combined_spf: int) -> None:
    print(f"  selected combined_spf: {combined_spf}")
    for info in series_infos:
        print(
            f"  {info.hop_id}: csv_spf={info.original_spokes_per_frame} "
            f"n_spokes={info.n_spokes} n_samples={info.n_samples} n_coils={info.n_coils} "
            f"images_per_slab={info.images_per_slab}"
        )
    for stats in rebin_stats:
        print(f"  {format_stats_line(stats)}")


class TeeStream(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_subject_log_path(output_root: Path, step_name: str, subject_id: str) -> Path:
    log_dir = output_root / "logs" / step_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{subject_id}.log"


@contextlib.contextmanager
def tee_subject_log(log_path: Path):
    with open(log_path, "a", encoding="utf-8") as log_file:
        tee_out = TeeStream(sys.stdout, log_file)
        tee_err = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print()
            print(f"> subject log: {log_path}")
            yield


def workflow_recon_coils(ksp: np.ndarray, coil_config: CoilCalibrationConfig):
    return CoilMapEstimator(config=coil_config, device=COIL_DEVICE).estimate(ksp)


def prepare_subject_inputs(csv_path: Path, data_root: Path) -> tuple[str, list[SeriesInfo], int, dict[str, RebinStats]]:
    subject_id = subject_id_from_csv(csv_path)
    subject_root = data_root / subject_id
    configs = read_csv_config(csv_path)
    series_infos = collect_subject_series_infos(configs, subject_root)
    if STEP1_SLICE_IDX >= len(series_infos[0].slice_files):
        raise IndexError(
            f"STEP1_SLICE_IDX={STEP1_SLICE_IDX} outside available slice range 0..{len(series_infos[0].slice_files) - 1}"
        )
    combined_spf, rebin_stats = choose_combined_spokes_per_frame(series_infos)
    return subject_id, series_infos, combined_spf, {stats.hop_id: stats for stats in rebin_stats}


def save_step1_debug_artifacts(
    graph_dir: Path,
    step1_info: SeriesInfo,
    step1_stats: RebinStats,
    segmentation,
) -> None:
    metadata_lines = [
        f"step1_hop_id={step1_info.hop_id}",
        f"step1_original_spf={step1_info.original_spokes_per_frame}",
        f"step1_target_spf={step1_stats.target_spokes_per_frame}",
        f"step1_num_frames={step1_stats.num_frames}",
        f"step1_used_spokes={step1_stats.used_spokes}",
        f"step1_dropped_spokes={step1_stats.dropped_spokes}",
        f"step1_start_spoke={step1_stats.start_spoke}",
        f"baseline_idx={list(np.asarray(segmentation.baseline_idx, dtype=int))}",
        f"early_idx={list(np.asarray(segmentation.early_idx, dtype=int))}",
    ]
    (graph_dir / "step1_debug_info.txt").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    np.save(graph_dir / "baseline_idx.npy", np.asarray(segmentation.baseline_idx, dtype=np.int32))
    np.save(graph_dir / "early_idx.npy", np.asarray(segmentation.early_idx, dtype=np.int32))
    np.save(graph_dir / "norm_early_enh.npy", np.asarray(segmentation.norm_early_enh, dtype=np.float32))
    np.save(graph_dir / "pre_cleanup_vascular_mask.npy", np.asarray(segmentation.pre_cleanup_vascular_mask, dtype=bool))
    np.save(graph_dir / "vascular_mask.npy", np.asarray(segmentation.vascular_mask, dtype=bool))
    np.save(graph_dir / "tissue_mask.npy", np.asarray(segmentation.tissue_mask, dtype=bool))


def summarize_array_debug(name: str, array) -> str:
    arr = np.asarray(array)
    finite = np.isfinite(arr).all()
    abs_arr = np.abs(arr)
    return (
        f"{name}: shape={arr.shape} dtype={arr.dtype} finite={finite} "
        f"abs_min={float(abs_arr.min()):.6g} abs_max={float(abs_arr.max()):.6g}"
    )


def run_step1_subject(
    csv_path: Path,
    data_root: Path,
    output_root: Path,
    coil_thresh: float,
    recon_device,
    step1_hop_id: str = "DCE",
) -> tuple[str, Path]:
    subject_id, series_infos, combined_spf, stats_map = prepare_subject_inputs(csv_path, data_root)
    combined_hop_id = build_combined_hop_id(series_infos)
    step1_info = select_step1_series_info(series_infos, step1_hop_id=step1_hop_id)
    step1_stats = compute_rebin_stats(
        hop_id=step1_info.hop_id,
        original_spf=step1_info.original_spokes_per_frame,
        n_spokes=step1_info.n_spokes,
        target_spf=step1_info.original_spokes_per_frame,
    )
    rebin_stats = [stats_map[info.hop_id] for info in series_infos]
    step1_dir = get_step1_dir(output_root, subject_id, combined_hop_id)
    graph_dir = get_step1_graph_dir(output_root, subject_id, combined_hop_id)
    step1_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"step1 subject: {subject_id}")
    print(f"csv: {csv_path}")
    print_subject_summary(series_infos, rebin_stats, combined_spf)
    print(f"  recon_device: {recon_device}")
    print(f"  coil_device: {COIL_DEVICE}")
    print(f"  step1_hop_id: {step1_info.hop_id}")
    print(f"  {format_stats_line(step1_stats)}")

    workflow = make_step1_workflow(spokes_per_frame=step1_stats.target_spokes_per_frame, coil_thresh=coil_thresh)
    step1_traj = build_series_traj(step1_info, step1_stats)
    step1_ksp = load_series_slice_kspace(step1_info, step1_stats, STEP1_SLICE_IDX)

    start_time = time.perf_counter()
    mps = workflow.estimate_coils(step1_ksp)
    img_lowres = workflow.reconstruct_lowres_series(step1_ksp, mps, traj=step1_traj)
    segmentation = workflow.segment_vascular_and_tissue(img_lowres)
    vascular_basis, tissue_basis, _ = workflow.estimate_segmented_basis(img_lowres, segmentation)
    elapsed = time.perf_counter() - start_time

    if vascular_basis is None or tissue_basis is None:
        raise ValueError("Segmented basis output is required but vascular/tissue basis is missing.")

    save_step1_graphs(SimpleNamespace(img_lowres=img_lowres, segmentation=segmentation), graph_dir)
    save_step1_debug_artifacts(graph_dir, step1_info, step1_stats, segmentation)
    basis = np.concatenate([vascular_basis[:, :3], tissue_basis[:, :3]], axis=1)
    basis_path = get_basis_path(output_root, subject_id, combined_hop_id)
    workflow.save_basis(basis, basis_path)

    print(f"  step1 saved basis: {basis_path}")
    print(f"  step1 subject={subject_id} group={combined_hop_id} slice={STEP1_SLICE_IDX}")
    print(f"  step1 source_hop={step1_info.hop_id} frames={step1_traj.shape[0]} spf={step1_stats.target_spokes_per_frame}")
    print(f"  step1 total time: {format_elapsed(elapsed)}")
    return subject_id, basis_path


def run_step2_subject(
    csv_path: Path,
    data_root: Path,
    output_root: Path,
    coil_thresh: float,
    recon_device,
    slice_indices: list[int] | None = None,
) -> tuple[str, list[tuple[str, str, str, str]]]:
    subject_id, series_infos, combined_spf, stats_map = prepare_subject_inputs(csv_path, data_root)
    combined_hop_id = build_combined_hop_id(series_infos)
    basis_path = require_basis_path(output_root, subject_id, combined_hop_id)
    step2_dir = get_step2_dir(output_root, subject_id, combined_hop_id)
    graph_dir = get_step2_graph_dir(output_root, subject_id, combined_hop_id)
    step2_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"step2 subject: {subject_id}")
    print(f"csv: {csv_path}")
    print(f"  basis: {basis_path}")
    print(f"  recon_device: {recon_device}")
    print(f"  coil_device: {COIL_DEVICE}")

    combined_traj = build_combined_traj(series_infos, stats_map)
    failures: list[tuple[str, str, str, str]] = []
    elapsed_times: list[float] = []
    total_start = time.perf_counter()
    basis_debug = None
    basis_h5 = None
    try:
        import h5py  # local import to keep module import lighter

        with h5py.File(basis_path, "r") as h5_file:
            basis_h5 = np.asarray(h5_file["bases"][:])
        basis_debug = summarize_array_debug("basis_h5", basis_h5)
    except Exception as exc:  # noqa: BLE001
        basis_debug = f"basis_h5: failed_to_read error={exc}"

    target_slices = list(range(len(series_infos[0].slice_files))) if slice_indices is None else slice_indices

    for slice_idx in target_slices:
        out_path = step2_dir / f"{combined_hop_id}_slice_{slice_idx:03d}.h5"
        workflow = make_step2_workflow(
            out_path=out_path,
            slice_idx=slice_idx,
            combined_hop_id=combined_hop_id,
            coil_thresh=coil_thresh,
            recon_device=recon_device,
        )
        combined_ksp = None
        coil_maps = None
        recon_result = None

        try:
            slice_start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            print(f"    {format_gpu_mem_stats(f'slice {slice_idx:03d} pre')}")
            combined_ksp = load_combined_slice_kspace(series_infos, stats_map, slice_idx)
            coil_maps = workflow_recon_coils(combined_ksp, workflow.config.coil)
            print(f"    {format_gpu_mem_stats(f'slice {slice_idx:03d} post-coils')}")
            recon_result = workflow.run_slice(
                ksp=combined_ksp,
                traj=combined_traj,
                mps=coil_maps,
                fbasis_path=basis_path,
                spokes_per_frame=combined_spf,
            )
            print(f"    {format_gpu_mem_stats(f'slice {slice_idx:03d} post-recon')}")
            save_slice_h5(
                out_path=out_path,
                acq_slice=np.asarray(recon_result.img_dyn),
                hop_id=combined_hop_id,
                spokes_per_frame=combined_spf,
                N_time=recon_result.img_dyn.shape[0],
                slice_idx=slice_idx,
                smax=np.max(recon_result.img_dyn_abs),
            )
            elapsed = time.perf_counter() - slice_start
            elapsed_times.append(elapsed)
            if slice_idx == STEP1_SLICE_IDX:
                save_step2_tic_graphs(recon_result, graph_dir)
            print(f"    slice {slice_idx:03d} recon time {format_elapsed(elapsed)} | saved {out_path}")
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc):
                raise
            print(f"    DEBUG slice {slice_idx:03d} combined_hop_id={combined_hop_id}")
            for row in collect_slice_debug_rows(series_infos, stats_map, slice_idx):
                print(f"      {row}")
            if "combined_ksp" in locals():
                print(f"      {summarize_array_debug('combined_ksp', combined_ksp)}")
            if "coil_maps" in locals():
                print(f"      {summarize_array_debug('coil_maps', coil_maps)}")
            print(f"      {basis_debug}")
            print(
                f"      combined_traj_shape={getattr(combined_traj, 'shape', None)} "
                f"basis_path={basis_path}"
            )
            failures.append((subject_id, combined_hop_id, f"slice_{slice_idx:03d}", str(exc)))
            print(f"    FAILED slice {slice_idx:03d}: {exc}")
        finally:
            recon_result = None
            coil_maps = None
            combined_ksp = None
            workflow = None
            cleanup_slice_recon_state()
            print(f"    {format_gpu_mem_stats(f'slice {slice_idx:03d} post-cleanup')}")

    total_elapsed = time.perf_counter() - total_start
    if elapsed_times:
        avg_elapsed = total_elapsed / len(elapsed_times)
        print(
            f"  step2 total time: {format_elapsed(total_elapsed)} | "
            f"avg per slice {format_elapsed(avg_elapsed)} | "
            f"slices {len(elapsed_times)}"
        )
    return subject_id, failures
