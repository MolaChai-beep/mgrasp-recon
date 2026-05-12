import csv
import os
import sys
from pathlib import Path

repo_root = Path.cwd()
src_root = repo_root / "src"
if src_root.exists() and str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torch

from mgrasp_recon import (
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
from mgrasp_recon.recon_utils import get_traj, infer_kspace_dims, list_slice_files, read_csv_config


LOWRES_SHAPES = [(256, 256), (192, 192), (128, 128)]
STEP1_OUTPUT_ROOT = repo_root / "outputs" / "step1_test_multi_low_res"
OUTPUT_ROOT = repo_root / "outputs" / "noise_check"
FRAME_TIME_SEC = 3.8
LAMBDA_TEST_VAL = [1e-3]
VESSEL_VOXEL = (200, 135)
TISSUE_VOXELS = [(180, 120)]


def _shape_tag(shape):
    return f"lowres_{shape[0]}x{shape[1]}"


def _lambda_tag(value):
    return f"{value:.0e}".replace("-", "m")


def _save_figure(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_metrics_csv(out_path, rows):
    fieldnames = [
        "resolution",
        "lambda",
        "vessel_row",
        "vessel_col",
        "vascular_mask_size",
        "tissue_mask_size",
        "single_voxel_high_freq_std",
        "vessel_roi_high_freq_std",
        "tissue_roi_high_freq_std",
        "single_voxel_std",
        "vessel_roi_std",
        "tissue_roi_std",
        "single_voxel_peak_enhancement",
        "vessel_roi_peak_enhancement",
        "tissue_roi_peak_enhancement",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class Args:
    base_dir = "//home/naiqianluan/DCE-MRI/data/DCE_data/20250827-110742-Gross_MeyerA/RAVE_files/h5slices_wt_acqT/"
    csv_path = "/home/naiqianluan/DCE-MRI/data/DCE_data/20250827-110742-Gross_MeyerA/RAVE_files/config_subject.csv"


args = Args()
subject_id = "Gross_MeyerA"
ticker = TicAnalyzer()
device = sp.Device(0 if torch.cuda.is_available() else -1)

print("> device ", device)
print()
print(f"Reading configurations from: {args.csv_path}")
configs = read_csv_config(args.csv_path)
print(f"Found {len(configs)} configurations to process")
print()

for config in configs:
    args.hop_id = config["hop_id"]
    args.spokes_per_frame = 46
    args.slice_idx = 47
    args.slice_inc = 1
    args.images_per_slab = config["images_per_slab"]

    print("=" * 60)
    print(f"Processing {args.hop_id}")
    print("=" * 60)
    print(f"  spokes_per_frame: {args.spokes_per_frame}")
    print(f"  slice_idx: {args.slice_idx}")
    print(f"  slice_inc: {args.slice_inc}")
    print(f"  images_per_slab: {args.images_per_slab}")

    hop_dir = os.path.join(args.base_dir, args.hop_id)
    if not os.path.exists(hop_dir):
        raise FileNotFoundError(f"Directory not found: {hop_dir}")

    slice_files = list_slice_files(hop_dir)
    n_slices_total = len(slice_files)
    print(f"> Found {n_slices_total} slice files in {hop_dir}")
    n_coils, n_samples, n_spokes, n_slices = infer_kspace_dims(slice_files[0])
    print(
        f"> Inferred k-space dimensions from first slice: "
        f"slices={n_slices}, spokes={n_spokes}, samples={n_samples}, coils={n_coils}"
    )
    base_res = n_samples // 2
    n_time = n_spokes // args.spokes_per_frame
    if n_time <= 0:
        raise ValueError(
            f"n_time={n_time}. Check spokes_per_frame={args.spokes_per_frame} vs n_spokes={n_spokes}"
        )

    traj = get_traj(N_spokes=args.spokes_per_frame, N_time=n_time, base_res=base_res, gind=1)
    print(f"> dims: coils={n_coils}, spokes={n_spokes}, samples={n_samples}, n_time={n_time}")
    print(f"  traj shape: {traj.shape}")

    s = args.slice_idx
    sf = slice_files[s]
    print()
    print(f">>> slice {str(s).zfill(3)} | {os.path.basename(sf)}")

    hop_output_root = OUTPUT_ROOT / subject_id / args.hop_id
    hop_output_root.mkdir(parents=True, exist_ok=True)

    vessel_resolution_curves = {}
    tissue_resolution_curves = {}
    metric_rows = []

    for img_shape in LOWRES_SHAPES:
        shape_tag = _shape_tag(img_shape)
        step1_hop_dir = STEP1_OUTPUT_ROOT / subject_id / args.hop_id / shape_tag
        basis_path = step1_hop_dir / "fbasis.h5"
        if not basis_path.exists():
            raise FileNotFoundError(f"Missing step1 basis: {basis_path}")

        basis_output_dir = hop_output_root / shape_tag
        basis_output_dir.mkdir(parents=True, exist_ok=True)

        print()
        print(f">>> running {shape_tag}")
        print(f"  basis path: {basis_path}")
        print(f"  output dir: {basis_output_dir}")

        basis_workflow = BasisPreparationWorkflow(
            BasisPreparationConfig(
                spokes_per_frame=args.spokes_per_frame,
                lowres=LowResReconConfig(
                    img_shape=img_shape,
                    ns_low=256,
                    method="adjoint",
                    normalize=False,
                    return_complex=False,
                    use_ramp_filter=True,
                    verbose=True,
                ),
                coil=CoilCalibrationConfig(thresh=0.02, verbose=True),
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
        step1_result = basis_workflow.run(slice_files=slice_files, slice_idx=s, traj=traj)
        seg = step1_result.segmentation
        basis = step1_result.basis

        fig_overlay, _ = ticker.plot_roi_overlay(
            background_img=seg.mean_img,
            vascular_mask=seg.vascular_mask,
            tissue_mask=seg.tissue_mask,
            vessel_coord=VESSEL_VOXEL,
            title=f"{shape_tag}: vessel and normal brain ROIs",
        )
        _save_figure(fig_overlay, basis_output_dir / "roi_overlay.png")

        fig_basis, _ = ticker.plot_basis_curves(
            basis=basis,
            title=f"{shape_tag}: basis curves",
        )
        _save_figure(fig_basis, basis_output_dir / "basis_curves.png")

        for lambda_test in LAMBDA_TEST_VAL:
            lambda_label = f"{lambda_test:.3g}"
            lambda_str = _lambda_tag(lambda_test)
            lambda_output_dir = basis_output_dir / f"lambda_{lambda_str}"
            lambda_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"    lambda = {lambda_label}")

            workflow = SliceReconstructionWorkflow(
                SliceReconstructionConfig(
                    recon=ReconstructionConfig(
                        nbasis=5,
                        cbasis=False,
                        add_constant=True,
                        lamda=lambda_test,
                        regu="TV",
                        regu_axes=(-2, -1),
                        max_iter=10,
                        solver="ADMM",
                        use_dcf=True,
                        show_pbar=False,
                        verbose=True,
                    ),
                    coil=CoilCalibrationConfig(thresh=0.02, verbose=True),
                    hop_id=args.hop_id,
                    slice_idx=s,
                    coil_device=-1,
                    recon_device=device,
                )
            )
            recon_result = workflow.reconstruct_slice(
                slice_file=sf,
                traj=traj,
                fbasis_path=basis_path,
                spokes_per_frame=args.spokes_per_frame,
                slice_idx=s,
            )
            recon_seg = basis_workflow.segmentation_analyzer.segment_enhancement_series(
                recon_result.img_dyn_abs
            )

            summary = ticker.summarize_roi_tics(
                recon_result.img_dyn_abs,
                vascular_mask=recon_seg.vascular_mask,
                tissue_mask=recon_seg.tissue_mask,
                vessel_coord=VESSEL_VOXEL,
                normalize=False,
            )
            metrics = ticker.summarize_noise_metrics(
                recon_result.img_dyn_abs,
                vascular_mask=recon_seg.vascular_mask,
                tissue_mask=recon_seg.tissue_mask,
                vessel_coord=VESSEL_VOXEL,
                normalize=False,
            )

            fig_overlay_marked, _ = ticker.plot_roi_overlay(
                background_img=recon_result.img_dyn_abs.mean(axis=0),
                vascular_mask=recon_seg.vascular_mask,
                tissue_mask=recon_seg.tissue_mask,
                vessel_coord=VESSEL_VOXEL,
                title=f"{shape_tag}, lambda={lambda_label}: vessel and normal brain ROIs",
            )
            _save_figure(fig_overlay_marked, lambda_output_dir / "roi_overlay_recon.png")

            fig_tic, _ = ticker.plot_roi_tic_summary(
                img_dyn=recon_result.img_dyn_abs,
                vascular_mask=recon_seg.vascular_mask,
                tissue_mask=recon_seg.tissue_mask,
                vessel_coord=VESSEL_VOXEL,
                frame_time_sec=FRAME_TIME_SEC,
                normalize=False,
                title=f"{shape_tag}, lambda={lambda_label}: single voxel vs ROI TICs",
            )
            _save_figure(fig_tic, lambda_output_dir / "tic_single_vs_roi.png")

            fig_voxels, _ = ticker.show_selected_voxels(
                background_img=recon_result.img_dyn_abs.mean(axis=0),
                vessel_coord=VESSEL_VOXEL,
                tissue_coords=TISSUE_VOXELS,
                vascular_mask=recon_seg.vascular_mask,
                tissue_mask=recon_seg.tissue_mask,
                title=f"{shape_tag}, lambda={lambda_label}: selected vessel/tissue voxels",
            )
            _save_figure(fig_voxels, lambda_output_dir / "selected_voxels.png")

            fig_voxel_tics, _ = ticker.plot_time_intensity_curves(
                img_dyn=recon_result.img_dyn_abs,
                vessel_coord=VESSEL_VOXEL,
                tissue_coords=TISSUE_VOXELS,
                frame_time_sec=FRAME_TIME_SEC,
                normalize=False,
                title=f"{shape_tag}, lambda={lambda_label}: selected voxel TICs",
            )
            _save_figure(fig_voxel_tics, lambda_output_dir / "selected_voxel_tics.png")

            vessel_resolution_curves[shape_tag] = summary["vessel_roi"]
            tissue_resolution_curves[shape_tag] = summary["tissue_roi"]

            metric_rows.append(
                {
                    "resolution": shape_tag,
                    "lambda": lambda_label,
                    "vessel_row": summary["vessel_coord"][0],
                    "vessel_col": summary["vessel_coord"][1],
                    "vascular_mask_size": metrics["vascular_mask_size"],
                    "tissue_mask_size": metrics["tissue_mask_size"],
                    "single_voxel_high_freq_std": metrics["single_voxel"]["high_freq_std"],
                    "vessel_roi_high_freq_std": metrics["vessel_roi"]["high_freq_std"],
                    "tissue_roi_high_freq_std": metrics["tissue_roi"]["high_freq_std"],
                    "single_voxel_std": metrics["single_voxel"]["std"],
                    "vessel_roi_std": metrics["vessel_roi"]["std"],
                    "tissue_roi_std": metrics["tissue_roi"]["std"],
                    "single_voxel_peak_enhancement": metrics["single_voxel"]["peak_enhancement"],
                    "vessel_roi_peak_enhancement": metrics["vessel_roi"]["peak_enhancement"],
                    "tissue_roi_peak_enhancement": metrics["tissue_roi"]["peak_enhancement"],
                }
            )

    fig_vessel_cmp, _ = ticker.plot_resolution_comparison(
        curves_by_label=vessel_resolution_curves,
        frame_time_sec=FRAME_TIME_SEC,
        normalize=False,
        title="Vessel ROI TIC comparison across resolutions",
    )
    _save_figure(fig_vessel_cmp, hop_output_root / "vessel_roi_tic_comparison.png")

    fig_tissue_cmp, _ = ticker.plot_resolution_comparison(
        curves_by_label=tissue_resolution_curves,
        frame_time_sec=FRAME_TIME_SEC,
        normalize=False,
        title="Normal brain ROI TIC comparison across resolutions",
    )
    _save_figure(fig_tissue_cmp, hop_output_root / "normal_brain_roi_tic_comparison.png")

    summary_path = hop_output_root / "noise_metrics.csv"
    _write_metrics_csv(summary_path, metric_rows)

    corr_path = hop_output_root / "resolution_curve_correlation.csv"
    with open(corr_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["curve_type", "pair", "correlation"])
        for pair, value in ticker.compare_resolution_tics(vessel_resolution_curves).items():
            writer.writerow(["vessel_roi", pair, f"{value:.6f}"])
        for pair, value in ticker.compare_resolution_tics(tissue_resolution_curves).items():
            writer.writerow(["normal_brain_roi", pair, f"{value:.6f}"])

    print(f"Saved noise metrics to: {summary_path}")
    print(f"Saved resolution correlation summary to: {corr_path}")
