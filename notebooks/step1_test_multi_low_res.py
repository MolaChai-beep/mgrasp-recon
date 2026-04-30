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
    SegmentationConfig,
)
from mgrasp_recon.recon_utils import get_traj, infer_kspace_dims, list_slice_files, read_csv_config
from mgrasp_recon.visualization import plot_segmentation_summary


LOWRES_SHAPES = [(256, 256), (192, 192), (128, 128)]
OUTPUT_ROOT = repo_root / "outputs" / "step1_test_multi_low_res"


def _shape_tag(shape):
    return f"lowres_{shape[0]}x{shape[1]}"


def _save_figure(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class Args:
    base_dir = "//home/naiqianluan/DCE-MRI/data/DCE_data/20250827-110742-Gross_MeyerA/RAVE_files/h5slices_wt_acqT/"
    csv_path = "/home/naiqianluan/DCE-MRI/data/DCE_data/20250827-110742-Gross_MeyerA/RAVE_files/config_subject.csv"

args = Args()
subject_id = "Gross_MeyerA"

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

    s = 47
    sf = slice_files[s]
    print()
    print(f">>> slice {str(s).zfill(3)} | {os.path.basename(sf)}")
    hop_output_root = OUTPUT_ROOT / subject_id / args.hop_id

    for img_shape in LOWRES_SHAPES:
        shape_tag = _shape_tag(img_shape)
        run_output_dir = hop_output_root / shape_tag
        run_output_dir.mkdir(parents=True, exist_ok=True)

        print()
        print(f">>> running {shape_tag}")
        print(f"  output dir: {run_output_dir}")

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

        step1_result = basis_workflow.run(slice_files=slice_files, slice_idx=s, traj=traj)
        img_init = step1_result.img_lowres
        mps = step1_result.mps
        seg = step1_result.segmentation
        vascular_basis = step1_result.vascular_basis
        tissue_basis = step1_result.tissue_basis

        print("  mps shape:", np.asarray(sp.to_device(mps, sp.cpu_device)).shape)
        print("  img_init shape:", img_init.shape)

        fig_preview, axes_preview = plt.subplots(1, 3, figsize=(12, 4))
        for i, tt in enumerate([0, img_init.shape[0] // 2, img_init.shape[0] - 1]):
            axes_preview[i].imshow(np.abs(img_init[tt]), cmap="gray")
            axes_preview[i].set_title(f"Frame {tt}")
            axes_preview[i].axis("off")
        _save_figure(fig_preview, run_output_dir / "img_lowres_preview.png")

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
        _save_figure(fig_seg, run_output_dir / "segmentation_summary.png")

        K_show = 5

        fig_basis, ax_basis = plt.subplots(figsize=(12, 6))
        for k in range(min(K_show, vascular_basis.shape[1])):
            ax_basis.plot(vascular_basis[:, k], linewidth=2, label=f"Vascular PC{k + 1}")
        for k in range(min(K_show, tissue_basis.shape[1])):
            ax_basis.plot(tissue_basis[:, k], linestyle="--", linewidth=2, label=f"Tissue PC{k + 1}")

        ax_basis.set_title("Vascular and Tissue PCA components")
        ax_basis.set_xlabel("Frame")
        ax_basis.set_ylabel("Amplitude")
        ax_basis.legend(ncol=2, fontsize=9)
        _save_figure(fig_basis, run_output_dir / "basis_components.png")

        vascular_energy = np.sum(vascular_basis ** 2, axis=0)
        tissue_energy = np.sum(tissue_basis ** 2, axis=0)

        print("vascular basis energy:", vascular_energy[:10])
        print("tissue basis energy:", tissue_energy[:10])

        fig_energy, axes_energy = plt.subplots(1, 2, figsize=(12, 4))
        axes_energy[0].plot(vascular_energy, "o-")
        axes_energy[0].set_title("Vascular basis energy")
        axes_energy[0].set_xlabel("Component")
        axes_energy[0].set_ylabel("Energy")

        axes_energy[1].plot(tissue_energy, "o-")
        axes_energy[1].set_title("Tissue basis energy")
        axes_energy[1].set_xlabel("Component")
        axes_energy[1].set_ylabel("Energy")
        _save_figure(fig_energy, run_output_dir / "basis_energy.png")

        basis = np.concatenate([vascular_basis[:, :3], tissue_basis[:, :3]], axis=1)
        out_path = run_output_dir / "fbasis.h5"
        basis_workflow.save_basis(basis, out_path)
        print(f"Saved basis to: {out_path}")




