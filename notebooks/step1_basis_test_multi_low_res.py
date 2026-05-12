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
    SegmentationConfig,
)
from mgrasp_recon.recon_utils import (
    estimate_pca_basis,
    get_traj,
    infer_kspace_dims,
    list_slice_files,
    read_csv_config,
)
from mgrasp_recon.visualization import plot_segmentation_summary


LOWRES_SHAPES = [(256, 256), (192, 192), (128, 128)]
OUTPUT_ROOT = repo_root / "outputs" / "step1_basis_test_multi_low_res"


def _shape_tag(shape):
    return f"lowres_{shape[0]}x{shape[1]}"


def _save_figure(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _explained_variance_ratio(singular_values, k):
    singular_values = np.asarray(singular_values, dtype=np.float64)
    power = singular_values ** 2
    total = np.sum(power)
    if total <= 0:
        return np.zeros(k, dtype=np.float64)
    return power[:k] / total


def _principal_angle_similarity(basis_a, basis_b):
    q_a, _ = np.linalg.qr(np.asarray(basis_a, dtype=np.float64))
    q_b, _ = np.linalg.qr(np.asarray(basis_b, dtype=np.float64))
    cosines = np.linalg.svd(q_a.T @ q_b, compute_uv=False)
    return float(np.mean(np.clip(cosines, 0.0, 1.0)))


def _subspace_similarity_matrix(items, key):
    n_items = len(items)
    matrix = np.zeros((n_items, n_items), dtype=np.float64)
    for i in range(n_items):
        for j in range(n_items):
            matrix[i, j] = _principal_angle_similarity(items[i][key], items[j][key])
    return matrix


def _save_similarity_outputs(out_dir, names, matrix, stem, title):
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["shape_tag", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *[f"{value:.6f}" for value in row]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean cosine of principal angles")
    _save_figure(fig, out_dir / f"{stem}.png")


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
    hop_output_root.mkdir(parents=True, exist_ok=True)

    basis_comparison_items = []
    summary_rows = []

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

        vascular_basis_recalc, vascular_singular_values = estimate_pca_basis(
            img_init,
            mask=seg.vascular_mask,
            K=vascular_basis.shape[1],
            remove_mean=True,
        )
        tissue_basis_recalc, tissue_singular_values = estimate_pca_basis(
            img_init,
            mask=seg.tissue_mask,
            K=tissue_basis.shape[1],
            remove_mean=True,
        )

        vascular_flip = np.sign(np.sum(vascular_basis * vascular_basis_recalc, axis=0))
        tissue_flip = np.sign(np.sum(tissue_basis * tissue_basis_recalc, axis=0))
        vascular_flip[vascular_flip == 0] = 1.0
        tissue_flip[tissue_flip == 0] = 1.0
        vascular_basis_aligned = vascular_basis * vascular_flip
        tissue_basis_aligned = tissue_basis * tissue_flip

        np.save(run_output_dir / "vascular_singular_values.npy", vascular_singular_values.astype(np.float32))
        np.save(run_output_dir / "tissue_singular_values.npy", tissue_singular_values.astype(np.float32))

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

        fig_basis, ax_basis = plt.subplots(figsize=(12, 6))
        for k in range(vascular_basis_aligned.shape[1]):
            ax_basis.plot(vascular_basis_aligned[:, k], linewidth=2, label=f"Vascular PC{k + 1}")
        for k in range(tissue_basis_aligned.shape[1]):
            ax_basis.plot(tissue_basis_aligned[:, k], linestyle="--", linewidth=2, label=f"Tissue PC{k + 1}")
        ax_basis.set_title("Vascular and Tissue PCA components (sign-aligned)")
        ax_basis.set_xlabel("Frame")
        ax_basis.set_ylabel("Amplitude")
        ax_basis.legend(ncol=2, fontsize=9)
        _save_figure(fig_basis, run_output_dir / "basis_components.png")

        vascular_energy = np.sum(vascular_basis_aligned ** 2, axis=0)
        tissue_energy = np.sum(tissue_basis_aligned ** 2, axis=0)
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

        vascular_evr = _explained_variance_ratio(vascular_singular_values, vascular_basis.shape[1])
        tissue_evr = _explained_variance_ratio(tissue_singular_values, tissue_basis.shape[1])

        fig_singular, axes_singular = plt.subplots(1, 2, figsize=(12, 4))
        axes_singular[0].plot(np.arange(1, len(vascular_singular_values[: vascular_basis.shape[1]]) + 1), vascular_singular_values[: vascular_basis.shape[1]], "o-")
        axes_singular[0].set_title("Vascular singular values")
        axes_singular[0].set_xlabel("Component")
        axes_singular[0].set_ylabel("Singular value")
        axes_singular[1].plot(np.arange(1, len(tissue_singular_values[: tissue_basis.shape[1]]) + 1), tissue_singular_values[: tissue_basis.shape[1]], "o-")
        axes_singular[1].set_title("Tissue singular values")
        axes_singular[1].set_xlabel("Component")
        axes_singular[1].set_ylabel("Singular value")
        _save_figure(fig_singular, run_output_dir / "singular_values.png")

        fig_evr, axes_evr = plt.subplots(1, 2, figsize=(12, 4))
        axes_evr[0].plot(np.arange(1, len(vascular_evr) + 1), vascular_evr, "o-")
        axes_evr[0].set_title("Vascular explained variance ratio")
        axes_evr[0].set_xlabel("Component")
        axes_evr[0].set_ylabel("Variance ratio")
        axes_evr[1].plot(np.arange(1, len(tissue_evr) + 1), tissue_evr, "o-")
        axes_evr[1].set_title("Tissue explained variance ratio")
        axes_evr[1].set_xlabel("Component")
        axes_evr[1].set_ylabel("Variance ratio")
        _save_figure(fig_evr, run_output_dir / "explained_variance_ratio.png")

        basis = np.concatenate([vascular_basis_aligned[:, :3], tissue_basis_aligned[:, :3]], axis=1)
        out_path = run_output_dir / "fbasis.h5"
        basis_workflow.save_basis(basis, out_path)
        print(f"Saved basis to: {out_path}")

        basis_comparison_items.append(
            {
                "shape_tag": shape_tag,
                "vascular_basis": vascular_basis_aligned,
                "tissue_basis": tissue_basis_aligned,
            }
        )
        summary_rows.append(
            {
                "shape_tag": shape_tag,
                "vascular_singular_values": vascular_singular_values[:5],
                "tissue_singular_values": tissue_singular_values[:5],
                "vascular_cumvar_1": float(np.sum(vascular_evr[:1])),
                "vascular_cumvar_2": float(np.sum(vascular_evr[:2])),
                "vascular_cumvar_3": float(np.sum(vascular_evr[:3])),
                "tissue_cumvar_1": float(np.sum(tissue_evr[:1])),
                "tissue_cumvar_2": float(np.sum(tissue_evr[:2])),
                "tissue_cumvar_3": float(np.sum(tissue_evr[:3])),
            }
        )

    shape_names = [item["shape_tag"] for item in basis_comparison_items]
    vascular_similarity = _subspace_similarity_matrix(basis_comparison_items, "vascular_basis")
    tissue_similarity = _subspace_similarity_matrix(basis_comparison_items, "tissue_basis")

    _save_similarity_outputs(
        hop_output_root,
        shape_names,
        vascular_similarity,
        "vascular_subspace_similarity",
        "Vascular subspace similarity",
    )
    _save_similarity_outputs(
        hop_output_root,
        shape_names,
        tissue_similarity,
        "tissue_subspace_similarity",
        "Tissue subspace similarity",
    )

    summary_path = hop_output_root / "basis_comparison_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "shape_tag",
                "vascular_s1",
                "vascular_s2",
                "vascular_s3",
                "vascular_s4",
                "vascular_s5",
                "tissue_s1",
                "tissue_s2",
                "tissue_s3",
                "tissue_s4",
                "tissue_s5",
                "vascular_cumvar_1",
                "vascular_cumvar_2",
                "vascular_cumvar_3",
                "tissue_cumvar_1",
                "tissue_cumvar_2",
                "tissue_cumvar_3",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["shape_tag"],
                    *[f"{value:.6g}" for value in row["vascular_singular_values"]],
                    *[f"{value:.6g}" for value in row["tissue_singular_values"]],
                    f'{row["vascular_cumvar_1"]:.6f}',
                    f'{row["vascular_cumvar_2"]:.6f}',
                    f'{row["vascular_cumvar_3"]:.6f}',
                    f'{row["tissue_cumvar_1"]:.6f}',
                    f'{row["tissue_cumvar_2"]:.6f}',
                    f'{row["tissue_cumvar_3"]:.6f}',
                ]
            )

    print(f"Saved hop-level basis comparison summary to: {summary_path}")
