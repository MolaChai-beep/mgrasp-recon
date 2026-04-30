import os
import sys
from pathlib import Path

script_root = Path(__file__).resolve().parent
repo_root = script_root / "mgrasp_recon_code"
src_root = repo_root / "src"
if src_root.exists() and str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torch

from mgrasp_recon import (
    CoilCalibrationConfig,
    ReconstructionConfig,
    SliceReconstructionConfig,
    SliceReconstructionWorkflow,
    TicAnalyzer,
)
from mgrasp_recon.recon_utils import get_traj, infer_kspace_dims, list_slice_files, read_csv_config


LOWRES_SHAPES = [(256, 256), (192, 192), (128, 128)]
STEP1_OUTPUT_ROOT = Path("/home/naiqianluan/DCE-MRI/code/mgrasp-recon/outputs/step1_test_multi_low_res")

OUTPUT_ROOT = Path("/home/naiqianluan/DCE-MRI/code/mgrasp-recon/outputs/step2_test_multi_low_res")
FRAME_TIME_SEC = 3.8
VOXEL_LIST = [(180, 120), (200, 135)]
LAMBDA_TEST_VAL = [1e-3]


def _shape_tag(shape):
    return f"lowres_{shape[0]}x{shape[1]}"


def _save_recon_arrays(out_path, result):
    np.savez_compressed(
        out_path,
        coeff_maps=np.asarray(result.coeff_maps),
        img_dyn_cplx=np.asarray(result.img_dyn_cplx),
        img_dyn_abs=np.asarray(result.img_dyn_abs),
        basisoption=np.asarray(result.basisoption),
        mps=np.asarray(result.mps),
        ksp=np.asarray(result.ksp),
    )


def _save_figure(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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

    s = 47
    sf = slice_files[s]
    print()
    print(f">>> slice {str(s).zfill(3)} | {os.path.basename(sf)}")

    step1_hop_dir = STEP1_OUTPUT_ROOT / subject_id / args.hop_id
    output_hop_dir = OUTPUT_ROOT / subject_id / args.hop_id

    for img_shape in LOWRES_SHAPES:
        basis_name = _shape_tag(img_shape)
        basis_path = step1_hop_dir / basis_name / "fbasis.h5"
        if not basis_path.exists():
            raise FileNotFoundError(f"Missing step1 basis: {basis_path}")

        basis_output_dir = output_hop_dir / basis_name
        basis_output_dir.mkdir(parents=True, exist_ok=True)

        print()
        print(f">>> running {basis_name}")
        print(f"  basis path: {basis_path}")
        print(f"  output dir: {basis_output_dir}")

        for lambda_test in LAMBDA_TEST_VAL:
            lambda_label = f"{lambda_test:.3g}"
            lambda_str = f"{lambda_test:.0e}".replace("-", "m")
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

            print("    coeff_maps shape:", recon_result.coeff_maps.shape)
            print("    img_dyn_abs shape:", recon_result.img_dyn_abs.shape)

            _save_recon_arrays(
                lambda_output_dir / f"recon_result_{basis_name}_lambda_{lambda_str}.npz",
                recon_result,
            )

            fig_frames, axes_frames = plt.subplots(1, 3, figsize=(12, 4))
            fig_frames.suptitle(f"{basis_name}, lambda = {lambda_label}", fontsize=14)
            for i, idx in enumerate(
                [0, recon_result.img_dyn_abs.shape[0] // 2, recon_result.img_dyn_abs.shape[0] - 1]
            ):
                axes_frames[i].imshow(recon_result.img_dyn_abs[idx], cmap="gray")
                axes_frames[i].set_title(f"Frame {idx}")
                axes_frames[i].axis("off")
            fig_frames.tight_layout(rect=[0, 0, 1, 0.92])
            fig_frames.savefig(
                lambda_output_dir / f"recon_frames_{basis_name}_lambda_{lambda_str}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig_frames)

            mean_img = recon_result.img_dyn_abs.mean(axis=0)
            for voxel_coord in VOXEL_LIST:
                row, col = voxel_coord

                fig_voxel, ax_voxel = plt.subplots(figsize=(6, 6))
                ax_voxel.imshow(mean_img, cmap="gray")
                ax_voxel.scatter([col], [row], c="red", s=80)
                ax_voxel.text(col + 2, row - 2, f"({row}, {col})", color="yellow")
                ax_voxel.set_title(f"{basis_name}, voxel {voxel_coord}\nlambda = {lambda_label}")
                ax_voxel.axis("off")
                _save_figure(
                    fig_voxel,
                    lambda_output_dir / f"voxel_location_{basis_name}_r{row}_c{col}_lambda_{lambda_str}.png",
                )

                curve = ticker.extract_voxel_tic(recon_result.img_dyn_abs, voxel_coord, normalize=False)
                time_axis = np.arange(len(curve)) * FRAME_TIME_SEC

                fig_tic, ax_tic = plt.subplots(figsize=(7, 4))
                ax_tic.plot(time_axis, curve, linewidth=2)
                ax_tic.set_xlabel("Time (s)")
                ax_tic.set_ylabel("Intensity")
                ax_tic.set_title(f"{basis_name}, TIC at voxel {voxel_coord}\nlambda = {lambda_label}")
                ax_tic.grid(alpha=0.3)
                _save_figure(
                    fig_tic,
                    lambda_output_dir / f"tic_{basis_name}_r{row}_c{col}_lambda_{lambda_str}.png",
                )
