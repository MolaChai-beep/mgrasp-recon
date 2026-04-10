"""Batch Step 2 reconstruction for all slices of one patient.

This script mirrors the configuration used in step2_basis_recon.ipynb, but
reconstructs every slice file for each hop listed in config_subject.csv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import sigpy as sp
import torch


def _add_repo_src_to_path() -> Path:
    """Add the repository src directory when running this file directly."""
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if src_root.exists() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return repo_root


REPO_ROOT = _add_repo_src_to_path()

from mgrasp_recon import ReconstructionConfig, SliceReconstructionConfig, SliceReconstructionWorkflow  # noqa: E402
from mgrasp_recon.recon_utils import get_traj, infer_kspace_dims, list_slice_files, read_csv_config  # noqa: E402


# Step 2 notebook defaults.
SUBJECT_ID = "Gross_MeyerA"
BASE_DIR = "//home/naiqianluan/DCE-MRI/data/DCE_data/20250827-110742-Gross_MeyerA/RAVE_files/h5slices_wt_acqT/"
CSV_PATH = "/home/naiqianluan/DCE-MRI/data/DCE_data/20250827-110742-Gross_MeyerA/RAVE_files/config_subject.csv"
SPOKES_PER_FRAME = 46
BASIS_CONFIGS = [
    ("basis8", "/home/naiqianluan/DCE-MRI/data/test-pro-grasp/fbasis_8.h5", 8),
]
LAMBDA_VALUES = [1e-3]
SAVE_DIR = REPO_ROOT / "lambda_test_outputs"
START_SLICE_IDX = 49


def lambda_to_output_label(lambda_value: float) -> str:
    return f"{lambda_value:.0e}".replace("-", "m")


def make_reconstruction_workflow(
    hop_id: str,
    slice_idx: int,
    output_path: Path,
    nbasis: int,
    lambda_value: float,
    recon_device: sp.Device,
) -> SliceReconstructionWorkflow:
    return SliceReconstructionWorkflow(
        SliceReconstructionConfig(
            recon=ReconstructionConfig(
                nbasis=nbasis,
                cbasis=False,
                add_constant=True,
                lamda=lambda_value,
                regu="TV",
                regu_axes=(-2, -1),
                max_iter=10,
                solver="ADMM",
                use_dcf=True,
                show_pbar=False,
            ),
            save_h5=True,
            out_path=output_path,
            hop_id=hop_id,
            slice_idx=slice_idx,
            coil_device=-1,
            recon_device=recon_device,
        )
    )


def reconstruct_slice(
    slice_file: str,
    traj: np.ndarray,
    hop_id: str,
    slice_idx: int,
    basis_name: str,
    fbasis_path: str,
    nbasis: int,
    lambda_value: float,
    recon_device: sp.Device,
    patient_output_dir: Path,
) -> None:
    lambda_str = lambda_to_output_label(lambda_value)
    output_path = patient_output_dir / f"{hop_id}_slice_{slice_idx:03d}_{basis_name}_lambda_{lambda_str}.h5"

    workflow = make_reconstruction_workflow(
        hop_id=hop_id,
        slice_idx=slice_idx,
        output_path=output_path,
        nbasis=nbasis,
        lambda_value=lambda_value,
        recon_device=recon_device,
    )
    result = workflow.reconstruct_slice(
        slice_file=slice_file,
        traj=traj,
        fbasis_path=fbasis_path,
        spokes_per_frame=SPOKES_PER_FRAME,
        slice_idx=slice_idx,
    )

    print(f"    saved slice recon to: {output_path}")
    print("    coeff_maps shape:", result.coeff_maps.shape)
    print("    img_dyn_abs shape:", result.img_dyn_abs.shape)
    print("    mps shape:", np.asarray(sp.to_device(result.mps, sp.cpu_device)).shape)


def main() -> int:
    recon_device = sp.Device(0 if torch.cuda.is_available() else -1)
    failures: list[tuple[str, str, str]] = []

    print("> patient ", SUBJECT_ID)
    print("> device ", recon_device)
    print()
    print(f"Reading configurations from: {CSV_PATH}")
    configs = read_csv_config(CSV_PATH)
    print(f"Found {len(configs)} configurations to process")
    print()

    for config in configs:
        hop_id = config["hop_id"]
        images_per_slab = config["images_per_slab"]
        hop_dir = os.path.join(BASE_DIR, hop_id)

        print("=" * 60)
        print(f"Processing {hop_id}")
        print("=" * 60)
        print(f"  patient: {SUBJECT_ID}")
        print(f"  spokes_per_frame: {SPOKES_PER_FRAME}")
        print(f"  images_per_slab: {images_per_slab}")

        if not os.path.exists(hop_dir):
            raise FileNotFoundError(f"Directory not found: {hop_dir}")

        slice_files = list_slice_files(hop_dir)
        slice_files = slice_files[49:]
        print(f"> Found {len(slice_files)} slice files in {hop_dir}")

        n_coils, n_samples, n_spokes, n_slices = infer_kspace_dims(slice_files[0])
        print(
            "> Inferred k-space dimensions from first slice: "
            f"slices={n_slices}, spokes={n_spokes}, samples={n_samples}, coils={n_coils}"
        )

        base_res = n_samples // 2
        n_time = n_spokes // SPOKES_PER_FRAME
        if n_time <= 0:
            raise ValueError(
                f"n_time={n_time}. Check spokes_per_frame={SPOKES_PER_FRAME} vs n_spokes={n_spokes}"
            )

        traj = get_traj(N_spokes=SPOKES_PER_FRAME, N_time=n_time, base_res=base_res, gind=1)
        print(f"> dims: coils={n_coils}, spokes={n_spokes}, samples={n_samples}, n_time={n_time}")
        print(f"  traj shape: {traj.shape}")

        for basis_name, fbasis_path, nbasis in BASIS_CONFIGS:
            print()
            print(f">>> Testing {basis_name} | nbasis={nbasis}")

            for lambda_value in LAMBDA_VALUES:
                lambda_label = f"{lambda_value:.3g}"
                lambda_str = lambda_to_output_label(lambda_value)
                patient_output_dir = SAVE_DIR / basis_name / f"recon_h5_lambda_{lambda_str}" / SUBJECT_ID
                patient_output_dir.mkdir(parents=True, exist_ok=True)

                print(f"    lambda = {lambda_label}")
                print(f"    patient output dir: {patient_output_dir}")
                
                if START_SLICE_IDX < 0 or START_SLICE_IDX >= len(slice_files):
                                    raise ValueError(
                                        f"START_SLICE_IDX={START_SLICE_IDX} is outside the available slice range "
                                        f"0..{len(slice_files) - 1}"
                                    )

                print(f"    starting from slice index: {START_SLICE_IDX:03d}")
                print(f"    skipping {START_SLICE_IDX} already-processed slice(s)")

                for slice_idx, slice_file in enumerate(slice_files[START_SLICE_IDX:], start=START_SLICE_IDX):
                    print()
                    print(f">>> slice {slice_idx:03d} | {os.path.basename(slice_file)}")
                    try:
                        reconstruct_slice(
                            slice_file=slice_file,
                            traj=traj,
                            hop_id=hop_id,
                            slice_idx=slice_idx,
                            basis_name=basis_name,
                            fbasis_path=fbasis_path,
                            nbasis=nbasis,
                            lambda_value=lambda_value,
                            recon_device=recon_device,
                            patient_output_dir=patient_output_dir,
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append((hop_id, slice_file, str(exc)))
                        print(f"    FAILED: {exc}")

    print()
    print("=" * 60)
    if failures:
        print(f"Finished with {len(failures)} failed slices:")
        for hop_id, slice_file, error in failures:
            print(f"  {hop_id} | {slice_file} | {error}")
        return 1

    print("Finished successfully with no failed slices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
