"""Batch Step 3 DICOM export for all reconstructed slices of one patient.

This script mirrors the layout of step2_recon_all_slices.py, but exports the
reconstructed ``temptv`` series to DICOM using one template scan plus
``par.json`` metadata.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pydicom


def _add_repo_src_to_path() -> Path:
    """Add the repository src directory when running this file directly."""
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if src_root.exists() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return repo_root


REPO_ROOT = _add_repo_src_to_path()

from mgrasp_recon.recon_utils import read_csv_config  # noqa: E402


# Step 3 notebook defaults.
SUBJECT_ID = "Gross_MeyerA"
CSV_PATH = "/home/naiqianluan/DCE-MRI/data/DCE_data/20250827-110742-Gross_MeyerA/RAVE_files/config_subject.csv"
PAR_JSON_ROOT = "/home/naiqianluan/DCE-MRI/data/DCE_data/20250827-110742-Gross_MeyerA/RAVE_files/h5slices_wt_acqT/"
TEMPLATE_DCM_PATH = "/home/naiqianluan/DCE-MRI/data/DCE_data/20250827-110742-Gross_MeyerA/example_template.dcm"
BASIS_CONFIGS = [
    ("basis8", 8),
]
LAMBDA_VALUES = [1e-3]
RECON_INPUT_ROOT = REPO_ROOT / "lambda_test_outputs"
DICOM_OUTPUT_ROOT = REPO_ROOT / "dicom_exports"
TARGET_MAX = 533
DATASET_NAME = "temptv"
THICKNESS_MM = 2.0
MINUTES_PER_288_SPOKES = 2.5


def lambda_to_output_label(lambda_value: float) -> str:
    return f"{lambda_value:.0e}".replace("-", "m")


def load_par_json(par_json_path: str | Path) -> dict[str, Any]:
    with open(par_json_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {par_json_path}")
    return payload


def list_reconstructed_slice_files(recon_input_dir: str | Path, hop_id: str) -> list[str]:
    root = Path(recon_input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Reconstruction input directory not found: {root}")

    pattern = f"{hop_id}_slice_*_*.h5"
    files = list(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No reconstructed slice files found in {root} for hop_id={hop_id}")

    def sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"_slice_(\d+)_", path.name)
        if match is None:
            raise ValueError(f"Could not infer slice index from {path}")
        return int(match.group(1)), path.name

    return [str(path) for path in sorted(files, key=sort_key)]


def infer_slice_idx(slice_file: str, dataset_name: str = DATASET_NAME) -> int:
    with h5py.File(slice_file, "r") as h5_file:
        if dataset_name not in h5_file:
            raise KeyError(f"'{dataset_name}' dataset not found in {slice_file}")
        attr_slice = h5_file[dataset_name].attrs.get("slice")

    if attr_slice is not None:
        try:
            slice_idx = int(attr_slice)
            if slice_idx >= 0:
                return slice_idx
        except (TypeError, ValueError):
            pass

    match = re.search(r"_slice_(\d+)_", Path(slice_file).name)
    if match is None:
        raise ValueError(f"Could not infer slice index from {slice_file}")
    return int(match.group(1))


def load_reconstructed_series(slice_file: str, dataset_name: str = DATASET_NAME) -> np.ndarray:
    with h5py.File(slice_file, "r") as h5_file:
        if dataset_name not in h5_file:
            raise KeyError(f"'{dataset_name}' dataset not found in {slice_file}")
        data = h5_file[dataset_name][:]

    series = np.squeeze(np.abs(data))
    series = np.rot90(series, k=2, axes=(-2, -1))
    if series.ndim != 3:
        raise ValueError(f"Expected {dataset_name} to have shape (T, H, W), got {series.shape} in {slice_file}")
    return series.astype(np.float32, copy=False)


def compute_global_max(slice_files: list[str], dataset_name: str = DATASET_NAME) -> float:
    global_max = 0.0
    for slice_file in slice_files:
        series = load_reconstructed_series(slice_file, dataset_name=dataset_name)
        global_max = max(global_max, float(np.max(series)))

    if global_max <= 0:
        raise ValueError("global_max <= 0, check input data (all zeros?)")

    return global_max


def apply_par_to_dataset(ds, par: dict[str, Any], te_ms: float | None = None, thickness_mm: float = THICKNESS_MM):
    if "TR" in par and par["TR"] is not None:
        ds.RepetitionTime = float(par["TR"]) * 1000.0

    if te_ms is None and isinstance(par.get("TE"), (list, tuple)) and par["TE"]:
        te_ms = float(par["TE"][0])
    if te_ms is not None:
        ds.EchoTime = float(te_ms)

    if "flipangle" in par and par["flipangle"] is not None:
        ds.FlipAngle = float(par["flipangle"])

    if "fieldStrength" in par and par["fieldStrength"] is not None:
        ds.MagneticFieldStrength = float(par["fieldStrength"])

    if "bandwidth" in par and par["bandwidth"] is not None:
        ds.PixelBandwidth = float(par["bandwidth"])

    imsize = par.get("imsize")
    if isinstance(imsize, (list, tuple)) and len(imsize) == 2:
        ny, nx = int(imsize[0]), int(imsize[1])
        ds.Rows = ny
        ds.Columns = nx

        fov = par.get("FOV")
        if isinstance(fov, (list, tuple)) and len(fov) == 2:
            fovy, fovx = float(fov[0]), float(fov[1])
            ds.PixelSpacing = [fovy / ny, fovx / nx]
            ds.FOV = [fovy, fovx]

    ds.SliceThickness = float(thickness_mm)
    ds.SpacingBetweenSlices = float(thickness_mm)
    return ds


def compute_slice_geometry(ds, slice_idx: int, n_slices: int) -> tuple[float, list[float]]:
    slice_spacing_mm = float(ds.SpacingBetweenSlices)
    z0 = -(n_slices // 2) * slice_spacing_mm
    slice_loc = z0 + slice_idx * slice_spacing_mm

    if not hasattr(ds, "FOV"):
        raise ValueError("Template dataset is missing derived FOV after par.json application.")

    fovy, fovx = ds.FOV
    ipp_x = -float(fovx) / 2.0
    ipp_y = -float(fovy) / 2.0
    ipp_z = float(slice_loc)
    return float(slice_loc), [ipp_x, ipp_y, ipp_z]


def prepare_output_dataset(
    template_ds,
    par: dict[str, Any],
    hop_id: str,
    slice_idx: int,
    frame_idx: int,
    n_slices: int,
    spokes_per_frame: int,
    pixel_array: np.ndarray,
    start_dt: datetime,
) -> pydicom.Dataset:
    ds = apply_par_to_dataset(template_ds.copy(), par, thickness_mm=THICKNESS_MM)

    frame_delay = timedelta(minutes=frame_idx * MINUTES_PER_288_SPOKES * spokes_per_frame / 288.0)
    dt_delay = start_dt + frame_delay
    slice_loc, image_position = compute_slice_geometry(ds, slice_idx=slice_idx, n_slices=n_slices)

    ds.ContentDate = start_dt.strftime("%Y%m%d")
    ds.ContentTime = dt_delay.strftime("%H%M%S.%f")
    ds.AcquisitionTime = dt_delay.strftime("%H%M%S.%f")
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.ImagePositionPatient = image_position
    ds.SliceLocation = float(slice_loc)

    ds.Rows = int(pixel_array.shape[0])
    ds.Columns = int(pixel_array.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelData = pixel_array.astype(np.uint16, copy=False).tobytes()
    ds["PixelData"].VR = "OW"

    ds[0x00100020].value = hop_id
    ds[0x00201041].value = float(slice_loc)
    ds[0x00200010].value = "1"
    ds[0x00080018].value = str(frame_idx * n_slices + (slice_idx + 1))
    ds.InstanceNumber = int(frame_idx * n_slices + (slice_idx + 1))
    return ds


def export_slice_series(
    slice_file: str,
    hop_id: str,
    spokes_per_frame: int,
    template_ds,
    par: dict[str, Any],
    global_max: float,
    n_slices: int,
    output_dir: Path,
    target_max: int = TARGET_MAX,
    dataset_name: str = DATASET_NAME,
) -> int:
    slice_idx = infer_slice_idx(slice_file, dataset_name=dataset_name)
    series = load_reconstructed_series(slice_file, dataset_name=dataset_name)

    scale = float(target_max) / float(global_max)
    scaled_series = series * scale
    start_dt = datetime.now()

    print(f"    exporting slice {slice_idx:03d} | {os.path.basename(slice_file)}")
    print(f"    data shape: {series.shape}")
    print(f"    scaled min/max: {np.amin(scaled_series):.4f} / {np.amax(scaled_series):.4f}")

    written_files = 0
    for frame_idx in range(scaled_series.shape[0]):
        ds = prepare_output_dataset(
            template_ds=template_ds,
            par=par,
            hop_id=hop_id,
            slice_idx=slice_idx,
            frame_idx=frame_idx,
            n_slices=n_slices,
            spokes_per_frame=spokes_per_frame,
            pixel_array=scaled_series[frame_idx],
            start_dt=start_dt,
        )
        out_path = output_dir / f"{hop_id}_slice{slice_idx:03d}_frame{frame_idx:03d}.dcm"
        ds.save_as(str(out_path))
        written_files += 1

    return written_files


def main() -> int:
    failures: list[tuple[str, str, str]] = []

    print("> patient ", SUBJECT_ID)
    print(f"Reading configurations from: {CSV_PATH}")
    configs = read_csv_config(CSV_PATH)
    print(f"Found {len(configs)} configurations to process")
    print()

    for basis_name, _nbasis in BASIS_CONFIGS:
        for lambda_value in LAMBDA_VALUES:
            lambda_label = f"{lambda_value:.3g}"
            lambda_str = lambda_to_output_label(lambda_value)
            recon_input_dir = RECON_INPUT_ROOT / basis_name / f"recon_h5_lambda_{lambda_str}" / SUBJECT_ID
            dicom_output_base = DICOM_OUTPUT_ROOT / basis_name / f"dicom_lambda_{lambda_str}" / SUBJECT_ID

            print("=" * 60)
            print(f"Exporting {basis_name} | lambda={lambda_label}")
            print(f"  recon input dir: {recon_input_dir}")
            print(f"  DICOM output root: {dicom_output_base}")
            print("=" * 60)

            for config in configs:
                hop_id = config["hop_id"]
                spokes_per_frame = config["spokes_per_frame"]
                par_json_path = os.path.join(PAR_JSON_ROOT, hop_id, "par.json")
                output_dir = dicom_output_base / hop_id

                print()
                print(f">>> {hop_id}")
                print(f"    spokes_per_frame: {spokes_per_frame}")

                try:
                    slice_files = list_reconstructed_slice_files(recon_input_dir, hop_id)
                    slice_indices = [infer_slice_idx(path) for path in slice_files]
                    if sorted(slice_indices) != slice_indices:
                        raise ValueError(f"Slice file ordering is not monotonic for {hop_id}: {slice_indices}")

                    par = load_par_json(par_json_path)
                    template_ds = pydicom.dcmread(TEMPLATE_DCM_PATH)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    global_max = compute_global_max(slice_files)
                    print(f"    global_max: {global_max:.4f}")
                    print(f"    output dir: {output_dir}")

                    written_files = 0
                    for slice_file in slice_files:
                        written_files += export_slice_series(
                            slice_file=slice_file,
                            hop_id=hop_id,
                            spokes_per_frame=spokes_per_frame,
                            template_ds=template_ds,
                            par=par,
                            global_max=global_max,
                            n_slices=len(slice_files),
                            output_dir=output_dir,
                        )

                    print(f"    wrote {written_files} DICOM files")
                except Exception as exc:  # noqa: BLE001
                    failures.append((hop_id, str(recon_input_dir), str(exc)))
                    print(f"    FAILED: {exc}")

    print()
    print("=" * 60)
    if failures:
        print(f"Finished with {len(failures)} failed hop(s):")
        for hop_id, source, error in failures:
            print(f"  {hop_id} | {source} | {error}")
        return 1

    print("Finished successfully with no failed hops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
