"""Batch Step 3 DICOM export for all reconstructed slices of one patient.

This script follows the Step 2 batch layout and exports the reconstructed
``temptv`` series to DICOM using one template scan plus ``par.json`` metadata.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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


def lambda_to_output_label(lambda_value: float) -> str:
    return f"{lambda_value:.0e}".replace("-", "m")


@dataclass(frozen=True)
class DicomExportConfig:
    template_dcm_path: str | Path
    par_json_path: str | Path
    output_dir: str | Path
    hop_id: str
    lambda_value: float
    series_description: str = "MGRASP Step3 temptv"
    series_number_offset: int = 300


@dataclass(frozen=True)
class DicomExportResult:
    written_files: int
    global_max: float
    output_dir: Path


class Step3DicomExporter:
    """Export reconstructed Step 2 H5 slices to a DICOM series."""

    def __init__(self, config: DicomExportConfig):
        self.config = config

    def export(self, slice_files: list[str]) -> DicomExportResult:
        if not slice_files:
            raise ValueError("slice_files must not be empty.")

        template = self._load_template_dataset()
        par_metadata = self._load_par_metadata()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_max = self._compute_global_max(slice_files)
        if global_max <= 0:
            global_max = 1.0

        series_uid = self._generate_uid()
        written_files = 0
        total_instances = self._count_total_instances(slice_files)
        instance_number = 1

        for slice_order, slice_file in enumerate(slice_files):
            img_dyn_abs, slice_idx = self._load_reconstructed_series(slice_file)
            num_frames = int(img_dyn_abs.shape[0])

            for frame_idx in range(num_frames):
                dataset = template.copy()
                pixel_array = self._to_uint16(img_dyn_abs[frame_idx], global_max)
                self._apply_common_metadata(
                    dataset=dataset,
                    pixel_array=pixel_array,
                    slice_idx=slice_idx,
                    slice_order=slice_order,
                    frame_idx=frame_idx,
                    instance_number=instance_number,
                    total_instances=total_instances,
                    series_uid=series_uid,
                    par_metadata=par_metadata,
                )

                out_name = f"{self.config.hop_id}_slice_{slice_idx:03d}_frame_{frame_idx:03d}.dcm"
                out_path = output_dir / out_name
                dataset.save_as(str(out_path), write_like_original=False)
                written_files += 1
                instance_number += 1

        return DicomExportResult(
            written_files=written_files,
            global_max=float(global_max),
            output_dir=output_dir,
        )

    def _load_template_dataset(self):
        try:
            import pydicom
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "pydicom is required for Step 3 DICOM export. Install it in the runtime environment before running this script."
            ) from exc

        return pydicom.dcmread(str(self.config.template_dcm_path))

    def _load_par_metadata(self) -> dict[str, Any]:
        par_path = Path(self.config.par_json_path)
        if not par_path.exists():
            return {}
        with par_path.open("r", encoding="utf-8") as file_obj:
            loaded = json.load(file_obj)
        return loaded if isinstance(loaded, dict) else {}

    def _compute_global_max(self, slice_files: list[str]) -> float:
        global_max = 0.0
        for slice_file in slice_files:
            with h5py.File(slice_file, "r") as h5_file:
                if "temptv" not in h5_file:
                    raise KeyError(f"'temptv' dataset not found in {slice_file}")
                current = float(np.max(np.abs(h5_file["temptv"][:])))
                global_max = max(global_max, current)
        return global_max

    def _count_total_instances(self, slice_files: list[str]) -> int:
        total = 0
        for slice_file in slice_files:
            with h5py.File(slice_file, "r") as h5_file:
                total += int(h5_file["temptv"].shape[0])
        return total

    def _load_reconstructed_series(self, slice_file: str) -> tuple[np.ndarray, int]:
        with h5py.File(slice_file, "r") as h5_file:
            if "temptv" not in h5_file:
                raise KeyError(f"'temptv' dataset not found in {slice_file}")
            img_dyn_abs = np.abs(np.asarray(h5_file["temptv"][:]))
            attr_slice = h5_file["temptv"].attrs.get("slice")

        slice_idx = self._infer_slice_idx(slice_file, attr_slice)
        if img_dyn_abs.ndim != 3:
            raise ValueError(f"Expected temptv to have shape (T, H, W), got {img_dyn_abs.shape} in {slice_file}")
        return img_dyn_abs.astype(np.float32, copy=False), slice_idx

    def _infer_slice_idx(self, slice_file: str, attr_slice: Any) -> int:
        if attr_slice is not None:
            try:
                value = int(attr_slice)
                if value >= 0:
                    return value
            except (TypeError, ValueError):
                pass

        match = re.search(r"_slice_(\d+)_", Path(slice_file).name)
        if match is None:
            raise ValueError(f"Could not infer slice index from {slice_file}")
        return int(match.group(1))

    def _to_uint16(self, frame: np.ndarray, global_max: float) -> np.ndarray:
        scaled = np.clip(np.asarray(frame, dtype=np.float32) / max(global_max, 1e-8), 0.0, 1.0)
        return np.round(scaled * np.iinfo(np.uint16).max).astype(np.uint16)

    def _apply_common_metadata(
        self,
        dataset,
        pixel_array: np.ndarray,
        slice_idx: int,
        slice_order: int,
        frame_idx: int,
        instance_number: int,
        total_instances: int,
        series_uid: str,
        par_metadata: dict[str, Any],
    ) -> None:
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.SOPInstanceUID = generate_uid()
        dataset.SeriesInstanceUID = series_uid
        dataset.InstanceNumber = int(instance_number)
        dataset.AcquisitionNumber = int(frame_idx + 1)
        dataset.ImagesInAcquisition = int(total_instances)
        dataset.Rows = int(pixel_array.shape[0])
        dataset.Columns = int(pixel_array.shape[1])
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.PixelRepresentation = 0
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelData = pixel_array.tobytes()
        dataset.SeriesDescription = f"{self.config.series_description} | hop={self.config.hop_id} | lambda={self.config.lambda_value:.3g}"
        dataset.SeriesNumber = int(getattr(dataset, "SeriesNumber", 0)) + self.config.series_number_offset

        if hasattr(dataset, "SliceLocation"):
            dataset.SliceLocation = float(slice_idx)

        image_position = self._resolve_image_position(dataset, par_metadata, slice_order)
        if image_position is not None:
            dataset.ImagePositionPatient = image_position

        image_orientation = self._resolve_image_orientation(dataset, par_metadata)
        if image_orientation is not None:
            dataset.ImageOrientationPatient = image_orientation

        spacing = self._resolve_spacing(dataset, par_metadata)
        if spacing is not None:
            dataset.PixelSpacing = spacing

        slice_thickness = self._resolve_scalar(par_metadata, ["slice_thickness", "SliceThickness"])
        if slice_thickness is not None:
            dataset.SliceThickness = slice_thickness

    def _resolve_image_position(self, dataset, par_metadata: dict[str, Any], slice_order: int) -> list[float] | None:
        position = self._resolve_sequence(
            par_metadata,
            [
                "image_position_patient",
                "ImagePositionPatient",
                "image_position",
            ],
            expected_len=3,
        )
        if position is not None:
            return position

        if hasattr(dataset, "ImagePositionPatient"):
            base = [float(value) for value in dataset.ImagePositionPatient]
            if len(base) == 3:
                base[2] = base[2] + float(slice_order)
                return base
        return None

    def _resolve_image_orientation(self, dataset, par_metadata: dict[str, Any]) -> list[float] | None:
        orientation = self._resolve_sequence(
            par_metadata,
            [
                "image_orientation_patient",
                "ImageOrientationPatient",
                "image_orientation",
            ],
            expected_len=6,
        )
        if orientation is not None:
            return orientation

        if hasattr(dataset, "ImageOrientationPatient"):
            values = [float(value) for value in dataset.ImageOrientationPatient]
            if len(values) == 6:
                return values
        return None

    def _resolve_spacing(self, dataset, par_metadata: dict[str, Any]) -> list[float] | None:
        spacing = self._resolve_sequence(
            par_metadata,
            [
                "pixel_spacing",
                "PixelSpacing",
                "in_plane_resolution",
            ],
            expected_len=2,
        )
        if spacing is not None:
            return spacing

        if hasattr(dataset, "PixelSpacing"):
            values = [float(value) for value in dataset.PixelSpacing]
            if len(values) == 2:
                return values
        return None

    def _resolve_sequence(self, payload: dict[str, Any], keys: list[str], expected_len: int) -> list[float] | None:
        value = self._resolve_value(payload, keys)
        if isinstance(value, (list, tuple)) and len(value) >= expected_len:
            try:
                return [float(item) for item in value[:expected_len]]
            except (TypeError, ValueError):
                return None
        return None

    def _resolve_scalar(self, payload: dict[str, Any], keys: list[str]) -> float | None:
        value = self._resolve_value(payload, keys)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_value(self, payload: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    def _generate_uid(self) -> str:
        from pydicom.uid import generate_uid

        return generate_uid()


def list_reconstructed_slice_files(recon_input_dir: str | Path, hop_id: str) -> list[str]:
    root = Path(recon_input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Reconstruction input directory not found: {root}")

    pattern = f"{hop_id}_slice_*_*.h5"
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No reconstructed slice files found in {root} for hop_id={hop_id}")

    def sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"_slice_(\d+)_", path.name)
        slice_idx = int(match.group(1)) if match is not None else -1
        return slice_idx, path.name

    return [str(path) for path in sorted(files, key=sort_key)]


def export_reconstructed_slices_to_dicom(
    slice_files: list[str],
    template_dcm_path: str | Path,
    par_json_path: str | Path,
    output_dir: str | Path,
    hop_id: str,
    lambda_value: float,
) -> DicomExportResult:
    exporter = Step3DicomExporter(
        DicomExportConfig(
            template_dcm_path=template_dcm_path,
            par_json_path=par_json_path,
            output_dir=output_dir,
            hop_id=hop_id,
            lambda_value=lambda_value,
        )
    )
    return exporter.export(slice_files)


def main() -> int:
    failures: list[tuple[str, str]] = []

    print("> patient ", SUBJECT_ID)
    print(f"Reading configurations from: {CSV_PATH}")
    configs = read_csv_config(CSV_PATH)
    print(f"Found {len(configs)} configurations to process")
    print()

    for basis_name, _nbasis in BASIS_CONFIGS:
        for lambda_value in LAMBDA_VALUES:
            lambda_str = lambda_to_output_label(lambda_value)
            recon_input_dir = RECON_INPUT_ROOT / basis_name / f"recon_h5_lambda_{lambda_str}" / SUBJECT_ID
            dicom_output_base = DICOM_OUTPUT_ROOT / basis_name / f"dicom_lambda_{lambda_str}" / SUBJECT_ID

            print("=" * 60)
            print(f"Exporting {basis_name} | lambda={lambda_value:.3g}")
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
                    result = export_reconstructed_slices_to_dicom(
                        slice_files=slice_files,
                        template_dcm_path=TEMPLATE_DCM_PATH,
                        par_json_path=par_json_path,
                        output_dir=output_dir,
                        hop_id=hop_id,
                        lambda_value=lambda_value,
                    )
                    print(f"    wrote {result.written_files} DICOM files")
                    print(f"    global_max: {result.global_max:.4f}")
                    print(f"    output dir: {result.output_dir}")
                except Exception as exc:  # noqa: BLE001
                    failures.append((hop_id, str(exc)))
                    print(f"    FAILED: {exc}")

    print()
    if failures:
        print(f"Finished with {len(failures)} failed hop(s):")
        for hop_id, error in failures:
            print(f"  {hop_id} | {error}")
        return 1

    print("Finished successfully with no failed hops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
