"""Shared reconstruction and IO utilities."""

from __future__ import annotations

import csv
import glob
import logging
import os
import re
from pathlib import Path

import h5py
import numpy as np
import sigpy as sp

from .config import CoilCalibrationConfig
from .espirit import EspiritCalib

LOGGER = logging.getLogger(__name__)


def _log(verbose: bool, message: str, *args) -> None:
    if verbose:
        LOGGER.info(message, *args)


def read_csv_config(csv_path):
    configs = []
    with open(csv_path, "r", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            if not any(row.values()):
                continue
            name = row["Name"].strip().strip("'").strip('"')
            configs.append(
                {
                    "hop_id": name,
                    "spokes_per_frame": int(row["spokes_per_frame"]),
                    "slice_idx": int(row["slice_idx"]),
                    "slice_inc": int(row["slice_inc"]),
                    "images_per_slab": int(row["SlicesPerSlab"]),
                }
            )
    return configs


def list_slice_files(hop_dir):
    slice_files = glob.glob(os.path.join(hop_dir, "slice*.h5"))
    if not slice_files:
        raise FileNotFoundError(f"No slice files found in {hop_dir}")
    return sorted(slice_files, key=lambda path: int(re.search(r"slice(\d+)\.h5", path).group(1)))


def infer_kspace_dims(slice_file):
    with h5py.File(slice_file, "r") as h5_file:
        ksp = h5_file["kspace"][:]
    if ksp.ndim != 4:
        raise ValueError(f"Unexpected kspace shape in {slice_file}: {ksp.shape}")
    return tuple(int(dim) for dim in ksp.shape)


def load_slice_kspace_for_coil(slice_file, verbose=False):
    with h5py.File(slice_file, "r") as h5_file:
        ksp = h5_file["kspace"][:]

    ksp_ri = np.transpose(ksp, (3, 2, 1, 0))
    _log(verbose, "loaded raw k-space %s -> %s", ksp.shape, ksp_ri.shape)
    return ksp_ri


def ri_to_coil_spokes_samples(ksp_ri):
    kc = (ksp_ri[0] + 1j * ksp_ri[1]).astype(np.complex64)
    return np.transpose(kc, (2, 0, 1))


def ramp_dcf_from_traj(coord, eps=1e-8, normalize=True):
    if coord.ndim == 4:
        traj2 = coord.reshape(-1, coord.shape[-2], 2)
    elif coord.ndim == 3:
        traj2 = coord
    else:
        raise ValueError(f"Unexpected traj shape {coord.shape}")

    weights = np.sqrt(traj2[..., 0] ** 2 + traj2[..., 1] ** 2)
    if normalize:
        weights = weights / (np.max(weights) + eps)
    return weights.astype(np.float32)


def apply_dcf(ksp, weights):
    if ksp.ndim == 3:
        return ksp * weights[None, :, :]
    if ksp.ndim == 2:
        return ksp * weights
    raise ValueError(f"Unexpected ksp shape {ksp.shape}")


def save_slice_h5(out_path, acq_slice, hop_id, spokes_per_frame, N_time, slice_idx, smax):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5_file:
        dset = h5_file.create_dataset("temptv", data=acq_slice)
        dset.attrs["hop_id"] = hop_id
        dset.attrs["spokes_per_frame"] = spokes_per_frame
        dset.attrs["number_of_frames"] = N_time
        dset.attrs["slice"] = slice_idx
        dset.attrs["max"] = float(smax)


def save_pca_basis_h5(out_path, basis, nbasis=None, dtype=np.float32):
    basis = np.asarray(basis)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got shape {basis.shape}")

    tdim, kdim = basis.shape
    if nbasis is None:
        nbasis = min(tdim, kdim)

    if tdim >= kdim:
        basis_tk = basis[:, :nbasis]
        basis_kt = basis_tk.T
    else:
        basis_kt = basis[:nbasis, :]
        basis_tk = basis_kt.T

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5_file:
        dset = h5_file.create_dataset("bases", data=basis_kt.astype(dtype))
        dset.attrs["num_frames"] = int(basis_tk.shape[0])
        dset.attrs["num_basis"] = int(basis_tk.shape[1])


def load_pca_basis_h5(h5_path, nbasis=None, dtype=np.float32):
    with h5py.File(h5_path, "r") as h5_file:
        basis_kt = np.asarray(h5_file["bases"][:])

    if basis_kt.ndim != 2:
        raise ValueError(f"'bases' dataset must be 2D, got {basis_kt.shape}.")
    if nbasis is not None:
        basis_kt = basis_kt[:nbasis]
    return basis_kt.T.astype(dtype, copy=False)


def make_basis_option(basis, cbasis=False, add_constant=True, dtype=np.complex64):
    basis = np.asarray(basis)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got shape {basis.shape}.")

    tdim, kdim = basis.shape
    basis = basis.astype(np.float32, copy=False)

    if not cbasis:
        if add_constant:
            out = np.ones((tdim, kdim + 1), dtype=np.float32)
            out[:, :kdim] = basis
        else:
            out = basis.copy()
        out_dtype = dtype if np.issubdtype(dtype, np.complexfloating) else np.float32
        return out.astype(out_dtype)

    ncols = 2 * kdim + (1 if add_constant else 0)
    breal = np.zeros((tdim, ncols), dtype=np.float32)
    bimag = np.zeros((tdim, ncols), dtype=np.float32)
    breal[:, :kdim] = basis
    if add_constant:
        breal[:, kdim] = 1.0
        bimag[:, kdim + 1 : kdim + 1 + kdim] = basis
    else:
        bimag[:, kdim : kdim + kdim] = basis
    return (breal + 1j * bimag).astype(dtype)


def load_basis_option_from_h5(h5_path, nbasis=None, cbasis=False, add_constant=True):
    basis = load_pca_basis_h5(h5_path, nbasis=nbasis)
    return make_basis_option(
        basis,
        cbasis=cbasis,
        add_constant=add_constant,
        dtype=np.complex64 if cbasis else np.float32,
    )


def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):
    n_tot_spokes = N_spokes * N_time
    n_samples = base_res * 2
    base_lin = np.arange(n_samples).reshape(1, -1) - (n_samples - 1) / 2
    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)
    base_rot = np.arange(n_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((n_tot_spokes, n_samples, 2), dtype=np.float32)
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin
    traj /= 2
    traj = traj.reshape(N_time, N_spokes, n_samples, 2)
    return np.squeeze(traj)


def estimate_pca_basis(img_dyn, mask=None, K=5, remove_mean=True):
    x = np.abs(img_dyn)
    tdim = x.shape[0]

    if mask is None:
        matrix = x.reshape(tdim, -1).T
    else:
        matrix = x[:, np.asarray(mask, dtype=bool)].T

    if remove_mean:
        matrix = matrix - matrix.mean(axis=1, keepdims=True)

    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    basis = vh[:K].T
    return basis, singular_values


def _coerce_device(device):
    return sp.Device(device)


def _estimate_coil_maps(ksp_c_for_coil, device=sp.Device(-1), config: CoilCalibrationConfig | None = None):
    config = config or CoilCalibrationConfig()
    if ksp_c_for_coil.ndim != 3:
        raise ValueError(
            f"ksp_c_for_coil must be 3D (Nc, Nspokes, Nsamples), got {ksp_c_for_coil.shape}"
        )

    num_coils, num_spokes, num_samples = ksp_c_for_coil.shape
    base_res = num_samples // 2
    ishape = (num_coils, base_res, base_res)
    traj = get_traj(N_spokes=num_spokes, N_time=1, base_res=base_res, gind=1)

    ksp_used = np.asarray(ksp_c_for_coil, dtype=np.complex64)
    if config.use_dcf:
        dcf = ramp_dcf_from_traj(traj, normalize=True)
        ksp_used = apply_dcf(ksp_used, dcf)

    nufft = sp.linop.NUFFT(ishape, traj)
    coil_imgs = np.asarray(sp.to_device(nufft.H(ksp_used), sp.cpu_device), dtype=np.complex64)

    rss = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0))
    rss_max = float(rss.max())
    mask = rss > (config.mask_floor * rss_max)
    denom = np.maximum(rss, 1e-8)
    mps_rss = (coil_imgs / denom[None, :, :]) * mask[None, :, :]
    mps_rss = mps_rss[None, ...]

    _log(config.verbose, "coil estimation shape=%s use_espirit=%s", coil_imgs.shape, config.use_espirit)

    if not config.use_espirit:
        return sp.to_device(mps_rss, _coerce_device(device))

    ksp_cart = sp.to_device(sp.fft(coil_imgs, axes=(-2, -1)), _coerce_device(device))
    calib = EspiritCalib(
        ksp_cart,
        calib_width=config.calib_width,
        thresh=config.thresh,
        crop=config.crop,
        device=device,
        output_eigenvalue=True,
        show_pbar=False,
        verbose=config.verbose,
    )
    mps, _ = calib.run()
    mps = np.asarray(sp.to_device(mps, sp.cpu_device), dtype=np.complex64)
    return sp.to_device(mps, _coerce_device(device))


def _estimate_segmented_pca_bases(img_dyn, vascular_mask, tissue_mask, K=5, remove_mean=True):
    vascular_mask = np.asarray(vascular_mask, dtype=bool)
    tissue_mask = np.asarray(tissue_mask, dtype=bool)

    if not np.any(vascular_mask):
        raise ValueError("vascular_mask is empty.")
    if not np.any(tissue_mask):
        raise ValueError("tissue_mask is empty.")

    vascular_basis, vascular_s = estimate_pca_basis(img_dyn, mask=vascular_mask, K=K, remove_mean=remove_mean)
    tissue_basis, tissue_s = estimate_pca_basis(img_dyn, mask=tissue_mask, K=K, remove_mean=remove_mean)

    return {
        "vascular_basis": vascular_basis,
        "vascular_singular_values": vascular_s,
        "tissue_basis": tissue_basis,
        "tissue_singular_values": tissue_s,
    }
