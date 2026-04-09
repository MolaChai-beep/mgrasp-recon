# import function
import argparse
import inspect
import h5py
import os
import pathlib

import numpy as np
import csv

import re
import glob
import matplotlib.pyplot as plt

from ._bootstrap import ensure_repo_paths

ensure_repo_paths()

import sigpy as sp

import importlib
from .espirit import EspiritCalib

# import sys
# sys.path.insert(0, '/home/allisonchen/sigpy')

import torch
import cupy as cp

def read_csv_config(csv_path):
    """Read CSV file and return list of configuration dictionaries"""
    configs = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig removes BOM
        reader = csv.DictReader(f)
        
        for row in reader:
            # Skip empty rows
            if not any(row.values()):
                continue
            
            # Clean up the Name field - remove all quotes and whitespace
            name = row['Name'].strip().strip("'").strip('"')
            
            config = {
                'hop_id': name,
                'spokes_per_frame': int(row['spokes_per_frame']),
                'slice_idx': int(row['slice_idx']),
                'slice_inc': int(row['slice_inc']),
                'images_per_slab': int(row['SlicesPerSlab'])
            }
            configs.append(config)
    
    return configs

# new function

def list_slice_files(hop_dir):
    slice_files = glob.glob(os.path.join(hop_dir, 'slice*.h5'))
    if not slice_files:
        raise FileNotFoundError(f"No slice files found in {hop_dir}")

    slice_files = sorted(
        slice_files,
        key=lambda x: int(re.search(r'slice(\d+)\.h5', x).group(1))
    )
    return slice_files


def infer_kspace_dims(slice_file):
    """Read one slice file to infer (N_spokes, N_samples, N_coils)."""
    with h5py.File(slice_file, 'r') as f:
        ksp = f['kspace'][:]  # expected (2, spokes, samples, coils)

    if ksp.ndim != 4:
        raise ValueError(f"Unexpected kspace shape in {slice_file}: {ksp.shape}")

    N_coils,N_samples,N_spokes,num = ksp.shape
    return N_coils, N_samples, N_spokes, num


def load_slice_kspace_for_coil(slice_file):
    """
    Keep the old pipeline format used by your working notebook.

    Raw H5 layout:
        (coils, samples, spokes, 2)

    Return:
        ksp_ri: (2, spokes, samples, coils)
    """
    with h5py.File(slice_file, 'r') as f:
        ksp = f['kspace'][:]  # (coils, samples, spokes, 2)

    print(f"Loaded raw k-space from {slice_file}")
    print(f"  raw shape: {ksp.shape}, dtype: {ksp.dtype}")

    # Old working format: (2, spokes, samples, coils)
    ksp_ri = np.transpose(ksp, (3, 2, 1, 0))
    print(f"  ksp_ri shape: {ksp_ri.shape}")
    return ksp_ri


def ri_to_coil_spokes_samples(ksp_ri):
    """
    ksp_ri: (2, spokes, samples, coils)
    return: (coils, spokes, samples) complex64
    """
    kc = (ksp_ri[0] + 1j * ksp_ri[1]).astype(np.complex64)  # (spokes, samples, coils)
    kc = np.transpose(kc, (2, 0, 1))                        # (coils, spokes, samples)
    return kc   



def ramp_dcf_from_traj(coord, eps=1e-8, normalize=True):
    """
    traj: (N_spokes, N_samples, 2) /(N_time, spf, N_samples, 2)
    return w: (N_spokes, N_samples) /(N_time*spf, N_samples)
    """
    if coord.ndim == 4:
        # (N_time, spf, N_samples, 2) -> (N_spokes_total, N_samples, 2)
        traj2 = coord.reshape(-1, coord.shape[-2], 2)
    elif coord.ndim == 3:
        traj2 = coord
    else:
        raise ValueError(f"Unexpected traj shape {coord.shape}")

    kx = traj2[..., 0]
    ky = traj2[..., 1]
    w = np.sqrt(kx**2 + ky**2)

    if normalize:
        w = w / (np.max(w) + eps)

    return w.astype(np.float32)

def apply_dcf(ksp, w):
    """
    ksp: (N_coils, N_spokes_total, N_samples) or (N_spokes_total, N_samples)
    w:   (N_spokes_total, N_samples)
    """
    if ksp.ndim == 3:
        return ksp * w[None, :, :]
    elif ksp.ndim == 2:
        return ksp * w
    else:
        raise ValueError(f"Unexpected ksp shape {ksp.shape}")
    
def save_slice_h5(out_path, acq_slice, hop_id, spokes_per_frame, N_time, slice_idx, smax):
    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset("temptv", data=acq_slice)
        dset.attrs["hop_id"] = hop_id
        dset.attrs["spokes_per_frame"] = spokes_per_frame
        dset.attrs["number_of_frames"] = N_time
        dset.attrs["slice"] = slice_idx
        dset.attrs["max"] = float(smax)

def save_pca_basis_h5(out_path, basis, nbasis=None, dtype=np.float32):
    """
    Save PCA temporal basis in the H5 format expected by dce_recon_3d.py.

    Input:
        basis: (T, K) or (K, T)

    Stored in H5:
        dataset 'bases' with shape (K, T)
    """
    basis = np.asarray(basis)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got shape {basis.shape}")

    tdim, kdim = basis.shape
    if nbasis is None:
        nbasis = min(tdim, kdim)

    if tdim >= kdim:
        basis_tk = basis[:, :nbasis]   # (T, K)
        basis_kt = basis_tk.T          # (K, T)
    else:
        basis_kt = basis[:nbasis, :]   # already (K, T)
        basis_tk = basis_kt.T

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset("bases", data=basis_kt.astype(dtype))
        dset.attrs["num_frames"] = int(basis_tk.shape[0])
        dset.attrs["num_basis"] = int(basis_tk.shape[1])

    print(f"Saved basis to: {out_path}")
    print(f"in-memory basis shape (T, K): {basis_tk.shape}")
    print(f"stored H5 dataset shape (K, T): {basis_kt.shape}")

def load_pca_basis_h5(h5_path, nbasis=None, dtype=np.float32):
    """
    Load PCA basis from H5 and return it in in-memory layout (T, K).

    The reference recon script reads:
        bases = f['bases'][:].T

    so the stored dataset is expected to be (K, T).
    """
    with h5py.File(h5_path, "r") as f:
        basis_kt = f["bases"][:]

    basis_kt = np.asarray(basis_kt)
    if basis_kt.ndim != 2:
        raise ValueError(f"'bases' dataset must be 2D, got {basis_kt.shape}.")

    if nbasis is not None:
        basis_kt = basis_kt[:nbasis]

    basis_tk = basis_kt.T.astype(dtype, copy=False)
    print(
        f"Loaded PCA basis from {h5_path} | stored (K,T)={basis_kt.shape} | "
        f"in-memory (T,K)={basis_tk.shape}"
    )
    return basis_tk

def make_basis_option(basis, cbasis=False, add_constant=True, dtype=np.complex64):
    """
    Build basisoption in the same spirit as dce_recon_3d.py.

    Parameters
    ----------
    basis : ndarray
        Real-valued PCA basis of shape (T, K).
    cbasis : bool
        If False, return real basis plus optional constant column.
        If True, build the complex-expanded basis used in the reference code.
    add_constant : bool
        Append a constant temporal basis.
    dtype : numpy dtype
        Output dtype.

    Returns
    -------
    basisoption : ndarray
        Shape:
        - (T, K + 1) when cbasis=False and add_constant=True
        - (T, K) when cbasis=False and add_constant=False
        - (T, 2K + 1) when cbasis=True and add_constant=True
    """
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
        return out.astype(dtype if np.issubdtype(dtype, np.complexfloating) else np.float32)

    ncols = 2 * kdim + (1 if add_constant else 0)
    breal = np.zeros((tdim, ncols), dtype=np.float32)
    bimag = np.zeros((tdim, ncols), dtype=np.float32)
    breal[:, :kdim] = basis
    if add_constant:
        breal[:, kdim] = 1.0
        bimag[:, kdim + 1:kdim + 1 + kdim] = basis
    else:
        bimag[:, kdim:kdim + kdim] = basis

    return (breal + 1j * bimag).astype(dtype)


def load_basis_option_from_h5(h5_path, nbasis=None, cbasis=False, add_constant=True):
    """
    Convenience helper for step 2:
        load fbasis.h5 -> convert to basisoption used by recon.
    """
    basis = load_pca_basis_h5(h5_path, nbasis=nbasis)
    basisoption = make_basis_option(
        basis,
        cbasis=cbasis,
        add_constant=add_constant,
        dtype=np.complex64 if cbasis else np.float32,
    )
    print(f"basisoption shape: {basisoption.shape}")
    return basisoption




def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):

    N_tot_spokes = N_spokes * N_time

    N_samples = base_res * 2

    # base_lin = np.arange(N_samples).reshape(1, -1) - base_res
    base_lin = np.arange(N_samples).reshape(1, -1) - (N_samples-1)/2


    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    return np.squeeze(traj)

def estimate_pca_basis(img_dyn, mask=None, K=5, remove_mean=True):
    """
    img_dyn: (T, H, W)
    mask:    (H, W) bool, optional
    returns:
        basis: (T, K)
        S:     singular values
    """
    T = img_dyn.shape[0]

    X = np.abs(img_dyn)

    if mask is None:
        M = X.reshape(T, -1).T   # (Nvox, T)
    else:
        M = X[:, mask].T         # (Nvox_sel, T)

    if remove_mean:
        M = M - M.mean(axis=1, keepdims=True)

    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    basis = Vh[:K].T   # (T, K)

    return basis, S


# def get_coil(ksp_c_for_coil, device=sp.Device(-1),thresh = 0.01):
#     if ksp_c_for_coil.ndim != 3:
#         raise ValueError(
#             f"ksp_c_for_coil must be 3D (Nc, Nspokes, Nsamples), got {ksp_c_for_coil.shape}"
#         )

#     N_coils, N_spokes, N_samples = ksp_c_for_coil.shape
#     base_res = N_samples // 2
#     ishape = (N_coils, base_res, base_res)

#     print(f"[get_coil] k-space shape: {ksp_c_for_coil.shape}")
#     print(f"[get_coil] base_res: {base_res}")
#     print(f"[get_coil] N_coils: {N_coils}")

#     traj = get_traj(
#     N_spokes=N_spokes,
#     N_time=1,
#     base_res=base_res,
#     gind=1
#     )
#     print(f"Trajectory range in get_coil: [{traj.min():.3f}, {traj.max():.3f}]")
#     print(f"Trajectory shape: {traj.shape}")

#     F = sp.linop.NUFFT([N_coils, base_res, base_res], traj)
#     cim = F.H(ksp_c_for_coil)   # no DCF

#     cim_cpu = np.asarray(sp.to_device(cim, sp.cpu_device))

#     plt.figure(figsize=(12,6))
#     for i in range(8):
#         img = np.abs(cim_cpu[i])
#         plt.subplot(2,4,i+1)
#         plt.imshow(img, cmap='gray',
#                     vmin=np.percentile(img, 1),
#                     vmax=np.percentile(img, 99.5))
#         plt.title(f'coil {i}')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

#     calib = EspiritCalib(
#         cim,
#         device=device,
#         thresh=thresh,
#         crop=0.95, output_eigenvalue=True,
#         show_pbar=False,
#     )

#     mps, eig = calib.run()  

#     return sp.to_device(mps)

def get_coil(
    ksp_c_for_coil,
    device=sp.Device(-1),
    thresh=0.01,
    use_dcf=True,
    mask_floor=0.05,
    calib_width=24,
    crop=0.95,
    use_espirit=True,
    verbose=True,
):
    if ksp_c_for_coil.ndim != 3:
        raise ValueError(
            f"ksp_c_for_coil must be 3D (Nc, Nspokes, Nsamples), got {ksp_c_for_coil.shape}"
        )

    N_coils, N_spokes, N_samples = ksp_c_for_coil.shape
    base_res = N_samples // 2
    ishape = (N_coils, base_res, base_res)

    if verbose:
        print(f"[get_coil] k-space shape: {ksp_c_for_coil.shape}")
        print(f"[get_coil] base_res: {base_res}")
        print(f"[get_coil] N_coils: {N_coils}")

    traj = get_traj(
        N_spokes=N_spokes,
        N_time=1,
        base_res=base_res,
        gind=1
    )
    if verbose:
        print(f"[get_coil] trajectory range: [{traj.min():.3f}, {traj.max():.3f}]")
        print(f"[get_coil] trajectory shape: {traj.shape}")

    ksp_used = np.asarray(ksp_c_for_coil, dtype=np.complex64)
    if use_dcf:
        dcf = ramp_dcf_from_traj(traj, normalize=True)
        ksp_used = apply_dcf(ksp_used, dcf)
        if verbose:
            print(
                f"[get_coil] using ramp DCF, range: "
                f"[{float(dcf.min()):.3f}, {float(dcf.max()):.3f}]"
            )

    F = sp.linop.NUFFT(ishape, traj)
    coil_imgs = F.H(ksp_used)
    coil_imgs = sp.to_device(coil_imgs, sp.cpu_device)
    coil_imgs = np.asarray(coil_imgs, dtype=np.complex64)

    rss = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0))
    rss_max = float(rss.max())
    mask = rss > (mask_floor * rss_max)
    denom = np.maximum(rss, 1e-8)
    mps_rss = coil_imgs / denom[None, :, :]
    mps_rss *= mask[None, :, :]
    mps_rss = mps_rss[None, ...]

    if use_espirit:
        # Grid radial data to image domain, then FFT back to Cartesian k-space so
        # ESPIRiT receives calibration k-space rather than coil images.
        ksp_cart = sp.fft(coil_imgs, axes=(-2, -1))
        ksp_cart = sp.to_device(ksp_cart, device)

        if verbose:
            ksp_cart_cpu = np.asarray(sp.to_device(ksp_cart, sp.cpu_device))
            kc_abs = np.abs(ksp_cart_cpu)
            print(f"[get_coil] gridded Cartesian k-space shape: {ksp_cart_cpu.shape}")
            print(
                f"[get_coil] ksp_cart abs range: "
                f"[{float(kc_abs.min()):.3e}, {float(kc_abs.max()):.3e}]"
            )

        calib = EspiritCalib(
            ksp_cart,
            calib_width=calib_width,
            thresh=thresh,
            crop=crop,
            device=device,
            output_eigenvalue=True,
            show_pbar=False,
        )
        mps, eig = calib.run()
        mps = sp.to_device(mps, sp.cpu_device)
        mps = np.asarray(mps, dtype=np.complex64)

        if verbose:
            print(f"[get_coil] ESPIRiT mps shape: {mps.shape}")
            plt.figure(figsize=(5, 5))
            plt.imshow(np.abs(mps[0, 0]), cmap="gray")
            plt.title("abs(ESPIRiT mps[coil 0])")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return sp.to_device(mps, device)

    if verbose:
        print(f"[get_coil] coil_imgs shape: {coil_imgs.shape}")
        print(f"[get_coil] rss range: [{float(rss.min()):.3e}, {float(rss.max()):.3e}]")
        print(f"[get_coil] mps shape: {mps_rss.shape}")

        plt.figure(figsize=(12, 6))
        for i in range(min(8, N_coils)):
            img = np.abs(coil_imgs[i])
            plt.subplot(2, 4, i + 1)
            plt.imshow(
                img,
                cmap="gray",
                vmin=np.percentile(img, 1),
                vmax=np.percentile(img, 99.5),
            )
            plt.title(f"coil {i}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(rss, cmap="gray")
        plt.title("RSS composite")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(np.abs(mps_rss[0, 0]), cmap="gray")
        plt.title("abs(RSS mps[coil 0])")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return sp.to_device(mps_rss, device)

def estimate_segmented_pca_bases(
    img_dyn,
    vascular_mask,
    tissue_mask,
    K=5,
    remove_mean=True,
):
    """
    Estimate separate PCA bases for vascular and tissue regions.

    Parameters
    ----------
    img_dyn : ndarray
        Dynamic image series with shape (T, H, W).
    vascular_mask : ndarray
        Boolean mask with shape (H, W).
    tissue_mask : ndarray
        Boolean mask with shape (H, W).
    K : int
        Number of PCA components.
    remove_mean : bool
        Whether to subtract each voxel's temporal mean before SVD.

    Returns
    -------
    dict
        {
            "vascular_basis": (T, K),
            "vascular_singular_values": (...,),
            "tissue_basis": (T, K),
            "tissue_singular_values": (...,),
        }
    """
    vascular_mask = np.asarray(vascular_mask, dtype=bool)
    tissue_mask = np.asarray(tissue_mask, dtype=bool)

    if not np.any(vascular_mask):
        raise ValueError("vascular_mask is empty.")
    if not np.any(tissue_mask):
        raise ValueError("tissue_mask is empty.")

    vascular_basis, vascular_s = estimate_pca_basis(
        img_dyn,
        mask=vascular_mask,
        K=K,
        remove_mean=remove_mean,
    )

    tissue_basis, tissue_s = estimate_pca_basis(
        img_dyn,
        mask=tissue_mask,
        K=K,
        remove_mean=remove_mean,
    )

    return {
        "vascular_basis": vascular_basis,
        "vascular_singular_values": vascular_s,
        "tissue_basis": tissue_basis,
        "tissue_singular_values": tissue_s,
    }



# def get_coil(ksp_c_for_coil, device=sp.Device(-1),thresh = 0.01):
#     if ksp_c_for_coil.ndim != 3:
#         raise ValueError(
#             f"ksp_c_for_coil must be 3D (Nc, Nspokes, Nsamples), got {ksp_c_for_coil.shape}"
#         )

#     N_coils, N_spokes, N_samples = ksp_c_for_coil.shape
#     base_res = N_samples // 2
#     ishape = (N_coils, base_res, base_res)

#     print(f"[get_coil] k-space shape: {ksp_c_for_coil.shape}")
#     print(f"[get_coil] base_res: {base_res}")
#     print(f"[get_coil] N_coils: {N_coils}")

#     traj = get_traj(
#     N_spokes=N_spokes,
#     N_time=1,
#     base_res=base_res,
#     gind=1
#     )
#     print(f"Trajectory range in get_coil: [{traj.min():.3f}, {traj.max():.3f}]")
#     print(f"Trajectory shape: {traj.shape}")

#     F = sp.linop.NUFFT([N_coils, base_res, base_res], traj)
#     cim = F.H(ksp_c_for_coil)   # no DCF

#     cim_cpu = np.asarray(sp.to_device(cim, sp.cpu_device))

#     plt.figure(figsize=(12,6))
#     for i in range(8):
#         img = np.abs(cim_cpu[i])
#         plt.subplot(2,4,i+1)
#         plt.imshow(img, cmap='gray',
#                     vmin=np.percentile(img, 1),
#                     vmax=np.percentile(img, 99.5))
#         plt.title(f'coil {i}')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

#     calib = EspiritCalib(
#         cim,
#         device=device,
#         thresh=thresh,
#         crop=0.95, output_eigenvalue=True,
#         show_pbar=False,
#     )

#     mps, eig = calib.run()  

#     return sp.to_device(mps)



