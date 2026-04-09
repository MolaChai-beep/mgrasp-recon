import numpy as np

import torch

import pathlib
import sys

from ._bootstrap import ensure_repo_paths

# clear sigpy
for k in list(sys.modules.keys()):
    if k == 'sigpy' or k.startswith('sigpy.'):
        del sys.modules[k]

REPO_ROOT, SRC_ROOT = ensure_repo_paths()

from mgrasp_recon.recon_utils import (
    apply_dcf,
    estimate_pca_basis,
    get_coil,
    get_traj,
    load_basis_option_from_h5,
    ramp_dcf_from_traj,
    save_pca_basis_h5,
    save_slice_h5,
    load_slice_kspace_for_coil,
    ri_to_coil_spokes_samples,
)
from mgrasp_recon.interframe_recon import radial_lowres_pca_recon_2d

import sigpy as sp
from sigpy.mri.app import HighDimensionalRecon

import scipy as scp
# print("scipy.__file__ =", scp.__file__)
# print("scipy.__version__ =", scp.__version__)

from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_opening, binary_closing, binary_erosion
from scipy import ndimage as ndi

import argparse
import h5py
import os

import numpy as np
import csv

import re
import glob
import matplotlib.pyplot as plt
import torch
import cupy as cp


def run_subspace_recon_2d(
    ksp,
    traj,
    mps,
    fbasis_path,
    spokes_per_frame,
    nbasis=5,
    cbasis=False,
    add_constant=True,
    lamda=1e-3,
    regu="TV",
    regu_axes=[-2, -1],
    max_iter=10,
    solver="ADMM",
    use_dcf=True,
    device=None,
    show_pbar=False,
):
    """
    2D PCA/subspace reconstruction following the 3D reference code logic.

    Parameters
    ----------
    ksp : ndarray
        Complex k-space, shape (Nc, Nspokes_total, Ns)
    traj : ndarray
        Trajectory, shape (T, spf, Ns, 2)
    mps : ndarray
        Coil sensitivity maps, shape (1, Nc, Ny, Nx) or (Nc, 1, Ny, Nx)
    fbasis_path : str
        Path to saved PCA basis H5 file.
    spokes_per_frame : int
        Number of spokes per frame.
    nbasis : int
        Number of PCA basis vectors to load.
    cbasis : bool
        Whether to build complex basisoption like reference code.
    add_constant : bool
        Whether to append a constant basis.
    lamda : float
        Regularization weight.
    regu : str
        'TIK', 'TV', 'LLR', etc.
    regu_axes : list
        Regularization axes for HighDimensionalRecon.
    max_iter : int
        Iteration count.
    solver : str
        e.g. 'ADMM'
    use_dcf : bool
        Whether to use radial DCF weights.
    """

    if device is None:
        device = sp.Device(0 if torch.cuda.is_available() else -1)

    ksp = np.asarray(ksp, dtype=np.complex64)
    traj = np.asarray(traj, dtype=np.float32)

    if ksp.ndim != 3:
        raise ValueError(f"ksp must have shape (Nc, Nspokes_total, Ns), got {ksp.shape}")

    if traj.ndim != 4:
        raise ValueError(f"traj must have shape (T, spf, Ns, 2), got {traj.shape}")

    Nc, Nspokes_total, Ns = ksp.shape
    T = Nspokes_total // spokes_per_frame
    Nspokes_used = T * spokes_per_frame

    if traj.shape[0] != T or traj.shape[1] != spokes_per_frame or traj.shape[2] != Ns:
        raise ValueError(
            f"traj shape {traj.shape} is inconsistent with derived "
            f"(T, spf, Ns)=({T}, {spokes_per_frame}, {Ns})"
        )

    # Load basisoption exactly for step 2.
    basisoption = load_basis_option_from_h5(
        fbasis_path,
        nbasis=nbasis,
        cbasis=cbasis,
        add_constant=add_constant,
    )

    # The reference app expects basis.shape[0] == Ntime * Necho.
    # Here Necho = 1, so basisoption.shape[0] must equal T.
    if basisoption.shape[0] != T:
        raise ValueError(
            f"basisoption has {basisoption.shape[0]} time points, but recon uses T={T} frames."
        )

    # Prepare mps to shape (Nc, 1, Ny, Nx) for 2D HighDimensionalRecon.
    mps = np.asarray(mps, dtype=np.complex64)
    if mps.ndim == 4 and mps.shape[0] == 1:
        mps = np.transpose(mps, (1, 0, 2, 3))  # (1, Nc, Ny, Nx) -> (Nc, 1, Ny, Nx)
    elif mps.ndim == 4 and mps.shape[1] == 1:
        pass  # already (Nc, 1, Ny, Nx)
    else:
        raise ValueError(
            f"mps must have shape (1, Nc, Ny, Nx) or (Nc, 1, Ny, Nx), got {mps.shape}"
        )

    # Trim and frame the k-space like the 3D reference script.
    ksp_use = ksp[:, :Nspokes_used, :]                       # (Nc, T*spf, Ns)
    ksp_prep = np.swapaxes(ksp_use, 0, 1)                   # (T*spf, Nc, Ns)
    ksp_prep = ksp_prep.reshape(T, spokes_per_frame, Nc, Ns)
    ksp_prep = np.transpose(ksp_prep, (0, 2, 1, 3))         # (T, Nc, spf, Ns)
    ksp_prep = ksp_prep[:, None, :, None, :, :]             # (T, 1, Nc, 1, spf, Ns)

    # coord for 2D HighDimensionalRecon: (T, spf, Ns, 2)
    coord_prep = traj

    # weights shaped like ksp_prep, similar to the reference code.
    if use_dcf:
        dcf = np.sqrt(coord_prep[..., 0] ** 2 + coord_prep[..., 1] ** 2).astype(np.float32)
        weights = dcf[:, None, None, None, :, :]            # (T, 1, 1, 1, spf, Ns)
        weights = np.tile(weights, (1, 1, Nc, 1, 1, 1))     # (T, 1, Nc, 1, spf, Ns)
    else:
        weights = np.ones_like(ksp_prep, dtype=np.float32)

    print("ksp_prep shape:", ksp_prep.shape)
    print("coord_prep shape:", coord_prep.shape)
    print("weights shape:", weights.shape)
    print("mps shape:", mps.shape)
    print("basisoption shape:", basisoption.shape)

    # Run 2D subspace reconstruction.
    recon = HighDimensionalRecon(
        ksp_prep,
        mps,
        weights=weights,
        coord=coord_prep,
        basis=basisoption,
        lamda=lamda,
        regu=regu,
        regu_axes=regu_axes,
        max_iter=max_iter,
        solver=solver,
        device=device,
        show_pbar=show_pbar,
    ).run()

    recon = sp.to_device(recon, sp.cpu_device)
    recon = np.asarray(recon)

    print("raw recon output shape:", recon.shape)

    # Remove singleton dims. Expected result is roughly (Nbasis_eff, Ny, Nx).
    coeff_maps = np.squeeze(recon)

    if coeff_maps.ndim != 3:
        raise ValueError(
            f"Expected coefficient maps to be 3D after squeeze, got {coeff_maps.shape}"
        )

    # coeff_maps: (Keff, Ny, Nx)
    # basisoption: (T, Keff)
    if coeff_maps.shape[0] != basisoption.shape[1]:
        raise ValueError(
            f"Coefficient count mismatch: coeff_maps.shape[0]={coeff_maps.shape[0]} "
            f"but basisoption.shape[1]={basisoption.shape[1]}"
        )

    # Recover dynamic series: x(t, y, x) = sum_k basis(t, k) * coeff(k, y, x)
    img_dyn = np.einsum("tk,kyx->tyx", basisoption, coeff_maps)
    img_dyn_abs = np.abs(img_dyn)

    return coeff_maps, img_dyn, img_dyn_abs, basisoption




def run_subspace_recon_for_slice(
    slice_file,
    traj,
    fbasis_path,
    spokes_per_frame,
    hop_id=None,
    slice_idx=None,
    nbasis=5,
    cbasis=False,
    add_constant=True,
    lamda=1e-3,
    regu="TV",
    regu_axes=[-2, -1],
    max_iter=10,
    solver="ADMM",
    use_dcf=True,
    coil_thresh=0.02,
    coil_device=-1,
    recon_device=None,
    show_pbar=False,
    save_h5=False,
    out_path=None,
):
    """Run the full step-2 subspace recon pipeline for a single slice file."""
    if slice_idx is None:
        m = re.search(r"slice(\d+)\.h5$", os.path.basename(slice_file))
        if m is not None:
            slice_idx = int(m.group(1)) - 1

    print(f"Loading slice file: {slice_file}")
    ksp_ri = load_slice_kspace_for_coil(slice_file)
    print("ksp_coil shape:", ksp_ri.shape)
    ksp_c_for_coil = ri_to_coil_spokes_samples(ksp_ri)
    print("  k1 shape:", ksp_c_for_coil.shape)

    mps = get_coil(ksp_c_for_coil, device=coil_device, thresh=coil_thresh)
    print('  mps shape:', np.asarray(sp.to_device(mps, sp.cpu_device)).shape)

    coeff_maps, img_dyn_cplx, img_dyn_abs, basisoption = run_subspace_recon_2d(
        ksp=ksp_c_for_coil,
        traj=traj,
        mps=mps,
        fbasis_path=fbasis_path,
        spokes_per_frame=spokes_per_frame,
        nbasis=nbasis,
        cbasis=cbasis,
        add_constant=add_constant,
        lamda=lamda,
        regu=regu,
        regu_axes=regu_axes,
        max_iter=max_iter,
        solver=solver,
        use_dcf=use_dcf,
        device=recon_device,
        show_pbar=show_pbar,
    )

    if save_h5:
        if out_path is None:
            raise ValueError("out_path must be provided when save_h5=True")
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_slice_h5(
            out_path=out_path,
            acq_slice=np.asarray(img_dyn_cplx),
            hop_id=hop_id if hop_id is not None else "",
            spokes_per_frame=spokes_per_frame,
            N_time=img_dyn_cplx.shape[0],
            slice_idx=-1 if slice_idx is None else slice_idx,
            smax=np.max(img_dyn_abs),
        )
        print(f"Saved slice recon to: {out_path}")

    return {
        "coeff_maps": coeff_maps,
        "img_dyn_cplx": img_dyn_cplx,
        "img_dyn_abs": img_dyn_abs,
        "basisoption": basisoption,
        "mps": np.asarray(sp.to_device(mps, sp.cpu_device)),
        "ksp": ksp_c_for_coil,
    }
