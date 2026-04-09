"""2D subspace reconstruction orchestration internals used by workflow classes."""

from __future__ import annotations

import logging
import os
import re

import numpy as np
import sigpy as sp
import torch
from sigpy.mri.app import HighDimensionalRecon

from .config import Recon2DResult, ReconstructionConfig, SliceReconResult, SliceReconstructionConfig
from .recon_utils import _estimate_coil_maps, load_basis_option_from_h5, load_slice_kspace_for_coil, ramp_dcf_from_traj, ri_to_coil_spokes_samples, save_slice_h5

LOGGER = logging.getLogger(__name__)


def _log(verbose: bool, message: str, *args) -> None:
    if verbose:
        LOGGER.info(message, *args)


def _default_device(device):
    if device is not None:
        return device
    return sp.Device(0 if torch.cuda.is_available() else -1)


def _prepare_mps_for_highdim(mps: np.ndarray) -> np.ndarray:
    mps = np.asarray(mps, dtype=np.complex64)
    if mps.ndim == 4 and mps.shape[0] == 1:
        return np.transpose(mps, (1, 0, 2, 3))
    if mps.ndim == 4 and mps.shape[1] == 1:
        return mps
    raise ValueError(f"mps must have shape (1, Nc, Ny, Nx) or (Nc, 1, Ny, Nx), got {mps.shape}")


def _run_subspace_recon_2d(
    ksp,
    traj,
    mps,
    fbasis_path,
    spokes_per_frame,
    config: ReconstructionConfig | None = None,
    device=None,
) -> Recon2DResult:
    config = config or ReconstructionConfig()
    device = _default_device(device)

    ksp = np.asarray(ksp, dtype=np.complex64)
    traj = np.asarray(traj, dtype=np.float32)

    if ksp.ndim != 3:
        raise ValueError(f"ksp must have shape (Nc, Nspokes_total, Ns), got {ksp.shape}")
    if traj.ndim != 4:
        raise ValueError(f"traj must have shape (T, spf, Ns, 2), got {traj.shape}")

    num_coils, nspokes_total, num_samples = ksp.shape
    num_frames = nspokes_total // spokes_per_frame
    nspokes_used = num_frames * spokes_per_frame
    if traj.shape[0] != num_frames or traj.shape[1] != spokes_per_frame or traj.shape[2] != num_samples:
        raise ValueError(
            f"traj shape {traj.shape} is inconsistent with derived (T, spf, Ns)=({num_frames}, {spokes_per_frame}, {num_samples})"
        )

    basisoption = load_basis_option_from_h5(
        fbasis_path,
        nbasis=config.nbasis,
        cbasis=config.cbasis,
        add_constant=config.add_constant,
    )
    if basisoption.shape[0] != num_frames:
        raise ValueError(
            f"basisoption has {basisoption.shape[0]} time points, but recon uses T={num_frames} frames."
        )

    mps = _prepare_mps_for_highdim(mps)
    ksp_use = ksp[:, :nspokes_used, :]
    ksp_prep = np.swapaxes(ksp_use, 0, 1)
    ksp_prep = ksp_prep.reshape(num_frames, spokes_per_frame, num_coils, num_samples)
    ksp_prep = np.transpose(ksp_prep, (0, 2, 1, 3))
    ksp_prep = ksp_prep[:, None, :, None, :, :]

    if config.use_dcf:
        dcf = ramp_dcf_from_traj(traj, normalize=False).reshape(num_frames, spokes_per_frame, num_samples)
        weights = dcf[:, None, None, None, :, :]
        weights = np.tile(weights, (1, 1, num_coils, 1, 1, 1)).astype(np.float32, copy=False)
    else:
        weights = np.ones_like(ksp_prep, dtype=np.float32)

    _log(config.verbose, "subspace recon ksp=%s mps=%s basis=%s", ksp_prep.shape, mps.shape, basisoption.shape)

    recon = HighDimensionalRecon(
        ksp_prep,
        mps,
        weights=weights,
        coord=traj,
        basis=basisoption,
        lamda=config.lamda,
        regu=config.regu,
        regu_axes=config.regu_axes,
        max_iter=config.max_iter,
        solver=config.solver,
        device=device,
        show_pbar=config.show_pbar,
    ).run()

    recon = np.asarray(sp.to_device(recon, sp.cpu_device))
    coeff_maps = np.squeeze(recon)
    if coeff_maps.ndim != 3:
        raise ValueError(f"Expected coefficient maps to be 3D after squeeze, got {coeff_maps.shape}")
    if coeff_maps.shape[0] != basisoption.shape[1]:
        raise ValueError(
            f"Coefficient count mismatch: coeff_maps.shape[0]={coeff_maps.shape[0]} but basisoption.shape[1]={basisoption.shape[1]}"
        )

    img_dyn = np.einsum("tk,kyx->tyx", basisoption, coeff_maps)
    return Recon2DResult(coeff_maps=coeff_maps, img_dyn=img_dyn, img_dyn_abs=np.abs(img_dyn), basisoption=basisoption)


def _run_subspace_recon_for_slice(
    slice_file,
    traj,
    fbasis_path,
    spokes_per_frame,
    config: SliceReconstructionConfig | None = None,
    coil_thresh=None,
) -> SliceReconResult:
    config = config or SliceReconstructionConfig()
    slice_idx = config.slice_idx
    hop_id = config.hop_id or ""

    if slice_idx is None:
        match = re.search(r"slice(\d+)\.h5$", os.path.basename(slice_file))
        if match is not None:
            slice_idx = int(match.group(1)) - 1

    ksp_ri = load_slice_kspace_for_coil(slice_file, verbose=config.recon.verbose)
    ksp_c_for_coil = ri_to_coil_spokes_samples(ksp_ri)

    coil_config = config.coil if coil_thresh is None else type(config.coil)(
        thresh=coil_thresh,
        use_dcf=config.coil.use_dcf,
        mask_floor=config.coil.mask_floor,
        calib_width=config.coil.calib_width,
        crop=config.coil.crop,
        use_espirit=config.coil.use_espirit,
        verbose=config.coil.verbose or config.recon.verbose,
    )
    mps = _estimate_coil_maps(ksp_c_for_coil, device=config.coil_device, config=coil_config)

    recon = _run_subspace_recon_2d(
        ksp=ksp_c_for_coil,
        traj=traj,
        mps=mps,
        fbasis_path=fbasis_path,
        spokes_per_frame=spokes_per_frame,
        config=config.recon,
        device=config.recon_device,
    )

    if config.save_h5:
        if config.out_path is None:
            raise ValueError("out_path must be provided when save_h5=True")
        save_slice_h5(
            out_path=config.out_path,
            acq_slice=np.asarray(recon.img_dyn),
            hop_id=hop_id,
            spokes_per_frame=spokes_per_frame,
            N_time=recon.img_dyn.shape[0],
            slice_idx=-1 if slice_idx is None else slice_idx,
            smax=np.max(recon.img_dyn_abs),
        )

    return SliceReconResult(
        coeff_maps=recon.coeff_maps,
        img_dyn_cplx=recon.img_dyn,
        img_dyn_abs=recon.img_dyn_abs,
        basisoption=recon.basisoption,
        mps=np.asarray(sp.to_device(mps, sp.cpu_device)),
        ksp=ksp_c_for_coil,
    )
