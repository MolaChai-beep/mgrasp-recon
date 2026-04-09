"""Low-resolution interframe reconstruction helpers."""

from __future__ import annotations

import logging

import numpy as np
import sigpy as sp
import torch

from .config import LowResReconConfig
from .recon_utils import ramp_dcf_from_traj

LOGGER = logging.getLogger(__name__)


def _log(verbose: bool, message: str, *args) -> None:
    if verbose:
        LOGGER.info(message, *args)


def _cg_solve(normal_op, b, x0=None, max_iter=20, tol=1e-6):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - normal_op(x)
    p = r.copy()
    rsold = np.vdot(r, r).real

    if rsold < tol**2:
        return x

    for _ in range(max_iter):
        ap = normal_op(p)
        denom = np.vdot(p, ap).real
        if np.abs(denom) < 1e-12:
            break

        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = np.vdot(r, r).real

        if rsnew < tol**2:
            break

        beta = rsnew / (rsold + 1e-12)
        p = r + beta * p
        rsold = rsnew

    return x


def _resize_complex_frames(frames, out_shape):
    if tuple(frames.shape[-2:]) == tuple(out_shape):
        return frames

    real = torch.from_numpy(np.real(frames)).unsqueeze(1).float()
    imag = torch.from_numpy(np.imag(frames)).unsqueeze(1).float()
    real = torch.nn.functional.interpolate(real, size=out_shape, mode="bilinear", align_corners=False)
    imag = torch.nn.functional.interpolate(imag, size=out_shape, mode="bilinear", align_corners=False)
    return (real[:, 0].numpy() + 1j * imag[:, 0].numpy()).astype(np.complex64)


def _coerce_kspace_layout(ksp_one_slice: np.ndarray, ncoils_mps: int) -> np.ndarray:
    ksp_one_slice = np.asarray(ksp_one_slice)
    if ksp_one_slice.ndim == 4:
        if ksp_one_slice.shape[0] == 2:
            ksp_one_slice = ksp_one_slice[0] + 1j * ksp_one_slice[1]
        elif ksp_one_slice.shape[-1] == 2:
            ksp_one_slice = ksp_one_slice[..., 0] + 1j * ksp_one_slice[..., 1]
        else:
            raise ValueError(
                "4D k-space input must be either (2, Nspokes_total, Ns, Nc) or (Nspokes_total, Ns, Nc, 2). "
                f"Got {ksp_one_slice.shape}."
            )
    elif ksp_one_slice.ndim != 3:
        raise ValueError(
            "ksp_one_slice must have 3 dims (complex) or 4 dims (real/imag split). "
            f"Got {ksp_one_slice.shape}."
        )

    if ksp_one_slice.shape[-1] == ncoils_mps:
        return np.asarray(ksp_one_slice, dtype=np.complex64)
    if ksp_one_slice.shape[0] == ncoils_mps:
        return np.asarray(np.transpose(ksp_one_slice, (1, 2, 0)), dtype=np.complex64)

    raise ValueError(
        "3D k-space input must be either (Nspokes_total, Ns, Nc) or (Nc, Nspokes_total, Ns). "
        f"Got {ksp_one_slice.shape} with mps coils={ncoils_mps}."
    )


def _coerce_mps_layout(mps: np.ndarray) -> np.ndarray:
    mps = np.asarray(mps)
    if mps.ndim == 4:
        if mps.shape[0] == 1:
            mps = mps[0]
        elif mps.shape[1] == 1:
            mps = mps[:, 0]
        else:
            raise ValueError(
                f"4D mps must be either (1, Nc, H, W) or (Nc, 1, H, W). Got {mps.shape}."
            )
    elif mps.ndim != 3:
        raise ValueError(
            f"mps must have shape (Nc, H, W), (1, Nc, H, W), or (Nc, 1, H, W). Got {mps.shape}."
        )
    return np.asarray(mps, dtype=np.complex64)


def _radial_lowres_pca_recon_2d(ksp_one_slice, traj, spokes_per_frame, mps, config: LowResReconConfig):
    mps = _coerce_mps_layout(mps)
    ksp_one_slice = _coerce_kspace_layout(ksp_one_slice, mps.shape[0])
    traj = np.asarray(traj, dtype=np.float32)

    nspokes_total, ns, nc = ksp_one_slice.shape
    num_frames = nspokes_total // spokes_per_frame
    nspokes_used = num_frames * spokes_per_frame

    if num_frames == 0:
        raise ValueError(f"Not enough spokes ({nspokes_total}) for spokes_per_frame={spokes_per_frame}.")
    if traj.shape[:2] != (num_frames, spokes_per_frame):
        raise ValueError(
            f"traj shape {traj.shape} is inconsistent with derived framing {(num_frames, spokes_per_frame)}."
        )
    if mps.shape[0] != nc:
        raise ValueError(f"mps has {mps.shape[0]} coils, but k-space has {nc}.")
    if config.ns_low > ns:
        raise ValueError(f"ns_low={config.ns_low} exceeds available readout samples {ns}.")

    ksp_dyn = ksp_one_slice[:nspokes_used].reshape(num_frames, spokes_per_frame, ns, nc)
    center = ns // 2
    half_width = config.ns_low // 2
    start = center - half_width
    stop = start + config.ns_low
    if start < 0 or stop > ns:
        raise ValueError(f"Centered crop [{start}:{stop}] is outside the readout dimension {ns}.")

    ksp_low = ksp_dyn[:, :, start:stop, :]
    coord_low = traj[:, :, start:stop, :].copy()

    if tuple(mps.shape[-2:]) != tuple(config.img_shape):
        mps = _resize_complex_frames(mps, config.img_shape)

    if config.rescale_traj:
        coord_max = np.max(np.abs(coord_low))
        target_max = config.ns_low / 2.0
        if coord_max > 0:
            coord_low *= target_max / coord_max

    _log(config.verbose, "lowres recon shape=%s frames=%d coils=%d", config.img_shape, num_frames, nc)

    y = np.transpose(ksp_low, (0, 3, 1, 2)).astype(np.complex64, copy=False)
    dcf_low = None
    if config.use_ramp_filter:
        dcf_low = ramp_dcf_from_traj(coord_low, normalize=config.ramp_filter_normalize)
        dcf_low = dcf_low.reshape(num_frames, spokes_per_frame, config.ns_low).astype(np.float32, copy=False)

    def forward_frame(img_t, coord_t, dcf_t=None):
        out = np.zeros((nc, spokes_per_frame, config.ns_low), dtype=np.complex64)
        for coil_idx in range(nc):
            out[coil_idx] = sp.nufft(mps[coil_idx] * img_t, coord_t)
        if dcf_t is not None:
            out *= dcf_t[None, :, :]
        return out

    def adjoint_frame(ksp_t, coord_t, dcf_t=None):
        if dcf_t is not None:
            ksp_t = ksp_t * dcf_t[None, :, :]
        out = np.zeros(tuple(config.img_shape), dtype=np.complex64)
        for coil_idx in range(nc):
            out += np.conj(mps[coil_idx]) * sp.nufft_adjoint(ksp_t[coil_idx], coord_t, oshape=tuple(config.img_shape))
        return out

    img_dyn = np.zeros((num_frames,) + tuple(config.img_shape), dtype=np.complex64)
    method = config.method.lower()
    if method not in {"adjoint", "cg"}:
        raise ValueError(f"Unsupported method '{config.method}'. Use 'adjoint' or 'cg'.")

    for frame_idx in range(num_frames):
        coord_t = coord_low[frame_idx]
        dcf_t = None if dcf_low is None else dcf_low[frame_idx]
        rhs = adjoint_frame(y[frame_idx], coord_t, dcf_t=dcf_t)

        if method == "adjoint":
            img_dyn[frame_idx] = rhs
            continue

        def normal_op(img_t):
            return adjoint_frame(forward_frame(img_t, coord_t, dcf_t=dcf_t), coord_t) + config.cg_lamda * img_t

        img_dyn[frame_idx] = _cg_solve(normal_op, rhs, x0=rhs.copy(), max_iter=config.max_cg_iter, tol=config.cg_tol)

    if config.normalize:
        frame_norm = np.linalg.norm(img_dyn.reshape(num_frames, -1), axis=1, keepdims=True)
        frame_norm = np.maximum(frame_norm, 1e-8)
        img_dyn = img_dyn / frame_norm.reshape(num_frames, 1, 1)

    if config.return_complex:
        return img_dyn
    return np.abs(img_dyn).astype(np.float32, copy=False)
