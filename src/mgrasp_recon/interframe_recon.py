# # import function
# import argparse
# import inspect
# import h5py
# import os
# import pathlib

# import numpy as np
# import csv

# import re
# import glob
# import matplotlib.pyplot as plt
# import sigpy as sp

# import importlib
# from my_espirit import EspiritCalib

# # import sys
# # sys.path.insert(0, '/home/allisonchen/sigpy')

# import torch
# import cupy as cp

# def _cg_solve(normal_op, b, x0=None, max_iter=20, tol=1e-6):
#     """Complex-valued conjugate gradient solve for normal equations."""
#     x = np.zeros_like(b) if x0 is None else x0.copy()
#     r = b - normal_op(x)
#     p = r.copy()
#     rsold = np.vdot(r, r).real

#     if rsold < tol ** 2:
#         return x

#     for _ in range(max_iter):
#         ap = normal_op(p)
#         denom = np.vdot(p, ap).real
#         if np.abs(denom) < 1e-12:
#             break

#         alpha = rsold / denom
#         x = x + alpha * p
#         r = r - alpha * ap
#         rsnew = np.vdot(r, r).real

#         if rsnew < tol ** 2:
#             break

#         beta = rsnew / (rsold + 1e-12)
#         p = r + beta * p
#         rsold = rsnew

#     return x


# def _resize_complex_frames(frames, out_shape):
#     """Resize complex image frames with bilinear interpolation on real/imag parts."""
#     if tuple(frames.shape[-2:]) == tuple(out_shape):
#         return frames

#     real = torch.from_numpy(np.real(frames)).unsqueeze(1).float()
#     imag = torch.from_numpy(np.imag(frames)).unsqueeze(1).float()

#     real = torch.nn.functional.interpolate(
#         real, size=out_shape, mode="bilinear", align_corners=False
#     )
#     imag = torch.nn.functional.interpolate(
#         imag, size=out_shape, mode="bilinear", align_corners=False
#     )

#     return (real[:, 0].numpy() + 1j * imag[:, 0].numpy()).astype(np.complex64)

# def radial_lowres_pca_recon_2d(
#     ksp_one_slice,
#     traj,
#     spokes_per_frame,
#     mps,
#     img_shape=(96, 96),
#     ns_low=96,
#     method="adjoint",
#     cg_lamda=1e-3,
#     max_cg_iter=12,
#     cg_tol=1e-6,
#     normalize=True,
#     return_complex=False,
#     rescale_traj=False,
#     verbose=False,
# ):
#     """
#     Reconstruct a low-resolution dynamic series for PCA/subspace estimation.

#     Compared with ``radial_lowres_grasp_recon_2d``, this function intentionally
#     keeps the reconstruction prior weak so the temporal basis is driven mainly
#     by the measured data instead of temporal regularization.

#     Parameters
#     ----------
#     ksp_one_slice : ndarray
#         Supported layouts:
#         - complex (Nspokes_total, Ns, Nc)
#         - complex (Nc, Nspokes_total, Ns)
#         - real/imag split (2, Nspokes_total, Ns, Nc)
#         - real/imag split (Nspokes_total, Ns, Nc, 2)
#     traj : ndarray
#         Radial trajectory with shape (T, spf, Ns, 2).
#     spokes_per_frame : int
#         Number of spokes per frame.
#     mps : ndarray
#         Coil maps with shape (Nc, H, W), (1, Nc, H, W), or (Nc, 1, H, W).
#     img_shape : tuple[int, int]
#         Output low-resolution grid.
#     ns_low : int
#         Number of central readout samples kept for the low-resolution recon.
#     method : {"adjoint", "cg"}
#         ``adjoint`` is fast and usually a good default for PCA.
#         ``cg`` solves a weakly regularized SENSE normal equation.
#     cg_lamda : float
#         Tikhonov weight used only when ``method="cg"``.
#     normalize : bool
#         If True, divide each frame by its L2 norm before returning.
#     return_complex : bool
#         If True, return complex images; otherwise return magnitude images.
#     rescale_traj : bool
#         If True, rescale the cropped trajectory to match the target image grid.
#     verbose : bool
#         Print trajectory and reconstruction diagnostics.

#     Returns
#     -------
#     ndarray
#         Dynamic low-resolution series with shape (T, H, W).
#     """
#     ksp_one_slice = np.asarray(ksp_one_slice)

#     if ksp_one_slice.ndim == 4:
#         if ksp_one_slice.shape[0] == 2:
#             ksp_one_slice = ksp_one_slice[0] + 1j * ksp_one_slice[1]
#         elif ksp_one_slice.shape[-1] == 2:
#             ksp_one_slice = ksp_one_slice[..., 0] + 1j * ksp_one_slice[..., 1]
#         else:
#             raise ValueError(
#                 "4D k-space input must be either (2, Nspokes_total, Ns, Nc) "
#                 "or (Nspokes_total, Ns, Nc, 2). "
#                 f"Got {ksp_one_slice.shape}."
#             )
#     elif ksp_one_slice.ndim != 3:
#         raise ValueError(
#             "ksp_one_slice must have 3 dims (complex) or 4 dims (real/imag split). "
#             f"Got {ksp_one_slice.shape}."
#         )

#     mps = np.asarray(mps)
#     if mps.ndim == 4:
#         if mps.shape[0] == 1:
#             mps = mps[0]
#         elif mps.shape[1] == 1:
#             mps = mps[:, 0]
#         else:
#             raise ValueError(
#                 "4D mps must be either (1, Nc, H, W) or (Nc, 1, H, W). "
#                 f"Got {mps.shape}."
#             )
#     elif mps.ndim != 3:
#         raise ValueError(
#             "mps must have shape (Nc, H, W), (1, Nc, H, W), or (Nc, 1, H, W). "
#             f"Got {mps.shape}."
#         )

#     ncoils_mps = mps.shape[0]
#     if ksp_one_slice.shape[-1] == ncoils_mps:
#         pass
#     elif ksp_one_slice.shape[0] == ncoils_mps:
#         ksp_one_slice = np.transpose(ksp_one_slice, (1, 2, 0))
#     else:
#         raise ValueError(
#             "3D k-space input must be either (Nspokes_total, Ns, Nc) "
#             "or (Nc, Nspokes_total, Ns). "
#             f"Got {ksp_one_slice.shape} with mps coils={ncoils_mps}."
#         )

#     ksp_one_slice = np.asarray(ksp_one_slice, dtype=np.complex64)
#     traj = np.asarray(traj, dtype=np.float32)
#     mps = np.asarray(mps, dtype=np.complex64)

#     nspokes_total, ns, nc = ksp_one_slice.shape
#     t = nspokes_total // spokes_per_frame
#     nspokes_used = t * spokes_per_frame
#     if t == 0:
#         raise ValueError(
#             f"Not enough spokes ({nspokes_total}) for spokes_per_frame={spokes_per_frame}."
#         )
#     if traj.shape[:2] != (t, spokes_per_frame):
#         raise ValueError(
#             f"traj shape {traj.shape} is inconsistent with derived framing {(t, spokes_per_frame)}."
#         )
#     if mps.shape[0] != nc:
#         raise ValueError(f"mps has {mps.shape[0]} coils, but k-space has {nc}.")
#     if ns_low > ns:
#         raise ValueError(f"ns_low={ns_low} exceeds available readout samples {ns}.")

#     ksp_dyn = ksp_one_slice[:nspokes_used].reshape(t, spokes_per_frame, ns, nc)

#     c0 = ns // 2
#     h = ns_low // 2
#     s0 = c0 - h
#     s1 = s0 + ns_low
#     if s0 < 0 or s1 > ns:
#         raise ValueError(
#             f"Centered crop [{s0}:{s1}] is outside the readout dimension {ns}."
#         )

#     ksp_low = ksp_dyn[:, :, s0:s1, :]
#     coord_low = traj[:, :, s0:s1, :].copy()

#     if tuple(mps.shape[-2:]) != tuple(img_shape):
#         mps = _resize_complex_frames(mps, img_shape)

#     recon_shape = tuple(img_shape)

#     if rescale_traj:
#         coord_max = np.max(np.abs(coord_low))
#         target_max = ns_low / 2.0
#         if coord_max > 0:
#             coord_low *= (target_max / coord_max)

#     if verbose:
#         print(
#             f"[radial_lowres_pca_recon_2d] full traj range: "
#             f"[{traj.min():.3f}, {traj.max():.3f}]"
#         )
#         print(
#             f"[radial_lowres_pca_recon_2d] cropped/rescaled coord range: "
#             f"[{coord_low.min():.3f}, {coord_low.max():.3f}]"
#         )
#         print(
#             f"[radial_lowres_pca_recon_2d] recon_shape={recon_shape}, requested_img_shape={img_shape}, "
#             f"ns={ns}, ns_low={ns_low}, frames={t}, coils={nc}"
#         )

#     y = np.transpose(ksp_low, (0, 3, 1, 2)).astype(np.complex64, copy=False)

#     def forward_frame(img_t, coord_t):
#         out = np.zeros((nc, spokes_per_frame, ns_low), dtype=np.complex64)
#         for c in range(nc):
#             out[c] = sp.nufft(mps[c] * img_t, coord_t)
#         return out

#     def adjoint_frame(ksp_t, coord_t):
#         out = np.zeros(recon_shape, dtype=np.complex64)
#         for c in range(nc):
#             out += np.conj(mps[c]) * sp.nufft_adjoint(
#                 ksp_t[c], coord_t, oshape=recon_shape
#             )
#         return out

#     img_dyn = np.zeros((t,) + recon_shape, dtype=np.complex64)

#     method = method.lower()
#     if method not in {"adjoint", "cg"}:
#         raise ValueError(f"Unsupported method '{method}'. Use 'adjoint' or 'cg'.")

#     for frame_idx in range(t):
#         coord_t = coord_low[frame_idx]
#         rhs = adjoint_frame(y[frame_idx], coord_t)

#         if method == "adjoint":
#             img_dyn[frame_idx] = rhs
#             continue

#         def normal_op(img_t):
#             return adjoint_frame(forward_frame(img_t, coord_t), coord_t) + cg_lamda * img_t

#         img_dyn[frame_idx] = _cg_solve(
#             normal_op,
#             rhs,
#             x0=rhs.copy(),
#             max_iter=max_cg_iter,
#             tol=cg_tol,
#         )

#     if normalize:
#         frame_norm = np.linalg.norm(img_dyn.reshape(t, -1), axis=1, keepdims=True)
#         frame_norm = np.maximum(frame_norm, 1e-8)
#         img_dyn = img_dyn / frame_norm.reshape(t, 1, 1)

#     if tuple(img_shape) != recon_shape:
#         img_dyn = _resize_complex_frames(img_dyn, img_shape)

#     if return_complex:
#         return img_dyn

#     return np.abs(img_dyn).astype(np.float32, copy=False)

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
from .recon_utils import ramp_dcf_from_traj

# import sys
# sys.path.insert(0, '/home/allisonchen/sigpy')

import torch
import cupy as cp

def _cg_solve(normal_op, b, x0=None, max_iter=20, tol=1e-6):
    """Complex-valued conjugate gradient solve for normal equations."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - normal_op(x)
    p = r.copy()
    rsold = np.vdot(r, r).real

    if rsold < tol ** 2:
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

        if rsnew < tol ** 2:
            break

        beta = rsnew / (rsold + 1e-12)
        p = r + beta * p
        rsold = rsnew

    return x


def _resize_complex_frames(frames, out_shape):
    """Resize complex image frames with bilinear interpolation on real/imag parts."""
    if tuple(frames.shape[-2:]) == tuple(out_shape):
        return frames

    real = torch.from_numpy(np.real(frames)).unsqueeze(1).float()
    imag = torch.from_numpy(np.imag(frames)).unsqueeze(1).float()

    real = torch.nn.functional.interpolate(
        real, size=out_shape, mode="bilinear", align_corners=False
    )
    imag = torch.nn.functional.interpolate(
        imag, size=out_shape, mode="bilinear", align_corners=False
    )

    return (real[:, 0].numpy() + 1j * imag[:, 0].numpy()).astype(np.complex64)

def radial_lowres_pca_recon_2d(
    ksp_one_slice,
    traj,
    spokes_per_frame,
    mps,
    img_shape=(96, 96),
    ns_low=96,
    method="adjoint",
    cg_lamda=1e-3,
    max_cg_iter=12,
    cg_tol=1e-6,
    normalize=True,
    return_complex=False,
    rescale_traj=False,
    use_ramp_filter=False,
    ramp_filter_normalize=True,
    verbose=False,
):
    """
    Reconstruct a low-resolution dynamic series for PCA/subspace estimation.

    Compared with ``radial_lowres_grasp_recon_2d``, this function intentionally
    keeps the reconstruction prior weak so the temporal basis is driven mainly
    by the measured data instead of temporal regularization.

    Parameters
    ----------
    ksp_one_slice : ndarray
        Supported layouts:
        - complex (Nspokes_total, Ns, Nc)
        - complex (Nc, Nspokes_total, Ns)
        - real/imag split (2, Nspokes_total, Ns, Nc)
        - real/imag split (Nspokes_total, Ns, Nc, 2)
    traj : ndarray
        Radial trajectory with shape (T, spf, Ns, 2).
    spokes_per_frame : int
        Number of spokes per frame.
    mps : ndarray
        Coil maps with shape (Nc, H, W), (1, Nc, H, W), or (Nc, 1, H, W).
    img_shape : tuple[int, int]
        Output low-resolution grid.
    ns_low : int
        Number of central readout samples kept for the low-resolution recon.
    method : {"adjoint", "cg"}
        ``adjoint`` is fast and usually a good default for PCA.
        ``cg`` solves a weakly regularized SENSE normal equation.
    cg_lamda : float
        Tikhonov weight used only when ``method="cg"``.
    normalize : bool
        If True, divide each frame by its L2 norm before returning.
    return_complex : bool
        If True, return complex images; otherwise return magnitude images.
    rescale_traj : bool
        If True, rescale the cropped trajectory to match the target image grid.
    use_ramp_filter : bool
        If True, apply ramp DCF weights derived from the cropped trajectory.
    ramp_filter_normalize : bool
        If True, normalize the ramp DCF to [0, 1].
    verbose : bool
        Print trajectory and reconstruction diagnostics.

    Returns
    -------
    ndarray
        Dynamic low-resolution series with shape (T, H, W).
    """
    ksp_one_slice = np.asarray(ksp_one_slice)

    if ksp_one_slice.ndim == 4:
        if ksp_one_slice.shape[0] == 2:
            ksp_one_slice = ksp_one_slice[0] + 1j * ksp_one_slice[1]
        elif ksp_one_slice.shape[-1] == 2:
            ksp_one_slice = ksp_one_slice[..., 0] + 1j * ksp_one_slice[..., 1]
        else:
            raise ValueError(
                "4D k-space input must be either (2, Nspokes_total, Ns, Nc) "
                "or (Nspokes_total, Ns, Nc, 2). "
                f"Got {ksp_one_slice.shape}."
            )
    elif ksp_one_slice.ndim != 3:
        raise ValueError(
            "ksp_one_slice must have 3 dims (complex) or 4 dims (real/imag split). "
            f"Got {ksp_one_slice.shape}."
        )

    mps = np.asarray(mps)
    if mps.ndim == 4:
        if mps.shape[0] == 1:
            mps = mps[0]
        elif mps.shape[1] == 1:
            mps = mps[:, 0]
        else:
            raise ValueError(
                "4D mps must be either (1, Nc, H, W) or (Nc, 1, H, W). "
                f"Got {mps.shape}."
            )
    elif mps.ndim != 3:
        raise ValueError(
            "mps must have shape (Nc, H, W), (1, Nc, H, W), or (Nc, 1, H, W). "
            f"Got {mps.shape}."
        )

    ncoils_mps = mps.shape[0]
    if ksp_one_slice.shape[-1] == ncoils_mps:
        pass
    elif ksp_one_slice.shape[0] == ncoils_mps:
        ksp_one_slice = np.transpose(ksp_one_slice, (1, 2, 0))
    else:
        raise ValueError(
            "3D k-space input must be either (Nspokes_total, Ns, Nc) "
            "or (Nc, Nspokes_total, Ns). "
            f"Got {ksp_one_slice.shape} with mps coils={ncoils_mps}."
        )

    ksp_one_slice = np.asarray(ksp_one_slice, dtype=np.complex64)
    traj = np.asarray(traj, dtype=np.float32)
    mps = np.asarray(mps, dtype=np.complex64)

    nspokes_total, ns, nc = ksp_one_slice.shape
    t = nspokes_total // spokes_per_frame
    nspokes_used = t * spokes_per_frame
    if t == 0:
        raise ValueError(
            f"Not enough spokes ({nspokes_total}) for spokes_per_frame={spokes_per_frame}."
        )
    if traj.shape[:2] != (t, spokes_per_frame):
        raise ValueError(
            f"traj shape {traj.shape} is inconsistent with derived framing {(t, spokes_per_frame)}."
        )
    if mps.shape[0] != nc:
        raise ValueError(f"mps has {mps.shape[0]} coils, but k-space has {nc}.")
    if ns_low > ns:
        raise ValueError(f"ns_low={ns_low} exceeds available readout samples {ns}.")

    ksp_dyn = ksp_one_slice[:nspokes_used].reshape(t, spokes_per_frame, ns, nc)

    c0 = ns // 2
    h = ns_low // 2
    s0 = c0 - h
    s1 = s0 + ns_low
    if s0 < 0 or s1 > ns:
        raise ValueError(
            f"Centered crop [{s0}:{s1}] is outside the readout dimension {ns}."
        )

    ksp_low = ksp_dyn[:, :, s0:s1, :]
    coord_low = traj[:, :, s0:s1, :].copy()

    if tuple(mps.shape[-2:]) != tuple(img_shape):
        mps = _resize_complex_frames(mps, img_shape)

    recon_shape = tuple(img_shape)

    if rescale_traj:
        coord_max = np.max(np.abs(coord_low))
        target_max = ns_low / 2.0
        if coord_max > 0:
            coord_low *= (target_max / coord_max)

    if verbose:
        print(
            f"[radial_lowres_pca_recon_2d] full traj range: "
            f"[{traj.min():.3f}, {traj.max():.3f}]"
        )
        print(
            f"[radial_lowres_pca_recon_2d] cropped/rescaled coord range: "
            f"[{coord_low.min():.3f}, {coord_low.max():.3f}]"
        )
        print(
            f"[radial_lowres_pca_recon_2d] recon_shape={recon_shape}, requested_img_shape={img_shape}, "
            f"ns={ns}, ns_low={ns_low}, frames={t}, coils={nc}"
        )

    y = np.transpose(ksp_low, (0, 3, 1, 2)).astype(np.complex64, copy=False)
    dcf_low = None
    if use_ramp_filter:
        dcf_low = ramp_dcf_from_traj(
            coord_low,
            normalize=ramp_filter_normalize,
        ).reshape(t, spokes_per_frame, ns_low)
        dcf_low = dcf_low.astype(np.float32, copy=False)

        if verbose:
            print(
                f"[radial_lowres_pca_recon_2d] using ramp filter, range: "
                f"[{dcf_low.min():.3f}, {dcf_low.max():.3f}]"
            )

    def forward_frame(img_t, coord_t, dcf_t=None):
        out = np.zeros((nc, spokes_per_frame, ns_low), dtype=np.complex64)
        for c in range(nc):
            out[c] = sp.nufft(mps[c] * img_t, coord_t)
        if dcf_t is not None:
            out *= dcf_t[None, :, :]
        return out

    def adjoint_frame(ksp_t, coord_t, dcf_t=None):
        if dcf_t is not None:
            ksp_t = ksp_t * dcf_t[None, :, :]
        out = np.zeros(recon_shape, dtype=np.complex64)
        for c in range(nc):
            out += np.conj(mps[c]) * sp.nufft_adjoint(
                ksp_t[c], coord_t, oshape=recon_shape
            )
        return out

    img_dyn = np.zeros((t,) + recon_shape, dtype=np.complex64)

    method = method.lower()
    if method not in {"adjoint", "cg"}:
        raise ValueError(f"Unsupported method '{method}'. Use 'adjoint' or 'cg'.")

    for frame_idx in range(t):
        coord_t = coord_low[frame_idx]
        dcf_t = None if dcf_low is None else dcf_low[frame_idx]
        rhs = adjoint_frame(y[frame_idx], coord_t, dcf_t=dcf_t)

        if method == "adjoint":
            img_dyn[frame_idx] = rhs
            continue

        def normal_op(img_t):
            return (
                adjoint_frame(
                    forward_frame(img_t, coord_t, dcf_t=dcf_t),
                    coord_t,
                )
                + cg_lamda * img_t
            )

        img_dyn[frame_idx] = _cg_solve(
            normal_op,
            rhs,
            x0=rhs.copy(),
            max_iter=max_cg_iter,
            tol=cg_tol,
        )

    if normalize:
        frame_norm = np.linalg.norm(img_dyn.reshape(t, -1), axis=1, keepdims=True)
        frame_norm = np.maximum(frame_norm, 1e-8)
        img_dyn = img_dyn / frame_norm.reshape(t, 1, 1)

    if tuple(img_shape) != recon_shape:
        img_dyn = _resize_complex_frames(img_dyn, img_shape)

    if return_complex:
        return img_dyn

    return np.abs(img_dyn).astype(np.float32, copy=False)



