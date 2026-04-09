"""ESPIRiT calibration utilities."""

from __future__ import annotations

import logging

import numpy as np
import sigpy as sp

LOGGER = logging.getLogger(__name__)


class EspiritCalib(sp.app.App):
    """ESPIRiT calibration.

    Currently only supports outputting one set of maps by default.
    """

    def __init__(
        self,
        ksp,
        calib_width=24,
        thresh=0.02,
        kernel_width=6,
        crop=0.95,
        max_iter=100,
        sets=1,
        device=sp.cpu_device,
        output_eigenvalue=False,
        show_pbar=True,
        verbose=False,
    ):
        self.device = sp.Device(device)
        self.output_eigenvalue = output_eigenvalue
        self.crop = crop
        self.sets = sets
        self.verbose = verbose

        img_ndim = ksp.ndim - 1
        num_coils = len(ksp)
        with sp.get_device(ksp):
            calib_shape = [num_coils] + [calib_width] * img_ndim
            calib = sp.resize(ksp, calib_shape)
            calib = sp.to_device(calib, device)

        xp = self.device.xp
        with self.device:
            mat = sp.array_to_blocks(calib, [kernel_width] * img_ndim, [1] * img_ndim)
            mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
            mat = mat.transpose([1, 0, 2])
            mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])

            _, singular_values, vh = xp.linalg.svd(mat, full_matrices=False)
            keep = singular_values > thresh * singular_values.max()
            vh = vh[keep, :]

            self._log(
                "kept %d kernels out of %d with threshold %.4g",
                int(np.sum(np.asarray(keep))),
                singular_values.size,
                float(thresh),
            )

            num_kernels = len(vh)
            kernels = vh.reshape([num_kernels, num_coils] + [kernel_width] * img_ndim)
            img_shape = ksp.shape[1:]

            aha = xp.zeros(img_shape[::-1] + (num_coils, num_coils), dtype=ksp.dtype)
            for kernel in kernels:
                img_kernel = sp.ifft(sp.resize(kernel, ksp.shape), axes=range(-img_ndim, 0))
                a_h = xp.expand_dims(img_kernel.T, axis=-1)
                a = xp.conj(a_h.swapaxes(-1, -2))
                aha += a_h @ a

            aha *= sp.prod(img_shape) / kernel_width**img_ndim
            self.mps = xp.ones(ksp.shape[::-1] + (1,), dtype=ksp.dtype)

            def forward(x):
                with sp.get_device(x):
                    return aha @ x

            def normalize(x):
                with sp.get_device(x):
                    return xp.sum(xp.abs(x) ** 2, axis=-2, keepdims=True) ** 0.5

            alg = sp.alg.PowerMethod(forward, self.mps, norm_func=normalize, max_iter=max_iter)
            while not alg.done():
                alg.update()

            self.max_eig = np.array([alg.max_eig])

            if self.sets > 1:
                u, s, _ = xp.linalg.svd(aha, full_matrices=False)
                self.mps = u[..., : self.sets]
                self.max_eig = s[..., : self.sets]

        super().__init__(alg, show_pbar=show_pbar)

    def _log(self, message, *args):
        if self.verbose:
            LOGGER.info(message, *args)

    def _output(self):
        xp = self.device.xp
        with self.device:
            mps = []
            max_eig = []
            for s in range(self.mps.shape[-1]):
                c = self.mps.T[s]
                c *= xp.conj(c[0] / xp.maximum(xp.abs(c[0]), 1e-8))

                me = self.max_eig.T[s]
                mask = xp.squeeze(me) > self.crop
                while mask.ndim < c.ndim:
                    mask = xp.expand_dims(mask, axis=0)
                c *= mask

                mps.append(c)
                max_eig.append(me)

            mps = xp.array(mps)
            max_eig = xp.array(max_eig)

        if self.output_eigenvalue:
            return mps, max_eig
        return mps
