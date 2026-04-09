
import numpy as np
from ._bootstrap import ensure_repo_paths

ensure_repo_paths()

import sigpy as sp

from sigpy.mri import linop, nlop
from sigpy.mri.dims import *
import numpy as np
import matplotlib.pyplot as plt


class EspiritCalib(sp.app.App):
    """ESPIRiT calibration.

    Currently only supports outputting one set of maps.

    Args:
        ksp (array): k-space array of shape [num_coils, n_ndim, ..., n_1]
        calib (tuple of ints): length-2 image shape.
        thresh (float): threshold for the calibration matrix.
        kernel_width (int): kernel width for the calibration matrix.
        max_power_iter (int): maximum number of power iterations.
        device (Device): computing device.
        crop (int): cropping threshold.

    Returns:
        array: ESPIRiT maps of the same shape as ksp.

    References:
        Martin Uecker, Peng Lai, Mark J. Murphy, Patrick Virtue, Michael Elad,
        John M. Pauly, Shreyas S. Vasanawala, and Michael Lustig
        ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI:
        Where SENSE meets GRAPPA.
        Magnetic Resonance in Medicine, 71:990-1001 (2014)

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
    ):
        print(">>> EspiritCalib __init__ CALLED", flush=True)
        self.device = sp.Device(device)
        self.output_eigenvalue = output_eigenvalue
        self.crop = crop

        img_ndim = ksp.ndim - 1
        num_coils = len(ksp)
        with sp.get_device(ksp):
            # Get calibration region
            calib_shape = [num_coils] + [calib_width] * img_ndim
            calib = sp.resize(ksp, calib_shape)
            calib = sp.to_device(calib, device)

        xp = self.device.xp
        with self.device:
            # Get calibration matrix.
            # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
            mat = sp.array_to_blocks(
                calib, [kernel_width] * img_ndim, [1] * img_ndim
            )
            mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
            mat = mat.transpose([1, 0, 2])
            mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])

            # Perform SVD on calibration matrix
            _, S, VH = xp.linalg.svd(mat, full_matrices=False)
            VH = VH[S > thresh * S.max(), :]

            print("Top 10 singular values:")
            print(S[:10],flush=True)
            print("S max:", S.max())
            print("S min:", S.min())
            S_np = np.asarray(S.get() if hasattr(S, "get") else S)
            plt.figure()
            plt.semilogy(S_np, '.-')
            plt.title("Singular values (log scale)")
            plt.xlabel("index")
            plt.ylabel("S")
            plt.grid(True)
            plt.show()

            thr = thresh * S_np.max()
            print("threshold =", thr)
            print("kept =", np.sum(S_np > thr), "out of", S_np.size)

            # Get kernels
            num_kernels = len(VH)
            print("Number of kernels kept:", num_kernels)
            kernels = VH.reshape(
                [num_kernels, num_coils] + [kernel_width] * img_ndim
            )
            img_shape = ksp.shape[1:]

            # Get covariance matrix in image domain
            AHA = xp.zeros(
                img_shape[::-1] + (num_coils, num_coils), dtype=ksp.dtype
            )
            for kernel in kernels:
                img_kernel = sp.ifft(
                    sp.resize(kernel, ksp.shape), axes=range(-img_ndim, 0)
                )
                aH = xp.expand_dims(img_kernel.T, axis=-1)
                a = xp.conj(aH.swapaxes(-1, -2))
                AHA += aH @ a

            AHA *= sp.prod(img_shape) / kernel_width**img_ndim
            self.mps = xp.ones(ksp.shape[::-1] + (1,), dtype=ksp.dtype)
            
            def forward(x):
                with sp.get_device(x):
                    return AHA @ x

            def normalize(x):
                with sp.get_device(x):
                    return (
                        xp.sum(xp.abs(x) ** 2, axis=-2, keepdims=True) ** 0.5
                    )

            alg = sp.alg.PowerMethod(
                forward, self.mps, norm_func=normalize, max_iter=max_iter
            )
            

            #------ test for updating function and eigenvalue computation ------
            print("before update iteration:", alg.iter)

            while not alg.done():
                alg.update()
                print("current iteration:", alg.iter)

            print("AHA abs max:", float(xp.abs(AHA).max()), flush=True)
            print("AHA abs mean:", float(xp.abs(AHA).mean()), flush=True)
            print("AHA min abs:", float(xp.min(xp.abs(AHA))))
            test_pixel = AHA[0,0]   
            print("test_pixel matrix:\n", test_pixel)
            print("PowerMethod max eig:", alg.max_eig, flush=True)


        super().__init__(alg, show_pbar=show_pbar)
        self.max_eig = np.array([alg.max_eig])

        self.sets = sets
        if self.sets > 1:
            U, S, VH = xp.linalg.svd(AHA, full_matrices=False)

            print(U.shape, S.shape, VH.shape)

            self.mps = U[..., :self.sets]
            self.max_eig = S[..., :self.sets]

        print('self.mps ', self.mps.shape)
        print('self.max_eig ', self.max_eig.shape)

    # def _output(self):
    #     xp = self.device.xp
    #     with self.device:
    #         mps = []
    #         max_eig = []
    #         for s in range(self.mps.shape[-1]):
    #             # Normalize phase with respect to first channel
    #             c = self.mps.T[s]
    #             c *= xp.conj(c[0] / xp.abs(c[0]))

    #             # Crop maps by thresholding eigenvalue
    #             me = self.max_eig.T[s]
    #             c *= me > self.crop

    #             mps.append( c )
    #             max_eig.append( me )

    #         mps = xp.array(mps)
    #         max_eig = xp.array(max_eig)
    def _output(self):
        xp = self.device.xp
        with self.device:
            mps = []
            max_eig = []
            for s in range(self.mps.shape[-1]):
                # Normalize phase with respect to first channel
                c = self.mps.T[s]
                c *= xp.conj(c[0] / xp.abs(c[0]))

                me = self.max_eig.T[s]   
                # take 2D eigenvalue map explicitly -> (256,256)
                me2d = me[0, :, :, 0]
                mask = (me2d > self.crop)

                # broadcast to coils: (64,256,256) * (1,256,256)
                c *= mask[None, :, :]

                mps.append( c )
                max_eig.append( me )

            mps = xp.array(mps)
            max_eig = xp.array(max_eig)

        if self.output_eigenvalue:
            return mps, max_eig
        else:
            return mps

