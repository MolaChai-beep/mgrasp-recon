import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class RefactorTests(unittest.TestCase):
    def test_make_basis_option_and_roundtrip(self):
        from mgrasp_recon.recon_utils import load_pca_basis_h5, make_basis_option, save_pca_basis_h5

        basis = np.arange(12, dtype=np.float32).reshape(4, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "basis.h5"
            save_pca_basis_h5(path, basis)
            loaded = load_pca_basis_h5(path)

        self.assertEqual(loaded.shape, basis.shape)
        np.testing.assert_allclose(loaded, basis)

        real_basis = make_basis_option(basis, cbasis=False, add_constant=True, dtype=np.float32)
        complex_basis = make_basis_option(basis, cbasis=True, add_constant=True)
        self.assertEqual(real_basis.shape, (4, 4))
        self.assertEqual(complex_basis.shape, (4, 7))

    def test_kspace_layout_and_tic_helpers(self):
        from mgrasp_recon import TicAnalyzer
        from mgrasp_recon.recon_utils import ri_to_coil_spokes_samples, ramp_dcf_from_traj

        ksp_ri = np.zeros((2, 3, 4, 2), dtype=np.float32)
        ksp_ri[0, 1, 2, 1] = 3.0
        ksp = ri_to_coil_spokes_samples(ksp_ri)
        self.assertEqual(ksp.shape, (2, 3, 4))
        self.assertEqual(ksp[1, 1, 2], 3.0)

        traj = np.zeros((2, 3, 4, 2), dtype=np.float32)
        traj[..., 0] = 1.0
        dcf = ramp_dcf_from_traj(traj)
        self.assertEqual(dcf.shape, (6, 4))
        self.assertTrue(np.all(dcf >= 0))

        img_dyn = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        curve = TicAnalyzer().extract_voxel_tic(img_dyn, (1, 2), normalize=True)
        self.assertEqual(curve.shape, (3,))
        self.assertGreater(curve[-1], curve[0])

    def test_imports_do_not_mutate_sys_path_or_sys_modules(self):
        sys.modules["sigpy.fake_marker"] = types.ModuleType("sigpy.fake_marker")
        import mgrasp_recon
        import mgrasp_recon.recon_2d as recon_2d

        path_before = list(sys.path)
        importlib.reload(mgrasp_recon)
        importlib.reload(recon_2d)

        self.assertEqual(path_before, sys.path)
        self.assertIn("sigpy.fake_marker", sys.modules)
        del sys.modules["sigpy.fake_marker"]

    def test_lowres_reconstructor_and_no_plot_default(self):
        from mgrasp_recon import LowResReconConfig, LowResReconstructor

        ksp = np.ones((4, 6, 2), dtype=np.complex64)
        traj = np.zeros((2, 2, 6, 2), dtype=np.float32)
        mps = np.ones((2, 4, 4), dtype=np.complex64)

        def fake_nufft(image, coord):
            return np.ones(coord.shape[:-1], dtype=np.complex64) * np.mean(image)

        def fake_nufft_adjoint(samples, coord, oshape):
            return np.ones(oshape, dtype=np.complex64) * np.mean(samples)

        reconstructor = LowResReconstructor(
            LowResReconConfig(
                img_shape=(4, 4),
                ns_low=4,
                method="adjoint",
                normalize=False,
                return_complex=False,
            )
        )

        with mock.patch("mgrasp_recon.interframe_recon.sp.nufft", side_effect=fake_nufft), \
             mock.patch("mgrasp_recon.interframe_recon.sp.nufft_adjoint", side_effect=fake_nufft_adjoint), \
             mock.patch("matplotlib.pyplot.show") as show_mock:
            out = reconstructor.reconstruct(ksp, traj, spokes_per_frame=2, mps=mps)

        self.assertEqual(out.shape, (2, 4, 4))
        show_mock.assert_not_called()

    def test_slice_reconstruction_workflow_run_slice(self):
        from mgrasp_recon import ReconstructionConfig, SliceReconstructionConfig, SliceReconstructionWorkflow

        class FakeRecon:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def run(self):
                return np.ones((3, 8, 8), dtype=np.complex64)

        workflow = SliceReconstructionWorkflow(
            SliceReconstructionConfig(
                recon=ReconstructionConfig(
                    nbasis=2,
                    lamda=1e-3,
                    regu="TV",
                    regu_axes=(-2, -1),
                    max_iter=10,
                    solver="ADMM",
                    use_dcf=True,
                    show_pbar=False,
                )
            )
        )
        ksp = np.ones((2, 4, 6), dtype=np.complex64)
        traj = np.zeros((2, 2, 6, 2), dtype=np.float32)
        mps = np.ones((1, 2, 8, 8), dtype=np.complex64)
        basis = np.ones((2, 3), dtype=np.float32)

        with mock.patch("mgrasp_recon.recon_2d.load_basis_option_from_h5", return_value=basis), \
             mock.patch("mgrasp_recon.recon_2d.HighDimensionalRecon", FakeRecon):
            result = workflow.run_slice(ksp, traj, mps, "dummy.h5", 2)

        self.assertEqual(result.coeff_maps.shape, (3, 8, 8))
        self.assertEqual(result.img_dyn.shape, (2, 8, 8))
        self.assertEqual(result.img_dyn_abs.shape, (2, 8, 8))
        self.assertEqual(result.basisoption.shape, (2, 3))

    def test_coil_map_estimator_no_plot_default(self):
        from mgrasp_recon import CoilCalibrationConfig, CoilMapEstimator

        class FakeAdjoint:
            def __call__(self, data):
                return np.ones((2, 4, 4), dtype=np.complex64)

        class FakeNUFFT:
            def __init__(self, ishape, traj):
                self.H = FakeAdjoint()

        estimator = CoilMapEstimator(CoilCalibrationConfig(use_espirit=False), device=-1)
        ksp = np.ones((2, 3, 8), dtype=np.complex64)
        with mock.patch("mgrasp_recon.recon_utils.sp.linop.NUFFT", FakeNUFFT), \
             mock.patch("mgrasp_recon.recon_utils.sp.to_device", side_effect=lambda arr, device=None: arr), \
             mock.patch("matplotlib.pyplot.show") as show_mock:
            mps = estimator.estimate(ksp)

        self.assertEqual(np.asarray(mps).shape, (1, 2, 4, 4))
        show_mock.assert_not_called()

    def test_segmentation_analyzer_defaults_and_plot_separation(self):
        from mgrasp_recon import SegmentationAnalyzer

        analyzer = SegmentationAnalyzer()
        img = np.ones((6, 12, 12), dtype=np.float32)
        img[:, 4:8, 4:8] += np.linspace(0, 1, 6)[:, None, None]

        with mock.patch("mgrasp_recon.visualization.plot_segmentation_summary") as summary_mock, \
             mock.patch("mgrasp_recon.visualization.plot_segmentation_frames") as frames_mock:
            result = analyzer.segment_enhancement_series(img)
            dyn = analyzer.segment_dynamic_series(img)

        self.assertEqual(result.vascular_mask.shape, (12, 12))
        self.assertEqual(result.tissue_mask.shape, (12, 12))
        self.assertEqual(dyn.tissue_mask.shape, (12, 12))
        summary_mock.assert_not_called()
        frames_mock.assert_not_called()

    def test_basis_estimator_segmented_basis_and_roundtrip(self):
        from mgrasp_recon import BasisEstimator

        img = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
        vascular_mask = np.zeros((4, 4), dtype=bool)
        vascular_mask[:2, :2] = True
        tissue_mask = np.zeros((4, 4), dtype=bool)
        tissue_mask[2:, 2:] = True

        estimator = BasisEstimator(nbasis=2)
        vascular_basis, tissue_basis = estimator.estimate_segmented_basis(img, vascular_mask, tissue_mask)
        self.assertEqual(vascular_basis.shape, (5, 2))
        self.assertEqual(tissue_basis.shape, (5, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "basis.h5"
            estimator.save_basis(path, vascular_basis)
            loaded = estimator.load_basis(path, nbasis=2)
        self.assertEqual(loaded.shape, vascular_basis.shape)

    def test_basis_preparation_workflow_returns_structured_result(self):
        from mgrasp_recon import BasisPreparationConfig, BasisPreparationWorkflow, SegmentationResult

        workflow = BasisPreparationWorkflow(BasisPreparationConfig(spokes_per_frame=2))
        segmentation = SegmentationResult(
            mean_img=np.ones((4, 4), dtype=np.float32),
            peak_img=np.ones((4, 4), dtype=np.float32),
            std_img=np.ones((4, 4), dtype=np.float32),
            baseline_mean=np.ones((4, 4), dtype=np.float32),
            norm_early_enh=np.ones((4, 4), dtype=np.float32),
            brain_mask=np.ones((4, 4), dtype=bool),
            brain_core_mask=np.ones((4, 4), dtype=bool),
            brain_ring_mask=np.zeros((4, 4), dtype=bool),
            pca_roi_mask=np.ones((4, 4), dtype=bool),
            vascular_mask=np.ones((4, 4), dtype=bool),
            tissue_mask=np.zeros((4, 4), dtype=bool),
            brain_mask_threshold=0.1,
            enhancement_threshold=0.2,
        )

        with mock.patch.object(workflow, "load_reference_slice", return_value=np.ones((2, 4, 8), dtype=np.complex64)), \
             mock.patch.object(workflow, "estimate_coils", return_value=np.ones((1, 2, 4, 4), dtype=np.complex64)), \
             mock.patch.object(workflow, "reconstruct_lowres_series", return_value=np.ones((2, 4, 4), dtype=np.float32)), \
             mock.patch.object(workflow, "segment_vascular_and_tissue", return_value=segmentation), \
             mock.patch.object(
                 workflow,
                 "estimate_segmented_basis",
                 return_value=(np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)),
             ), \
             mock.patch.object(workflow, "save_basis", return_value="basis.h5"):
            result = workflow.run(["slice1.h5", "slice2.h5", "slice3.h5"])

        self.assertEqual(result.slice_idx, 1)
        self.assertEqual(result.slice_file, "slice2.h5")
        self.assertEqual(result.basis_path, "basis.h5")
        self.assertEqual(result.img_lowres.shape, (2, 4, 4))

    def test_slice_reconstruction_workflow_calls_internal_slice_runner(self):
        from mgrasp_recon import SliceReconResult, SliceReconstructionConfig, SliceReconstructionWorkflow

        workflow = SliceReconstructionWorkflow(SliceReconstructionConfig())
        expected = SliceReconResult(
            coeff_maps=np.ones((3, 4, 4), dtype=np.complex64),
            img_dyn_cplx=np.ones((2, 4, 4), dtype=np.complex64),
            img_dyn_abs=np.ones((2, 4, 4), dtype=np.float32),
            basisoption=np.ones((2, 3), dtype=np.float32),
            mps=np.ones((1, 2, 4, 4), dtype=np.complex64),
            ksp=np.ones((2, 4, 8), dtype=np.complex64),
        )

        with mock.patch("mgrasp_recon.workflows._run_subspace_recon_for_slice", return_value=expected) as runner:
            result = workflow.reconstruct_slice("slice1.h5", np.zeros((2, 2, 8, 2), dtype=np.float32), "basis.h5", 2)

        self.assertIs(result, expected)
        runner.assert_called_once()

    def test_patient_workflow_run_all_slices_is_deterministic(self):
        from mgrasp_recon import BasisPreparationConfig, PatientReconstructionWorkflow, SliceReconstructionConfig

        workflow = PatientReconstructionWorkflow(
            hop_id="hop1",
            input_dir="dummy",
            basis_config=BasisPreparationConfig(spokes_per_frame=2),
            reconstruction_config=SliceReconstructionConfig(hop_id="hop1"),
        )

        fake_results = [mock.Mock(name="slice1"), mock.Mock(name="slice2")]
        fake_failures = [("slice3.h5", "failed")]
        with mock.patch.object(workflow, "_get_slice_files", return_value=["slice1.h5", "slice2.h5", "slice3.h5"]), \
             mock.patch.object(workflow.slice_workflow, "reconstruct_all_slices", return_value=(fake_results, fake_failures)) as runner:
            result = workflow.run_all_slices("dummy", np.zeros((2, 2, 8, 2), dtype=np.float32), "basis.h5")

        self.assertEqual(result.hop_id, "hop1")
        self.assertEqual(result.slice_results, fake_results)
        self.assertEqual(result.failed_slices, fake_failures)
        runner.assert_called_once_with(["slice1.h5", "slice2.h5", "slice3.h5"], mock.ANY, "basis.h5", 2)


if __name__ == "__main__":
    unittest.main()
