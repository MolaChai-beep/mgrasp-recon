import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
COMMON_PATH = CODE_DIR / "combined_subjects_common.py"
STEP1_PATH = CODE_DIR / "step1_combined_subjects.py"
STEP2_PATH = CODE_DIR / "step2_combined_subjects.py"
BATCH_PATH = CODE_DIR / "batch_step1_step2_combined_subjects.py"


def install_stubs():
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.ndarray = object
        numpy_stub.asarray = lambda value, dtype=None: value
        numpy_stub.arange = lambda *args, **kwargs: []
        numpy_stub.float32 = float
        numpy_stub.complex64 = complex
        numpy_stub.max = max
        sys.modules["numpy"] = numpy_stub

    if "matplotlib" not in sys.modules:
        matplotlib_stub = types.ModuleType("matplotlib")
        pyplot_stub = types.ModuleType("matplotlib.pyplot")
        pyplot_stub.close = lambda *args, **kwargs: None
        pyplot_stub.subplots = lambda *args, **kwargs: (None, [types.SimpleNamespace(imshow=lambda *a, **k: None, set_title=lambda *a, **k: None, axis=lambda *a, **k: None)] * 3)
        matplotlib_stub.pyplot = pyplot_stub
        sys.modules["matplotlib"] = matplotlib_stub
        sys.modules["matplotlib.pyplot"] = pyplot_stub

    if "sigpy" not in sys.modules:
        sigpy_stub = types.ModuleType("sigpy")

        class _Device:
            def __init__(self, device_id):
                self.id = device_id

            def __repr__(self):
                return f"Device({self.id})"

        sigpy_stub.Device = _Device
        sys.modules["sigpy"] = sigpy_stub

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        class _Cuda:
            class OutOfMemoryError(MemoryError):
                pass

            @staticmethod
            def is_available():
                return False

            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def device_count():
                return 0

            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def get_device_name(_index):
                return "stub-gpu"

        torch_stub.cuda = _Cuda
        sys.modules["torch"] = torch_stub

    if "mgrasp_recon" not in sys.modules:
        mgrasp_stub = types.ModuleType("mgrasp_recon")

        class _Dummy:
            def __init__(self, *args, **kwargs):
                self.config = types.SimpleNamespace(coil=None)

            def estimate(self, *args, **kwargs):
                return None

            def save_basis(self, *args, **kwargs):
                return None

        class _Ticker:
            def extract_voxel_tic(self, *args, **kwargs):
                return []

        mgrasp_stub.BasisPreparationConfig = _Dummy
        mgrasp_stub.BasisPreparationWorkflow = _Dummy
        mgrasp_stub.CoilCalibrationConfig = _Dummy
        mgrasp_stub.CoilMapEstimator = _Dummy
        mgrasp_stub.LowResReconConfig = _Dummy
        mgrasp_stub.ReconstructionConfig = _Dummy
        mgrasp_stub.SegmentationConfig = _Dummy
        mgrasp_stub.SliceReconstructionConfig = _Dummy
        mgrasp_stub.SliceReconstructionWorkflow = _Dummy
        mgrasp_stub.TicAnalyzer = _Ticker
        sys.modules["mgrasp_recon"] = mgrasp_stub

        recon_utils_stub = types.ModuleType("mgrasp_recon.recon_utils")
        recon_utils_stub.get_traj = lambda **kwargs: []
        recon_utils_stub.infer_kspace_dims = lambda *args, **kwargs: (34, 512, 128, 2)
        recon_utils_stub.list_slice_files = lambda *args, **kwargs: []
        recon_utils_stub.load_slice_kspace_for_coil = lambda *args, **kwargs: None
        recon_utils_stub.read_csv_config = lambda *args, **kwargs: []
        recon_utils_stub.ri_to_coil_spokes_samples = lambda *args, **kwargs: None
        recon_utils_stub.save_slice_h5 = lambda *args, **kwargs: None
        sys.modules["mgrasp_recon.recon_utils"] = recon_utils_stub

        visualization_stub = types.ModuleType("mgrasp_recon.visualization")
        visualization_stub.plot_segmentation_summary = lambda *args, **kwargs: (None, None)
        sys.modules["mgrasp_recon.visualization"] = visualization_stub


def load_module(name: str, path: Path):
    install_stubs()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BatchCombinedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = load_module("combined_subjects_common", COMMON_PATH)
        cls.step1 = load_module("step1_combined_subjects", STEP1_PATH)
        cls.step2 = load_module("step2_combined_subjects", STEP2_PATH)
        cls.batch = load_module("batch_step1_step2_combined_subjects", BATCH_PATH)

    def make_info(self, hop_id, csv_spf, n_spokes):
        return self.common.SeriesInfo(
            hop_id=hop_id,
            hop_dir=Path(f"/tmp/{hop_id}"),
            slice_files=["slice001.h5"] * 96,
            original_spokes_per_frame=csv_spf,
            n_coils=34,
            n_samples=512,
            n_spokes=n_spokes,
        )

    def test_choose_combined_spf_prefers_dce_and_keeps_fa_frames(self):
        infos = [
            self.make_info("DCE", 23, 2220),
            self.make_info("FA2", 1, 128),
            self.make_info("FA15", 1, 128),
            self.make_info("FA2p", 1, 128),
            self.make_info("FA15p", 1, 128),
        ]

        combined_spf, stats = self.common.choose_combined_spokes_per_frame(infos)

        self.assertEqual(combined_spf, 16)
        stats_map = {item.hop_id: item for item in stats}
        self.assertEqual(stats_map["DCE"].num_frames, 138)
        self.assertEqual(stats_map["DCE"].dropped_spokes, 12)
        self.assertEqual(stats_map["FA2"].num_frames, 8)
        self.assertEqual(stats_map["FA2"].dropped_spokes, 0)

    def test_choose_combined_spf_raises_when_no_candidate_survives(self):
        infos = [
            self.make_info("DCE", 23, 2220),
            self.make_info("FA2", 1, 40),
            self.make_info("FA15", 1, 40),
            self.make_info("FA2p", 1, 40),
            self.make_info("FA15p", 1, 40),
        ]

        with self.assertRaisesRegex(ValueError, "No valid combined_spokes_per_frame candidate found"):
            self.common.choose_combined_spokes_per_frame(infos)

    def test_compute_rebin_stats_reports_used_and_dropped_spokes(self):
        stats = self.common.compute_rebin_stats("DCE", original_spf=23, n_spokes=2220, target_spf=16)

        self.assertEqual(stats.num_frames, 138)
        self.assertEqual(stats.used_spokes, 2208)
        self.assertEqual(stats.dropped_spokes, 12)
        self.assertAlmostEqual(stats.dropped_ratio, 12 / 2220)

    def test_require_series_configs_reports_missing_hops(self):
        configs = [{"hop_id": "DCE", "spokes_per_frame": 23}]
        with self.assertRaisesRegex(ValueError, "Missing required series in CSV"):
            self.common.require_series_configs(configs)

    def test_require_basis_path_raises_when_step1_output_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "Step1 basis not found"):
                self.common.require_basis_path(Path(tmpdir), "subj01")

    def test_filter_csv_paths_is_shared_behavior(self):
        csv_paths = [
            Path("subjA_config.csv"),
            Path("subjB_config.csv"),
            Path("subjC_config.csv"),
        ]
        filtered = self.common.filter_csv_paths(csv_paths, ["subjB", "subjC"])
        self.assertEqual(filtered, [Path("subjB_config.csv"), Path("subjC_config.csv")])

    def test_all_entrypoints_accept_subjects_arg(self):
        for module in (self.step1, self.step2, self.batch):
            parser = self.common.build_common_arg_parser("x")
            parsed = parser.parse_args(["--csv-dir", "tmp", "--subjects", "s1", "s2"])
            self.assertEqual(parsed.subjects, ["s1", "s2"])


if __name__ == "__main__":
    unittest.main()
