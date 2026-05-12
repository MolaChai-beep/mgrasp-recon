"""Time-intensity curve helpers."""

from __future__ import annotations

import numpy as np

from .config import SegmentationResult
from .visualization import plot_basis_curves as _plot_basis_curves
from .visualization import plot_multi_resolution_tics as _plot_multi_resolution_tics
from .visualization import plot_time_intensity_curves as _plot_time_intensity_curves
from .visualization import show_selected_voxels as _show_selected_voxels


class TicAnalyzer:
    """Small analysis helper for extracting and plotting voxel TICs."""

    @staticmethod
    def _normalize_curve(curve: np.ndarray) -> np.ndarray:
        baseline = float(np.mean(curve[: min(5, len(curve))]))
        return curve / max(baseline, 1e-6)

    @staticmethod
    def _validate_series(img_dyn) -> np.ndarray:
        x = np.abs(np.asarray(img_dyn))
        if x.ndim != 3:
            raise ValueError(f"img_dyn must have shape (T, H, W), got {x.shape}")
        return x

    @staticmethod
    def _validate_mask(mask: np.ndarray, image_shape: tuple[int, int], name: str) -> np.ndarray:
        out = np.asarray(mask).astype(bool)
        if out.shape != image_shape:
            raise ValueError(f"{name} mask shape {out.shape} does not match image shape {image_shape}")
        if not np.any(out):
            raise ValueError(f"{name} mask is empty.")
        return out

    @staticmethod
    def _moving_average(curve: np.ndarray, window: int) -> np.ndarray:
        window = max(1, min(int(window), len(curve)))
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(curve, kernel, mode="same")

    def extract_voxel_tic(self, img_dyn, coord, normalize=False):
        x = self._validate_series(img_dyn)

        row, col = map(int, coord)
        if not (0 <= row < x.shape[1] and 0 <= col < x.shape[2]):
            raise IndexError(f"coord {coord} is outside image bounds {x.shape[1:]}")

        curve = x[:, row, col].astype(np.float32, copy=False)
        if normalize:
            curve = self._normalize_curve(curve)
        return curve

    def extract_roi_mean_tic(self, img_dyn, roi_mask, normalize=False):
        x = self._validate_series(img_dyn)
        mask = self._validate_mask(roi_mask, x.shape[1:], "ROI")
        curve = x[:, mask].mean(axis=1).astype(np.float32, copy=False)
        if normalize:
            curve = self._normalize_curve(curve)
        return curve

    def select_representative_voxel(self, img_dyn, roi_mask):
        x = self._validate_series(img_dyn)
        mask = self._validate_mask(roi_mask, x.shape[1:], "ROI")
        coords = np.argwhere(mask)
        scores = x[:, mask].max(axis=0)
        return tuple(map(int, coords[int(np.argmax(scores))]))

    def summarize_roi_tics(
        self,
        img_dyn,
        vascular_mask,
        tissue_mask,
        vessel_coord=None,
        normalize=False,
    ):
        x = self._validate_series(img_dyn)
        vascular_mask = self._validate_mask(vascular_mask, x.shape[1:], "vascular")
        tissue_mask = self._validate_mask(tissue_mask, x.shape[1:], "tissue")
        if np.any(vascular_mask & tissue_mask):
            raise ValueError("vascular_mask and tissue_mask must not overlap.")

        if vessel_coord is None:
            vessel_coord = self.select_representative_voxel(x, vascular_mask)

        vessel_single = self.extract_voxel_tic(x, vessel_coord, normalize=normalize)
        vessel_roi = self.extract_roi_mean_tic(x, vascular_mask, normalize=normalize)
        tissue_roi = self.extract_roi_mean_tic(x, tissue_mask, normalize=normalize)
        return {
            "vessel_coord": tuple(map(int, vessel_coord)),
            "vessel_single": vessel_single,
            "vessel_roi": vessel_roi,
            "tissue_roi": tissue_roi,
            "vascular_mask_size": int(np.sum(vascular_mask)),
            "tissue_mask_size": int(np.sum(tissue_mask)),
        }

    def curve_metrics(self, curve, smooth_window=5):
        curve = np.asarray(curve, dtype=np.float32)
        baseline = float(np.mean(curve[: min(5, len(curve))]))
        peak = float(np.max(curve))
        residual = curve - self._moving_average(curve, smooth_window)
        return {
            "baseline": baseline,
            "peak": peak,
            "peak_enhancement": float((peak - baseline) / max(baseline, 1e-6)),
            "wash_in_max": float(np.max(np.diff(curve))) if len(curve) > 1 else 0.0,
            "std": float(np.std(curve)),
            "high_freq_std": float(np.std(residual)),
        }

    def summarize_noise_metrics(
        self,
        img_dyn,
        vascular_mask,
        tissue_mask,
        vessel_coord=None,
        normalize=False,
    ):
        curves = self.summarize_roi_tics(
            img_dyn,
            vascular_mask=vascular_mask,
            tissue_mask=tissue_mask,
            vessel_coord=vessel_coord,
            normalize=normalize,
        )
        return {
            "vessel_coord": curves["vessel_coord"],
            "vascular_mask_size": curves["vascular_mask_size"],
            "tissue_mask_size": curves["tissue_mask_size"],
            "single_voxel": self.curve_metrics(curves["vessel_single"]),
            "vessel_roi": self.curve_metrics(curves["vessel_roi"]),
            "tissue_roi": self.curve_metrics(curves["tissue_roi"]),
        }

    def compare_resolution_tics(self, curves_by_label):
        labels = list(curves_by_label)
        if len(labels) < 2:
            return {}

        metrics = {}
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                lc = np.asarray(curves_by_label[left], dtype=np.float32)
                rc = np.asarray(curves_by_label[right], dtype=np.float32)
                if lc.shape != rc.shape:
                    raise ValueError(f"Curve shape mismatch for {left} vs {right}: {lc.shape} vs {rc.shape}")
                if len(lc) <= 1:
                    corr = 1.0
                elif np.std(lc) < 1e-8 and np.std(rc) < 1e-8:
                    corr = 1.0
                elif np.std(lc) < 1e-8 or np.std(rc) < 1e-8:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(lc, rc)[0, 1])
                metrics[f"{left}__vs__{right}"] = corr
        return metrics

    def summarize_resolution_comparison(
        self,
        series_by_label,
        normalize=False,
    ):
        summaries = {}
        vessel_roi_curves = {}
        tissue_roi_curves = {}
        for label, payload in series_by_label.items():
            if isinstance(payload, dict):
                img_dyn = payload["img_dyn"]
                vascular_mask = payload["vascular_mask"]
                tissue_mask = payload["tissue_mask"]
                vessel_coord = payload.get("vessel_coord")
            else:
                img_dyn = payload.img_dyn
                segmentation = payload.segmentation
                vascular_mask = segmentation.vascular_mask
                tissue_mask = segmentation.tissue_mask
                vessel_coord = None

            summary = self.summarize_roi_tics(
                img_dyn,
                vascular_mask=vascular_mask,
                tissue_mask=tissue_mask,
                vessel_coord=vessel_coord,
                normalize=normalize,
            )
            summaries[label] = summary
            vessel_roi_curves[label] = summary["vessel_roi"]
            tissue_roi_curves[label] = summary["tissue_roi"]

        return {
            "summaries": summaries,
            "vessel_roi_correlations": self.compare_resolution_tics(vessel_roi_curves),
            "tissue_roi_correlations": self.compare_resolution_tics(tissue_roi_curves),
        }

    def plot_time_intensity_curves(
        self,
        img_dyn,
        vessel_coord,
        tissue_coords=None,
        frame_time_sec=None,
        normalize=False,
        title="Time-intensity curves",
    ):
        x = self._validate_series(img_dyn)

        tissue_coords = [] if tissue_coords is None else list(tissue_coords)
        vessel_curve = self.extract_voxel_tic(x, vessel_coord, normalize=normalize)
        tissue_curves = [self.extract_voxel_tic(x, coord, normalize=normalize) for coord in tissue_coords]
        return _plot_time_intensity_curves(
            vessel_curve=vessel_curve,
            vessel_coord=tuple(map(int, vessel_coord)),
            tissue_curves=tissue_curves,
            tissue_coords=[tuple(map(int, coord)) for coord in tissue_coords],
            frame_time_sec=frame_time_sec,
            normalize=normalize,
            title=title,
        )

    def show_selected_voxels(
        self,
        background_img,
        vessel_coord,
        tissue_coords=None,
        vascular_mask=None,
        tissue_mask=None,
        title="Selected voxels",
    ):
        tissue_coords = [] if tissue_coords is None else list(tissue_coords)
        return _show_selected_voxels(
            background_img=np.asarray(background_img),
            vessel_coord=tuple(map(int, vessel_coord)),
            tissue_coords=[tuple(map(int, coord)) for coord in tissue_coords],
            vascular_mask=vascular_mask,
            tissue_mask=tissue_mask,
            title=title,
        )

    def plot_roi_tic_summary(
        self,
        img_dyn,
        vascular_mask,
        tissue_mask,
        vessel_coord=None,
        frame_time_sec=None,
        normalize=False,
        title="Single voxel and ROI TICs",
    ):
        summary = self.summarize_roi_tics(
            img_dyn,
            vascular_mask=vascular_mask,
            tissue_mask=tissue_mask,
            vessel_coord=vessel_coord,
            normalize=normalize,
        )
        curves = [
            ("Vessel single voxel", summary["vessel_single"], {"linewidth": 2.4, "color": "crimson"}),
            ("Vessel ROI mean", summary["vessel_roi"], {"linewidth": 2.4, "color": "darkorange"}),
            ("Normal brain ROI mean", summary["tissue_roi"], {"linewidth": 2.2, "color": "royalblue", "linestyle": "--"}),
        ]
        return _plot_time_intensity_curves(
            named_curves=curves,
            frame_time_sec=frame_time_sec,
            normalize=normalize,
            title=title,
        )

    def plot_roi_overlay(
        self,
        background_img,
        vascular_mask,
        tissue_mask,
        vessel_coord=None,
        title="Vessel and normal brain ROIs",
    ):
        return _show_selected_voxels(
            background_img=np.asarray(background_img),
            vessel_coord=None if vessel_coord is None else tuple(map(int, vessel_coord)),
            tissue_coords=None,
            vascular_mask=vascular_mask,
            tissue_mask=tissue_mask,
            title=title,
        )

    def plot_basis_curves(self, basis, title="Basis curves", labels=None):
        return _plot_basis_curves(basis=basis, title=title, labels=labels)

    def plot_resolution_comparison(
        self,
        curves_by_label,
        frame_time_sec=None,
        normalize=False,
        title="Vessel ROI TIC comparison across resolutions",
    ):
        return _plot_multi_resolution_tics(
            curves_by_label=curves_by_label,
            frame_time_sec=frame_time_sec,
            normalize=normalize,
            title=title,
        )

    def create_noise_report(
        self,
        img_dyn,
        segmentation: SegmentationResult,
        basis=None,
        vessel_coord=None,
        frame_time_sec=None,
        normalize=False,
        title_prefix="Noise check",
    ):
        summary = self.summarize_roi_tics(
            img_dyn,
            vascular_mask=segmentation.vascular_mask,
            tissue_mask=segmentation.tissue_mask,
            vessel_coord=vessel_coord,
            normalize=normalize,
        )
        metrics = self.summarize_noise_metrics(
            img_dyn,
            vascular_mask=segmentation.vascular_mask,
            tissue_mask=segmentation.tissue_mask,
            vessel_coord=summary["vessel_coord"],
            normalize=normalize,
        )

        figures = {
            "roi_overlay": self.plot_roi_overlay(
                background_img=segmentation.mean_img,
                vascular_mask=segmentation.vascular_mask,
                tissue_mask=segmentation.tissue_mask,
                vessel_coord=summary["vessel_coord"],
                title=f"{title_prefix}: vessel and normal brain ROIs",
            ),
            "tic_summary": self.plot_roi_tic_summary(
                img_dyn=img_dyn,
                vascular_mask=segmentation.vascular_mask,
                tissue_mask=segmentation.tissue_mask,
                vessel_coord=summary["vessel_coord"],
                frame_time_sec=frame_time_sec,
                normalize=normalize,
                title=f"{title_prefix}: single voxel vs ROI TICs",
            ),
        }
        if basis is not None:
            figures["basis_curves"] = self.plot_basis_curves(
                basis=basis,
                title=f"{title_prefix}: basis curves",
            )

        return {
            "summary": summary,
            "metrics": metrics,
            "figures": figures,
        }
