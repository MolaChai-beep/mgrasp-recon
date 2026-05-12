# visualize_step1_basis_comparison.py

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 和 step1_basis_test_multi_low_res.py 保持同一套根目录逻辑
REPO_ROOT = Path("/home/naiqianluan/DCE-MRI/code/mgrasp-recon")
OUTPUT_ROOT = REPO_ROOT / "outputs" / "step1_basis_test_multi_low_res"

SUBJECT_ID = "Gross_MeyerA"
HOP_ID = "DCE"

HOP_DIR = OUTPUT_ROOT / SUBJECT_ID / HOP_ID
LOWRES_DIRS = [
    HOP_DIR / "lowres_128x128",
    HOP_DIR / "lowres_192x192",
    HOP_DIR / "lowres_256x256",
]


def save_figure(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_similarity_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    matrix = np.array([[float(v) for v in row[1:]] for row in rows[1:]], dtype=np.float64)
    return names, matrix


def load_summary_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_overlay_curves(shape_dirs, out_dir):
    fig_sv, axes_sv = plt.subplots(1, 2, figsize=(12, 4))
    fig_ev, axes_ev = plt.subplots(1, 2, figsize=(12, 4))

    for shape_dir in shape_dirs:
        shape_tag = shape_dir.name

        vascular_path = shape_dir / "vascular_singular_values.npy"
        tissue_path = shape_dir / "tissue_singular_values.npy"

        print(f"checking {shape_tag}")
        print("  ", vascular_path)
        print("  ", tissue_path)

        if not vascular_path.exists():
            print(f"missing: {vascular_path}")
            continue
        if not tissue_path.exists():
            print(f"missing: {tissue_path}")
            continue

        vascular_s = np.load(vascular_path)
        tissue_s = np.load(tissue_path)

        vascular_evr = (vascular_s ** 2) / np.sum(vascular_s ** 2)
        tissue_evr = (tissue_s ** 2) / np.sum(tissue_s ** 2)

        xv = np.arange(1, len(vascular_s) + 1)
        xt = np.arange(1, len(tissue_s) + 1)

        axes_sv[0].plot(xv, vascular_s, "o-", label=shape_tag)
        axes_sv[1].plot(xt, tissue_s, "o-", label=shape_tag)

        axes_ev[0].plot(xv, np.cumsum(vascular_evr), "o-", label=shape_tag)
        axes_ev[1].plot(xt, np.cumsum(tissue_evr), "o-", label=shape_tag)

    axes_sv[0].set_title("Vascular singular values")
    axes_sv[1].set_title("Tissue singular values")
    axes_ev[0].set_title("Vascular cumulative explained variance")
    axes_ev[1].set_title("Tissue cumulative explained variance")

    for ax in [*axes_sv, *axes_ev]:
        ax.set_xlabel("Component")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes_sv[0].set_ylabel("Singular value")
    axes_sv[1].set_ylabel("Singular value")
    axes_ev[0].set_ylabel("Cumulative variance ratio")
    axes_ev[1].set_ylabel("Cumulative variance ratio")
    axes_ev[0].set_ylim(0, 1.02)
    axes_ev[1].set_ylim(0, 1.02)

    save_figure(fig_sv, out_dir / "overlay_singular_values.png")
    save_figure(fig_ev, out_dir / "overlay_cumulative_explained_variance.png")


def plot_summary_bars(summary_rows, out_dir):
    shape_tags = [row["shape_tag"] for row in summary_rows]
    vascular_c1 = [float(row["vascular_cumvar_1"]) for row in summary_rows]
    vascular_c3 = [float(row["vascular_cumvar_3"]) for row in summary_rows]
    tissue_c1 = [float(row["tissue_cumvar_1"]) for row in summary_rows]
    tissue_c3 = [float(row["tissue_cumvar_3"]) for row in summary_rows]

    x = np.arange(len(shape_tags))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(x - width / 2, vascular_c1, width, label="Top 1")
    axes[0].bar(x + width / 2, vascular_c3, width, label="Top 3")
    axes[0].set_title("Vascular cumulative variance")

    axes[1].bar(x - width / 2, tissue_c1, width, label="Top 1")
    axes[1].bar(x + width / 2, tissue_c3, width, label="Top 3")
    axes[1].set_title("Tissue cumulative variance")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(shape_tags, rotation=20, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Variance ratio")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    save_figure(fig, out_dir / "summary_cumulative_variance_bars.png")


def plot_similarity_heatmap(names, matrix, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_path)


def main():
    print("reading hop dir:", HOP_DIR)

    if not HOP_DIR.exists():
        raise FileNotFoundError(f"hop dir not found: {HOP_DIR}")

    shape_dirs = [path for path in LOWRES_DIRS if path.exists()]
    if not shape_dirs:
        raise FileNotFoundError(f"no lowres dirs found under: {HOP_DIR}")

    print("found lowres dirs:")
    for path in shape_dirs:
        print("  ", path)

    vis_dir = HOP_DIR / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    plot_overlay_curves(shape_dirs, vis_dir)

    summary_csv = HOP_DIR / "basis_comparison_summary.csv"
    if summary_csv.exists():
        summary_rows = load_summary_csv(summary_csv)
        plot_summary_bars(summary_rows, vis_dir)
    else:
        print("missing:", summary_csv)

    vascular_csv = HOP_DIR / "vascular_subspace_similarity.csv"
    if vascular_csv.exists():
        names, matrix = load_similarity_csv(vascular_csv)
        plot_similarity_heatmap(
            names,
            matrix,
            "Vascular subspace similarity",
            vis_dir / "vascular_subspace_similarity_heatmap.png",
        )
    else:
        print("missing:", vascular_csv)

    tissue_csv = HOP_DIR / "tissue_subspace_similarity.csv"
    if tissue_csv.exists():
        names, matrix = load_similarity_csv(tissue_csv)
        plot_similarity_heatmap(
            names,
            matrix,
            "Tissue subspace similarity",
            vis_dir / "tissue_subspace_similarity_heatmap.png",
        )
    else:
        print("missing:", tissue_csv)

    print("saved visualizations to:", vis_dir)


if __name__ == "__main__":
    main()
