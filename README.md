# MGRASP Reconstruction Code

Academic code repository for dynamic MRI reconstruction and vascular segmentation experiments.

## Structure

- `src/mgrasp_recon/`: core Python modules for reconstruction, coil estimation, segmentation, and TIC analysis
- `notebooks/`: exploratory and experiment notebooks
- `results/figures/`: representative figures used for checks or documentation
- `results/lambda_sweeps/`: saved outputs from lambda/basis sweeps

## Quick Start

1. Create an environment and install dependencies from `requirements.txt`.
2. Launch Jupyter from the repository root.
3. Open a notebook under `notebooks/`.

## Notes

- The default environment installs `sigpy` from the fork pinned in `pyproject.toml`.
- If you need to develop `sigpy` itself, work in a separate checkout and install it editable in your local environment.
- Raw input data is not stored in this repository.
