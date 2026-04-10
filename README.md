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

## Usage

- To run subject on server with nohup (standard output (print statements) and standard error messages will be redirected to a file named nohup.out)

    ```bash
    nohup ./.venv/bin/python ./notebooks/step2_recon_all_slices.py
    ```

- to stop the nohup process, first check the PID, then kill
    
    ```bash
    ps aux | grep step2_recon_all_slices.py
    kill <PID>
    ```