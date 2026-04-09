"""Path helpers for importing project modules during local development."""

from pathlib import Path
import sys


def ensure_repo_paths() -> tuple[Path, Path]:
    package_root = Path(__file__).resolve().parent
    src_root = package_root.parent
    repo_root = src_root.parent

    path_str = str(src_root)
    if src_root.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

    return repo_root, src_root
