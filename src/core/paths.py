"""Project paths, resolved from this file rather than the current directory.

Every default location (trained models, input images, output, dataset) is
derived from the repository root so commands work regardless of the working
directory. ``src/`` lives directly under the repo root, so the root is two
parents up from this module (``src/core/paths.py`` -> ``src`` -> repo root).
"""

from __future__ import annotations

from pathlib import Path

# src/core/paths.py -> src/core -> src -> <repo root>
PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parent

MODELS_DIR = REPO_ROOT / "models"
IMAGES_DIR = REPO_ROOT / "images"
OUTPUT_DIR = REPO_ROOT / "output"
DATASET_DIR = REPO_ROOT / "dataset"
RUNS_DIR = REPO_ROOT / "runs" / "segment"


def models_dir(*parts: str) -> str:
	"""Absolute path inside the shared ``models`` directory."""
	return str(MODELS_DIR.joinpath(*parts))


def images_dir() -> str:
	"""Default directory of images to run inference over."""
	return str(IMAGES_DIR)


def output_dir() -> str:
	"""Default directory where inference results are written."""
	return str(OUTPUT_DIR)


def dataset_path(*parts: str) -> str:
	"""Absolute path inside the shared ``dataset`` directory."""
	return str(DATASET_DIR.joinpath(*parts))


def runs_dir() -> str:
	"""Default Ultralytics ``runs/segment`` directory."""
	return str(RUNS_DIR)
