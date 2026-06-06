"""Automatic discovery of the training dataset under ``dataset/``.

Roboflow exports (and the download wizard) drop each dataset into its own
folder containing a ``data.yaml``. Rather than hard-coding a single path, the
resolver discovers every ``data.yaml`` under ``dataset/`` and:

* uses it directly when exactly one exists;
* asks the user to pick (arrow-key menu) when several exist;
* raises a friendly error when none exist.

This keeps ``train``/``tune``/``benchmark`` working no matter what the dataset
folder is named, while staying non-interactive-safe (a clear error instead of a
hang when stdin is not a terminal).
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from glob import glob

from core import paths


def discover_data_yamls() -> list[str]:
	"""Return every ``data.yaml`` found under the ``dataset/`` directory, sorted."""
	root = str(paths.DATASET_DIR)
	matches = glob(os.path.join(root, "**", "data.yaml"), recursive=True)
	return sorted(matches)


def _label(data_yaml: str) -> str:
	"""Human-friendly label: the dataset folder name relative to ``dataset/``."""
	root = str(paths.DATASET_DIR)
	folder = os.path.dirname(data_yaml)
	rel = os.path.relpath(folder, root)
	return rel if rel != "." else os.path.basename(folder)


def resolve_data_yaml(explicit: str | None = None) -> str:
	"""Resolve the path to the dataset ``data.yaml`` to train on.

	``explicit`` (when given and existing) wins. Otherwise the single dataset
	under ``dataset/`` is used automatically, or the user is asked to choose
	between several. Raises ``FileNotFoundError`` when none can be found.
	"""
	if explicit:
		if os.path.isfile(explicit):
			return explicit
		raise FileNotFoundError(f"data.yaml not found at: {explicit}")

	candidates = discover_data_yamls()

	if not candidates:
		raise FileNotFoundError(
			f"No dataset found under {paths.DATASET_DIR}.\n"
			"   Download one first:  uv run src/main.py download"
		)

	if len(candidates) == 1:
		print(f"📂  Using dataset: {_label(candidates[0])}")
		return candidates[0]

	# Several datasets → let the user pick (interactive), or fail clearly.
	if not sys.stdin.isatty():
		listing = "\n".join(f"     - {_label(c)}" for c in candidates)
		raise FileNotFoundError(
			"Multiple datasets found under dataset/ and stdin is not a terminal.\n"
			"   Pass one explicitly via the algorithm's data path, or run "
			"interactively.\n"
			f"   Available:\n{listing}"
		)

	try:
		import questionary
	except ImportError:
		listing = "\n".join(f"     - {_label(c)}" for c in candidates)
		raise FileNotFoundError(
			"Multiple datasets found and 'questionary' is not installed to choose "
			"between them.\n"
			"   Run: uv add questionary\n"
			f"   Available:\n{listing}"
		)

	choice = questionary.select(
		"Multiple datasets found — choose one to train on:",
		choices=[questionary.Choice(title=_label(c), value=c) for c in candidates],
	).ask()

	if choice is None:
		raise FileNotFoundError("No dataset selected.")

	return choice


# ---------------------------------------------------------------------------
# Automatic image-size detection
# ---------------------------------------------------------------------------
def _resolve_split_dir(data: dict, data_yaml: str, split: str) -> str | None:
	"""Locate the images directory for a dataset split (``train``/``val``/``test``).

	Roboflow exports point each split at a path like ``../train/images`` that is
	resolved against Ultralytics' ``datasets_dir`` rather than the ``data.yaml``
	location, so a naive join fails. We therefore try a few sensible bases and a
	``../``-stripped variant and return the first directory that exists.
	"""
	rel = data.get(split)
	if not rel:
		return None

	yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
	base = data.get("path")
	if base and not os.path.isabs(base):
		base = os.path.normpath(os.path.join(yaml_dir, base))

	stripped = re.sub(r"^(?:\.\./)+", "", rel)  # drop leading ../ segments
	for candidate_base in (b for b in (base, yaml_dir) if b):
		for candidate_rel in (rel, stripped):
			candidate = (
				candidate_rel
				if os.path.isabs(candidate_rel)
				else os.path.normpath(os.path.join(candidate_base, candidate_rel))
			)
			if os.path.isdir(candidate):
				return candidate
	return None


def _image_dimensions(path: str) -> tuple[int, int] | None:
	"""Return ``(height, width)`` of an image, reading only its header."""
	try:
		from PIL import Image

		with Image.open(path) as image:
			width, height = image.size
			return height, width
	except Exception:
		# Fall back to OpenCV (decodes fully) if Pillow can't read the header.
		from core import imaging

		image = imaging.read_image_bgr(path)
		if image is None:
			return None
		return image.shape[0], image.shape[1]


def detect_image_size(
	data_yaml: str,
	*,
	multiple: int = 32,
	max_samples: int = 50,
) -> int | list[int]:
	"""Infer the input image size to train at from the dataset's own images.

	Reads the dataset ``data.yaml``, locates the training images (falling back to
	the val/test splits), samples up to ``max_samples`` of them and returns the
	most common resolution: an ``int`` when the images are square, otherwise
	``[height, width]``. Each dimension is rounded up to the nearest ``multiple``
	(YOLO requires the input size to be a multiple of the max stride, 32).
	"""
	import yaml

	from core.imaging import IMAGE_EXTS

	with open(data_yaml) as handle:
		data = yaml.safe_load(handle) or {}

	image_dir = None
	for split in ("train", "val", "test"):
		image_dir = _resolve_split_dir(data, data_yaml, split)
		if image_dir:
			break

	if not image_dir:
		raise FileNotFoundError(
			f"Could not locate any image directory for {data_yaml} to detect the "
			"image size. Pass --size explicitly."
		)

	dimensions: Counter[tuple[int, int]] = Counter()
	for root, _dirs, files in os.walk(image_dir):
		for name in sorted(files):
			if not name.lower().endswith(IMAGE_EXTS):
				continue
			dims = _image_dimensions(os.path.join(root, name))
			if dims is not None:
				dimensions[dims] += 1
			if sum(dimensions.values()) >= max_samples:
				break
		if sum(dimensions.values()) >= max_samples:
			break

	if not dimensions:
		raise FileNotFoundError(
			f"No readable images found under {image_dir} to detect the image size. "
			"Pass --size explicitly."
		)

	if len(dimensions) > 1:
		print(
			f"ℹ️  Dataset has mixed image sizes {dict(dimensions)}; "
			"using the most common one."
		)

	(height, width), _count = dimensions.most_common(1)[0]

	def round_up(value: int) -> int:
		return int(math.ceil(value / multiple) * multiple)

	height, width = round_up(height), round_up(width)
	return height if height == width else [height, width]
