"""Interactive dataset download + training wizard for Roboflow projects."""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ── Model registry ──────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
	"YOLOv8-seg": {
		"prefix": "yolov8",
		"format": "yolov8",
		"sizes": [
			("nano",   "n"),
			("small",  "s"),
			("medium", "m"),
			("large",  "l"),
			("xlarge", "x"),
		],
	},
	"YOLOv9-seg": {
		"prefix": "yolov9",
		"format": "yolov9",
		"sizes": [
			("tiny",     "t"),
			("small",    "s"),
			("medium",   "m"),
			("compact",  "c"),
			("enhanced", "e"),
		],
	},
	"YOLOv10-seg": {
		"prefix": "yolov10",
		"format": "yolov10",
		"sizes": [
			("nano",      "n"),
			("small",     "s"),
			("medium",    "m"),
			("balanced",  "b"),
			("large",     "l"),
			("xlarge",    "x"),
		],
	},
	"YOLO11-seg": {
		"prefix": "yolo11",
		"format": "yolo11",
		"sizes": [
			("nano",   "n"),
			("small",  "s"),
			("medium", "m"),
			("large",  "l"),
			("xlarge", "x"),
		],
	},
	"YOLO12-seg": {
		"prefix": "yolo12",
		"format": "yolo12",
		"sizes": [
			("nano",   "n"),
			("small",  "s"),
			("medium", "m"),
			("large",  "l"),
			("xlarge", "x"),
		],
	},
	"YOLO26-seg": {
		"prefix": "yolo26",
		"format": "yolo26",
		"sizes": [
			("nano",   "n"),
			("small",  "s"),
			("medium", "m"),
			("large",  "l"),
			("xlarge", "x"),
		],
	},
}


def add_download_arguments(sp) -> None:
	sp.add_argument(
		"--version",
		type=int,
		default=None,
		help="Roboflow dataset version (skips the prompt when given).",
	)
	sp.add_argument(
		"--model",
		default=None,
		help="Base model weights file, e.g. yolo26s-seg.pt (skips the prompts when given).",
	)
	sp.add_argument(
		"--skip-train",
		action="store_true",
		help="Download the dataset only; skip training.",
	)
	sp.add_argument(
		"--size",
		type=int,
		default=None,
		help="Input image size used for training (default: auto-detect from the dataset images).",
	)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_env() -> None:
	try:
		from dotenv import load_dotenv
		load_dotenv()
	except ImportError:
		pass


def _require_questionary():
	try:
		import questionary
		return questionary
	except ImportError:
		print(
			"❌ questionary is not installed.\n"
			"   Run: uv add questionary  (or pip install questionary)",
			file=sys.stderr,
		)
		sys.exit(1)


def _find_data_yaml(dataset_location: str) -> str | None:
	"""Return the path to data.yaml inside a Roboflow download location."""
	direct = os.path.join(dataset_location, "data.yaml")
	if os.path.exists(direct):
		return direct

	import glob
	candidates = glob.glob(
		os.path.join(dataset_location, "**", "data.yaml"), recursive=True
	)
	return candidates[0] if candidates else None


def _select_model(q) -> str:
	"""Walk the family → size selection and return the weights filename."""
	family_name = q.select(
		"Base model family:",
		choices=list(MODELS.keys()),
		default="YOLO26-seg",
	).ask()
	if family_name is None:
		raise _Cancelled

	family = MODELS[family_name]
	size_labels = [f"{name} ({code})  →  {family['prefix']}{code}-seg.pt" for name, code in family["sizes"]]
	size_choice = q.select("Model size:", choices=size_labels).ask()
	if size_choice is None:
		raise _Cancelled

	size_index = size_labels.index(size_choice)
	size_code = family["sizes"][size_index][1]
	return f"{family['prefix']}{size_code}-seg.pt"


def _format_for_model(model_file: str) -> str:
	"""Map a weights filename (e.g. yolo26s-seg.pt) to its Roboflow export format."""
	name = os.path.basename(model_file).lower()
	# Longest prefix first so 'yolov10' wins over a hypothetical 'yolov1'.
	for family in sorted(MODELS.values(), key=lambda f: -len(f["prefix"])):
		if name.startswith(family["prefix"]):
			return family["format"]
	raise ValueError(
		f"Could not infer a Roboflow format from model '{model_file}'. "
		f"Expected one of: {', '.join(f['prefix'] for f in MODELS.values())}."
	)


class _Cancelled(Exception):
	"""Raised internally when the user aborts a prompt (Ctrl-C / Esc)."""


# ── Main entry point ─────────────────────────────────────────────────────────

def run_download(args) -> int:
	_load_env()

	api_key = os.environ.get("ROBOFLOW_API_KEY")
	if not api_key:
		print(
			"❌ ROBOFLOW_API_KEY is not set.\n"
			"   Add it to the .env file at the repository root.",
			file=sys.stderr,
		)
		return 1

	workspace    = os.environ.get("ROBOFLOW_WORKSPACE", "ashu-biqfs")
	project_name = os.environ.get("ROBOFLOW_PROJECT",   "manga-segment_v2")

	q = _require_questionary()

	print("📚  Roboflow dataset download & training")
	print(f"    workspace: {workspace}")
	print(f"    project:   {project_name}\n")

	try:
		# ── 1. Dataset version ───────────────────────────────────────────
		if args.version is not None:
			version_number = args.version
		else:
			version_str = q.text(
				"Dataset version number:",
				default="5",
				validate=lambda v: (v.strip().isdigit() and int(v) > 0)
				or "Enter a positive integer.",
			).ask()
			if version_str is None:
				return 1
			version_number = int(version_str.strip())

		# ── 2. Base model (family → size) ────────────────────────────────
		if args.model is not None:
			model_file = args.model
			download_format = _format_for_model(model_file)
		else:
			model_file = _select_model(q)
			download_format = _format_for_model(model_file)

		# ── 3. Train after download? ─────────────────────────────────────
		if args.skip_train:
			train_after = False
		else:
			train_after = q.confirm(
				"Start training right after the download?", default=True
			).ask()
			if train_after is None:
				return 1
	except _Cancelled:
		print("✋  Cancelled.")
		return 1

	# ── Confirmation summary ─────────────────────────────────────────────
	print("\n────────────────────────────────────────")
	print(f"  version  : v{version_number}")
	print(f"  format   : {download_format}")
	print(f"  model    : {model_file}")
	print(f"  training : {'yes' if train_after else 'no (download only)'}")
	print("────────────────────────────────────────\n")

	# ── Download into a dedicated per-version folder ─────────────────────
	try:
		from roboflow import Roboflow
	except ImportError:
		print(
			"❌ roboflow is not installed.\n"
			"   Run: uv add roboflow  (or pip install roboflow)",
			file=sys.stderr,
		)
		return 1

	from core import paths

	# Each version/format gets its own folder so downloads never collide with
	# (or get confused for) any pre-existing dataset under dataset/.
	location = os.path.join(
		str(paths.DATASET_DIR), f"{project_name}-v{version_number}-{download_format}"
	)
	os.makedirs(location, exist_ok=True)

	print(f"📦  Downloading dataset v{version_number} ({download_format}) → {location}")

	rf      = Roboflow(api_key=api_key)
	project = rf.workspace(workspace).project(project_name)
	version = project.version(version_number)
	dataset = version.download(download_format, location=location, overwrite=True)

	data_yaml = _find_data_yaml(getattr(dataset, "location", location)) or _find_data_yaml(location)
	if data_yaml is None:
		print(
			f"❌ Download finished but no data.yaml was found under {location}.",
			file=sys.stderr,
		)
		return 1

	print(f"✅  Dataset ready → {data_yaml}\n")

	if not train_after:
		print("ℹ️  Skipping training (download only).")
		return 0

	# ── Train ─────────────────────────────────────────────────────────────
	print("🚀  Starting training …\n")

	import algorithms

	algorithms.load_all()
	from core.registry import get_algorithm

	algo = get_algorithm("yolo", size=args.size)
	algo.model_name = model_file
	algo.data_path  = data_yaml
	algo.train()

	return 0
