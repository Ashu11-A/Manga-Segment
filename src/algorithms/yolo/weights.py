"""Trained-checkpoint resolution for the YOLO backend.

A single resolver replacing the legacy ``inference.py`` resolvers and
``utils.py:getModel``: it turns an explicit ``model_dir`` / ``.pt`` file, a
``model_id`` run selector, or "latest run" into the path of the ``.pt`` file
itself, fixing the old inconsistency where a directory and a file path were
returned interchangeably.
"""

from __future__ import annotations

import os
from glob import glob

from core import paths


def weights_from_dir(model_dir: str, weight: str = "best.pt") -> str:
	"""Resolve a checkpoint from an explicit model directory or ``.pt`` file.

	Accepts a direct path to a ``.pt`` file, a run directory containing
	``weights/<weight>`` (e.g. ``models/yolo``), or a directory containing the
	``.pt`` file directly (e.g. ``models/yolo/weights``).
	"""
	path = os.path.abspath(os.path.expanduser(model_dir))

	if os.path.isfile(path):
		return path

	candidates = [
		os.path.join(path, "weights", weight),
		os.path.join(path, weight),
	]
	for candidate in candidates:
		if os.path.isfile(candidate):
			return candidate

	raise FileNotFoundError(
		f"No '{weight}' checkpoint found in '{model_dir}'. "
		f"Looked for a .pt file or {candidates}."
	)


def resolve_weights(
	model_id: int | None = None,
	model_dir: str | None = None,
	runs_dir: str | None = None,
	weight: str = "best.pt",
) -> str:
	"""Resolve the path to a trained checkpoint.

	``model_dir`` (when given) takes priority and points directly at a trained
	model (``.pt`` file, a run dir like ``models/yolo``, or its ``weights`` dir).
	Otherwise the checkpoint is resolved inside ``runs/segment``: ``model_id``
	selects ``train<id>`` (``0`` → the un-suffixed ``train`` folder) and ``None``
	picks the most recent ``train*`` run. Always returns the path to the ``.pt``
	file itself.
	"""
	if runs_dir is None:
		runs_dir = paths.runs_dir()

	if model_dir is not None:
		return weights_from_dir(model_dir, weight)

	if model_id is None:
		runs = sorted(glob(os.path.join(runs_dir, "train*")))
		if not runs:
			raise FileNotFoundError(f"No 'train*' runs found in {runs_dir}.")
		run_dir = runs[-1]
	else:
		suffix = "" if model_id == 0 else str(model_id)
		run_dir = os.path.join(runs_dir, f"train{suffix}")

	weights_path = os.path.join(run_dir, "weights", weight)
	if not os.path.isfile(weights_path):
		raise FileNotFoundError(f"Weights not found: {weights_path}")
	return weights_path
