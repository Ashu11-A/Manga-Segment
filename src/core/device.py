"""Compute-device helpers shared by every algorithm.

Centralises the ``"cuda" if torch.cuda.is_available() else "cpu"`` logic that
was previously duplicated across the YOLO model, the benchmark worker and the
inference pipeline. ``torch`` is imported lazily so importing the framework
never pulls in the heavy dependency until a device is actually requested.
"""

from __future__ import annotations


def cuda_device_count() -> int:
	"""Number of visible CUDA devices (0 when CUDA is unavailable)."""
	try:
		import torch

		return torch.cuda.device_count() if torch.cuda.is_available() else 0
	except Exception:
		return 0


def resolve_device(device: str | None = None) -> str:
	"""Return an explicit device string, defaulting to CUDA when available.

	``device`` is returned untouched when provided, so callers can force a
	specific device (e.g. ``"cpu"`` or ``"cuda:1"``).
	"""
	if device:
		return device
	return "cuda:0" if cuda_device_count() > 0 else "cpu"


def supports_half(device: str) -> bool:
	"""Whether half precision is valid for ``device`` (CUDA only).

	Enabling ``half=True`` on CPU raises in Ultralytics/torch, so inference code
	must gate it on this.
	"""
	return device.startswith("cuda")
