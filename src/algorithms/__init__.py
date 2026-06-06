"""Algorithm plug-ins.

Backends live in sub-packages here (``algorithms/<name>/``), each exposing a
registered :class:`~core.algorithm.BaseAlgorithm`. Discovery is explicit and
lazy: call :func:`load_all` once at startup (the CLI and serving layer do) to
import every sub-package so it self-registers. Adding a new computer-vision
algorithm therefore needs no edits here — just drop in a new sub-package.

Importing this package is intentionally cheap (no backend/torch/tf imports), so
``load_all`` is the single place that triggers heavy registration.
"""

from __future__ import annotations

import importlib
import pkgutil

_loaded = False


def load_all() -> None:
	"""Import every algorithm sub-package so each registers itself (idempotent)."""
	global _loaded
	if _loaded:
		return
	for module in pkgutil.iter_modules(__path__):
		if module.ispkg and not module.name.startswith("_"):
			importlib.import_module(f"{__name__}.{module.name}")
	_loaded = True


__all__ = ["load_all"]
