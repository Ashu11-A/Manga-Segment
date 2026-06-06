"""A tiny registry that makes the framework extensible.

Each algorithm class decorates itself with :func:`register`; importing the
``algorithms`` package then imports every backend so the registry is populated.
The CLI and serving layer look algorithms up by name, so adding a new
computer-vision algorithm needs no changes here — only a new registered class.
"""

from __future__ import annotations

from typing import Any

from core.algorithm import BaseAlgorithm

_REGISTRY: dict[str, type[BaseAlgorithm]] = {}


def register(cls: type[BaseAlgorithm]) -> type[BaseAlgorithm]:
	"""Class decorator: register an algorithm under its ``name``."""
	if not cls.name:
		raise ValueError(f"{cls.__name__} must define a non-empty 'name'.")
	if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
		raise ValueError(f"Algorithm '{cls.name}' is already registered.")
	_REGISTRY[cls.name] = cls
	return cls


def available() -> list[str]:
	"""Sorted names of all registered algorithms."""
	return sorted(_REGISTRY)


def get_algorithm(name: str, **kwargs: Any) -> BaseAlgorithm:
	"""Instantiate a registered algorithm by name.

	``kwargs`` are forwarded to the algorithm constructor (``size``,
	``model_id``, ``model_dir`` — see :class:`~core.algorithm.BaseAlgorithm`).
	"""
	try:
		cls = _REGISTRY[name]
	except KeyError:
		raise KeyError(
			f"Unknown algorithm '{name}'. Available: {', '.join(available()) or '(none)'}."
		) from None
	return cls(**kwargs)
