"""The pluggable-algorithm contract.

``BaseAlgorithm`` is the single interface the CLI and serving layer talk to.
Every computer-vision algorithm (YOLO, U-Net, and any future addition) subclasses
it, registers itself (:mod:`core.registry`) and implements at least model
resolution + segmenter construction. Capabilities a backend does not provide
(e.g. U-Net has no benchmark) raise :class:`NotSupported`, which the CLI renders
as a friendly message instead of a traceback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from core.segmenter import BaseSegmenter


class NotSupported(NotImplementedError):
	"""Raised when an algorithm does not implement a requested capability."""


class BaseAlgorithm(ABC):
	"""Base class for every registered computer-vision algorithm."""

	#: Short identifier used on the CLI (``--algo <name>``) and registry key.
	name: ClassVar[str] = ""
	#: Human-readable name used in messages.
	display_name: ClassVar[str] = ""

	def __init__(
		self,
		*,
		size: int | list[int] | None = None,
		model_id: int | None = None,
		model_dir: str | None = None,
	) -> None:
		self.size = size
		self.model_id = model_id
		self.model_dir = model_dir

	# -- model selection + inference (every backend must support these) ------
	@abstractmethod
	def resolve_model_ref(
		self,
		*,
		model_id: int | None = None,
		model_dir: str | None = None,
	) -> Any:
		"""Resolve a trained-model reference for :meth:`build_segmenter`.

		Unifies trained-model selection across algorithms: an explicit
		``model_dir`` takes priority, otherwise ``model_id`` selects a known run,
		otherwise the backend's sensible default (e.g. the latest run) is used.
		"""

	@abstractmethod
	def build_segmenter(self, model_ref: Any, **kwargs: Any) -> BaseSegmenter:
		"""Construct the backend's :class:`~core.segmenter.BaseSegmenter`."""

	def test(
		self,
		*,
		images_dir: str,
		output_dir: str,
		model_id: int | None = None,
		model_dir: str | None = None,
		keep_classes: list[str] | None = None,
		save_masks: bool = True,
		save_segmented: bool = True,
		**segmenter_kwargs: Any,
	) -> list[str]:
		"""Resolve the model, build a segmenter and run directory inference.

		Implemented once here so no backend re-implements the wiring. ``model_id``
		/ ``model_dir`` fall back to the values supplied at construction time.
		``save_masks`` / ``save_segmented`` let a backend tune which artifacts are
		written (e.g. U-Net suppresses per-instance grayscale masks).
		"""
		ref = self.resolve_model_ref(
			model_id=model_id if model_id is not None else self.model_id,
			model_dir=model_dir if model_dir is not None else self.model_dir,
		)
		segmenter = self.build_segmenter(ref, **segmenter_kwargs)
		return segmenter.segment_directory(
			images_dir,
			output_dir,
			keep_classes=keep_classes,
			save_masks=save_masks,
			save_segmented=save_segmented,
		)

	# -- optional capabilities (default: not supported) ----------------------
	def train(self, **kwargs: Any) -> Any:
		self._unsupported("train")

	def tune(self, **kwargs: Any) -> Any:
		self._unsupported("tune")

	def convert(self, **kwargs: Any) -> Any:
		self._unsupported("convert")

	def benchmark(self, **kwargs: Any) -> Any:
		self._unsupported("benchmark")

	def report(self, **kwargs: Any) -> Any:
		self._unsupported("report")

	def _unsupported(self, action: str) -> Any:
		raise NotSupported(
			f"{self.display_name or self.name} does not support '{action}'."
		)
